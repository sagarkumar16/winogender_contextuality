import os
import re
import json
import pathlib
import pandas as pd
import ast
import typer
from datetime import datetime
from itertools import permutations
from winogender_contextuality.modeling.prompting import *
from winogender_contextuality.modeling.ModelProbs import *
from winogender_contextuality.config import *
from winogender_contextuality.utils import *


app = typer.Typer()

HF_KEY = os.environ.get("HF_KEY")

JOINT_KEYS = ("BLANK1", "BLANK2")

# Option orders for (BLANK1, BLANK2); 0 = male-first. 'matched' is the paper's two conditions.
JOINT_OPTION_ORDERS = {
    "matched": [(0, 0), (1, 1)],
    "all": [(0, 0), (0, 1), (1, 0), (1, 1)],
}


def parse_joint_completion(text: str) -> dict:
    """
    Parses a two-blank completion into {'BLANK1': ..., 'BLANK2': ...}.

    Takes the first {...} in the text, so a chat-turn header before the answer or trailing text
    after it does not matter. Anything unparseable becomes {'BLANK1': 'None', 'BLANK2': 'None'},
    which get_index() filters out.

    :param text: decoded completion, including the assistant prefill if one was used
    :return: dict with keys BLANK1 and BLANK2
    """
    match = re.search(r"\{[^{}]*\}", text)
    try:
        parsed = ast.literal_eval(match.group(0))
        if isinstance(parsed, dict) and all(k in parsed for k in JOINT_KEYS):
            return {k: str(parsed[k]).strip() for k in JOINT_KEYS}
    except Exception:
        pass
    return {k: "None" for k in JOINT_KEYS}


@app.command()
def generate_two_pronouns(
        mode: str,
        model_name: str,
        temperature: float,
        n_runs: int = 50,
        quantized: bool = True,
        assistant: bool = True,
        option_orders: str = "matched",
        input_dir: pathlib.Path = INTERIM_DATA_DIR,
        output_dir: pathlib.Path = INTERIM_DATA_DIR,
        start: int = 0,
        end: int | None = None,                 # inclusive
        input_file: str = "wino_pairs.tsv",
        output_file: pathlib.Path | None = None # single, shared output file
):
    """
    Collect JOINT measurements: both pronouns of a pair generated in a single completion.

    The two sentences are shown as one passage, BLANK1 in the sentence presented first and BLANK2
    in the second, and the model fills both. The contexts are the two passage orders, so in
    sent_order (1, 0) BLANK1 belongs to template_2. Unlike generate_one_pronoun, neither pronoun
    is fixed by the experimenter.

    :param mode:
    :param model_name:
    :param temperature:
    :param n_runs: sampled completions per (pair, sentence order, option order)
    :param quantized:
    :param assistant: whether to prefill the assistant turn with JOINT_ASSISTANT_PREFILL
    :param option_orders: 'matched' presents both blanks' options male-first, then both
                          female-first (the paper's two conditions); 'all' adds the mixed orders
    :param input_dir:
    :param output_dir: ndjson file
    :param start: 0-based start index (inclusive)
    :param end: end index (inclusive); if None, processes through last row
    :param input_file: filename from input_dir
    :param output_file: explicit NDJSON file path shared by all batches
    """

    if option_orders not in JOINT_OPTION_ORDERS:
        raise ValueError(f"option_orders must be one of {list(JOINT_OPTION_ORDERS)}, got {option_orders!r}")

    logger.add(LOG_DIR / f"data_collection_{datetime.now()}.log")

    input_fpath = input_dir / input_file

    # single shared output file
    if output_file is None:
        fname_model = model_name.split('/')[-1]
        dt = datetime.now().strftime('%H%M%d%m%y')
        output_fpath = output_dir / f"joint_measurements_{fname_model}_{temperature}_{dt}.ndjson"
    else:
        output_fpath = output_file
    output_fpath.parent.mkdir(parents=True, exist_ok=True)

    # file lock for parallel batches
    try:
        import fcntl
        def _lock_file(fh): fcntl.flock(fh, fcntl.LOCK_EX)
        def _unlock_file(fh): fcntl.flock(fh, fcntl.LOCK_UN)
    except Exception:
        def _lock_file(fh): pass
        def _unlock_file(fh): pass

    df = pd.read_csv(input_fpath, sep="\t")

    # Normalize slice
    n_rows = len(df)
    start = max(0, start)
    end_exclusive = n_rows if (end is None or end >= n_rows) else end + 1
    if start >= end_exclusive:
        logger.warning(f"No rows to process: start={start}, end={end}. n_rows={n_rows}.")
        return

    indices = df.index[start:end_exclusive]

    mp = ModelProbs(
        mode=mode,
        model_name=model_name,
        key=HF_KEY,
        model_path=MODELS_DIR,
        quantized=quantized
    )
    mp.load_model()

    pbar = tqdm(indices, desc=f"Collecting Joint Pronouns [{start}-{(end if end is not None else n_rows-1)}]")

    for idx in pbar:
        measurements_idx = []

        sentences = {0: df.template_1[idx], 1: df.template_2[idx]}
        pronouns = {0: ast.literal_eval(df.differences_1[idx]),
                    1: ast.literal_eval(df.differences_2[idx])}
        cases = {0: df.case_1[idx], 1: df.case_2[idx]}

        for s_perm in permutations(sentences.keys()):
            s1 = sentences[s_perm[0]]
            s2 = sentences[s_perm[1]]
            p1 = pronouns[s_perm[0]]
            p2 = pronouns[s_perm[1]]
            case1 = cases[s_perm[0]]
            case2 = cases[s_perm[1]]

            for i, j in JOINT_OPTION_ORDERS[option_orders]:
                p_list = [p1 if i == 0 else p1[::-1], p2 if j == 0 else p2[::-1]]
                prompt = role_content_base(*no_game_seq_prompt(p_list, [s1, s2], assistant=assistant))

                error_count = 0
                for _ in tqdm(range(n_runs)):
                    inputs, output = mp.get_completion(
                        prompt=prompt,
                        temperature=temperature,
                        max_new_tokens=24,
                        continue_final_message=assistant
                    )

                    # Decode only the generated tokens and put the prefill back in front, rather
                    # than slicing a fixed number of prompt tokens (which depends on the tokenizer).
                    generated = mp.tokenizer.decode(output.sequences[0][inputs.shape[1]:],
                                                    skip_special_tokens=True)
                    decoded_output = (JOINT_ASSISTANT_PREFILL + generated) if assistant else generated
                    json_output = parse_joint_completion(decoded_output)
                    if json_output["BLANK1"] == "None":
                        error_count += 1
                        logger.warning(f"Could not parse output: {decoded_output!r}. Error count {error_count}")

                    c = Context(
                        sent_order=s_perm,
                        pnoun_order=(i, j),
                        sentence_1=s1,
                        sentence_2=s2,
                        pronouns_1=p1,   # canonical [male, female] order, as in the other collectors
                        pronouns_2=p2,
                        case_1=case1,
                        case_2=case2
                    )

                    m = Measurement(
                        index=idx,
                        context=c,
                        measurement=json_output,
                        probabilities=None,
                        logits=None
                    )

                    measurements_idx.append(m)

                    # Append one-by-one to the single shared file
                    with open(output_fpath, "a") as f:
                        _lock_file(f)
                        f.write(json.dumps(asdict(m)) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                        _unlock_file(f)

                if error_count > 0:
                    logger.warning(f"{error_count}/{n_runs} not captured.")

        logger.info(f"Successfully collected {idx}. Writing {len(measurements_idx)} measurements to {output_fpath}.")


# TODO: Add capability to extract additional pronouns
@app.command()
def generate_one_pronoun(
        mode: str,
        model_name: str,
        temperature: float,
        n_runs: int = 1000,
        quantized: bool = True,
        assistant: bool = True,
        input_dir: pathlib.Path = INTERIM_DATA_DIR ,
        output_dir: pathlib.Path = INTERIM_DATA_DIR,
        start: int = 0,
        end: int | None = None,                 # inclusive
        input_file: str = "wino_pairs.tsv",
        output_file: pathlib.Path | None = None # single, shared output file
):
    """
    Collect single-pronoun logits and completions over a slice of the dataset.

    :param mode:
    :param model_name:
    :param temperature:
    :param n_runs:
    :param quantized:
    :param assistant:
    :param input_dir:
    :param output_dir: ndjson file
    :param start: 0-based start index (inclusive)
    :param end: end index (inclusive); if None, processes through last row
    :param input_file: filename from input_dir
    :param output_file: explicit NDJSON file path shared by all batches
    """

    logger.add(LOG_DIR / f"data_collection_{datetime.now()}.log")

    input_fpath = input_dir / input_file

    # single shared output file
    if output_file is None:
        fname_model = model_name.split('/')[-1]
        dt = datetime.now().strftime('%H%M%d%m%y')
        output_fpath = output_dir / f"one_pronoun_measurements_{fname_model}_{temperature}_{dt}.ndjson"
    else:
        output_fpath = output_file
    output_fpath.parent.mkdir(parents=True, exist_ok=True)

    # file lock for parallel batches
    try:
        import fcntl
        def _lock_file(fh): fcntl.flock(fh, fcntl.LOCK_EX)
        def _unlock_file(fh): fcntl.flock(fh, fcntl.LOCK_UN)
    except Exception:
        def _lock_file(fh): pass
        def _unlock_file(fh): pass

    df = pd.read_csv(input_fpath, sep="\t")

    # Normalize slice
    n_rows = len(df)
    start = max(0, start)
    end_exclusive = n_rows if (end is None or end >= n_rows) else end + 1
    if start >= end_exclusive:
        logger.warning(f"No rows to process: start={start}, end={end}. n_rows={n_rows}.")
        return

    indices = df.index[start:end_exclusive]

    mp = ModelProbs(
        mode=mode,
        model_name=model_name,
        key=HF_KEY,
        model_path=MODELS_DIR,
        quantized=quantized
    )
    mp.load_model()

    pbar = tqdm(indices, desc=f"Collecting Single Pronoun Logits [{start}-{(end if end is not None else n_rows-1)}]")

    for idx in pbar:
        measurements_idx = []

        sentences = {0: df.template_1[idx], 1: df.template_2[idx]}
        pronouns = {0: ast.literal_eval(df.differences_1[idx]),
                    1: ast.literal_eval(df.differences_2[idx])}
        cases = {0: df.case_1[idx], 1: df.case_2[idx]}

        for s_perm in permutations(sentences.keys()):
            s1 = sentences[s_perm[0]]
            s2 = sentences[s_perm[1]]
            p1 = pronouns[s_perm[0]]
            p2 = pronouns[s_perm[1]]
            case1 = cases[s_perm[0]]
            case2 = cases[s_perm[1]]

            first_sentences = [None, s1]

            for j, p2_perm in enumerate(permutations(p2)):
                error_count = 0
                for first_sentence in first_sentences:
                    for pnoun in p1:
                        if first_sentence is not None:
                            first_pronoun = pnoun
                            first_sentence_filled = first_sentence.replace('BLANK', first_pronoun)
                        else:
                            first_pronoun = None
                            first_sentence_filled = None

                        for _ in tqdm(range(n_runs)):
                            p_list = list(p2_perm)

                            prompt = role_content_base(*no_game_seq_logit_prompt(
                                option_set=p_list,
                                free_sentence=s2,
                                fixed_sentence=first_sentence_filled,
                                assistant=assistant
                            ))

                            model_logits = mp.get_raw_logits(prompt=prompt).cpu()
                            # Options of the FREE sentence (s2), in canonical [male, female] order.
                            # This used to be pronouns[1] (always template_2's options), which is
                            # only right when both sentences of a pair share a grammatical case.
                            pronoun_idxs = mp.get_token_ids(options=p2)
                            pronoun_logits = model_logits[sum(pronoun_idxs, [])].tolist()

                            inputs, output = mp.get_completion(
                                prompt=prompt,
                                temperature=temperature,
                                max_new_tokens=12,
                                continue_final_message=assistant
                            )

                            input_len = inputs.shape[1]

                            if "gemma" in model_name:
                                decoded_output_full = mp.tokenizer.decode(
                                    output.sequences[0],
                                    skip_special_tokens=True
                                )
                                try:
                                    decoded_output = decoded_output_full.split("model")[-1]
                                except Exception as e:
                                    error_count += 1
                                    json_output = {'BLANK': 'None'}
                                    logger.warning(
                                        f"Error {e} for output: {decoded_output_full}. Error count {error_count}")
                                    decoded_output = "{'BLANK': 'None'}"
                            else:
                                decoded_output = mp.tokenizer.decode(
                                    output.sequences[0][input_len - 5:],
                                    skip_special_tokens=True
                                )

                            try:
                                json_output = ast.literal_eval(decoded_output)
                            except Exception as e:
                                error_count += 1
                                json_output = {'BLANK': 'None'}
                                logger.warning(f"Error {e} for output: {decoded_output}. Error count {error_count}")

                            c = Context(
                                sent_order=s_perm,
                                pnoun_order=(first_pronoun, j),
                                sentence_1=first_sentence_filled,
                                sentence_2=s2,
                                pronouns_1=p1,
                                pronouns_2=p2, 
                                case_1 = case1,
                                case_2 = case2
                            )

                            m = Measurement(
                                index=idx,
                                context=c,
                                measurement=json_output,
                                probabilities=None,
                                logits=pronoun_logits
                            )

                            measurements_idx.append(m)

                            # Append one-by-one to the single shared file
                            with open(output_fpath, "a") as f:
                                _lock_file(f)
                                f.write(json.dumps(asdict(m)) + "\n")
                                f.flush()
                                os.fsync(f.fileno())
                                _unlock_file(f)

                            if error_count > 0:
                                logger.warning(f"{error_count}/{n_runs} not captured.")

        logger.info(f"Successfully collected {idx}. Writing {len(measurements_idx)} measurements to {output_fpath}.")

@app.command()
def generate_one_pronoun_noprompt(
        mode: str,
        model_name: str,
        temperature: float,
        n_runs: int = 1000,
        quantized: bool = True,
        input_dir: pathlib.Path = INTERIM_DATA_DIR ,
        output_dir: pathlib.Path = INTERIM_DATA_DIR,
        start: int = 0,
        end: int | None = None,                 # inclusive
        input_file: str = "wino_pairs.tsv",
        output_file: pathlib.Path | None = None # single, shared output file
):
    """
    Collect single-pronoun logits and completions over a slice of the dataset.
    Use the prompt which specifies to ignore ordering effects.

    :param mode:
    :param model_name:
    :param temperature:
    :param n_runs:
    :param quantized:
    :param input_dir:
    :param output_dir: ndjson file
    :param start: 0-based start index (inclusive)
    :param end: end index (inclusive); if None, processes through last row
    :param input_file: filename from input_dir
    :param output_file: explicit NDJSON file path shared by all batches
    """

    logger.add(LOG_DIR / f"data_collection_{datetime.now()}.log")

    input_fpath = input_dir / input_file

    # single shared output file
    if output_file is None:
        fname_model = model_name.split('/')[-1]
        dt = datetime.now().strftime('%H%M%d%m%y')
        output_fpath = output_dir / f"one_pronoun_measurements_{fname_model}_{temperature}_{dt}.ndjson"
    else:
        output_fpath = output_file
    output_fpath.parent.mkdir(parents=True, exist_ok=True)

    # file lock for parallel batches
    try:
        import fcntl
        def _lock_file(fh): fcntl.flock(fh, fcntl.LOCK_EX)
        def _unlock_file(fh): fcntl.flock(fh, fcntl.LOCK_UN)
    except Exception:
        def _lock_file(fh): pass
        def _unlock_file(fh): pass

    df = pd.read_csv(input_fpath, sep="\t")

    # Normalize slice
    n_rows = len(df)
    start = max(0, start)
    end_exclusive = n_rows if (end is None or end >= n_rows) else end + 1
    if start >= end_exclusive:
        logger.warning(f"No rows to process: start={start}, end={end}. n_rows={n_rows}.")
        return

    indices = df.index[start:end_exclusive]

    mp = ModelProbs(
        mode=mode,
        model_name=model_name,
        key=HF_KEY,
        model_path=MODELS_DIR,
        quantized=quantized
    )
    mp.load_model()

    pbar = tqdm(indices, desc=f"Collecting Single Pronoun Logits [{start}-{(end if end is not None else n_rows-1)}]")

    for idx in pbar:
        measurements_idx = []

        sentences = {0: df.template_1[idx], 1: df.template_2[idx]}
        pronouns = {0: ast.literal_eval(df.differences_1[idx]),
                    1: ast.literal_eval(df.differences_2[idx])}

        for s_perm in permutations(sentences.keys()):
            s1 = sentences[s_perm[0]]
            s2 = sentences[s_perm[1]]
            p1 = pronouns[s_perm[0]]
            p2 = pronouns[s_perm[1]]

            first_sentences = [None, s1]

            for j, p2_perm in enumerate(permutations(p2)):
                error_count = 0
                for first_sentence in first_sentences:
                    for pnoun in p1:
                        if first_sentence is not None:
                            first_pronoun = pnoun
                            first_sentence_filled = first_sentence.replace('BLANK', first_pronoun)
                        else:
                            first_pronoun = None
                            first_sentence_filled = None

                        for _ in tqdm(range(n_runs)):
                            p_list = list(p2_perm)
                            prompt = role_content_base(*no_game_seq_norder_prompt(
                                option_set=p_list,
                                free_sentence=s2,
                                fixed_sentence=first_sentence_filled
                            ))

                            model_logits = mp.get_raw_logits(prompt=prompt).cpu()
                            # Options of the FREE sentence (s2), in canonical [male, female] order.
                            pronoun_idxs = mp.get_token_ids(options=p2)
                            pronoun_logits = model_logits[sum(pronoun_idxs, [])].tolist()

                            inputs, output = mp.get_completion(
                                prompt=prompt,
                                temperature=temperature,
                                max_new_tokens=12
                            )

                            input_len = inputs.shape[1]
                            decoded_output = mp.tokenizer.decode(
                                output.sequences[0][input_len - 5:],
                                skip_special_tokens=True
                            )

                            try:
                                json_output = ast.literal_eval(decoded_output)
                            except Exception as e:
                                error_count += 1
                                json_output = {'BLANK': 'None'}
                                logger.warning(f"Error {e} for output: {decoded_output}. Error count {error_count}")

                            c = Context(
                                sent_order=s_perm,
                                pnoun_order=(first_pronoun, j),
                                sentence_1=first_sentence_filled,
                                sentence_2=s2,
                                pronouns_1=p1,
                                pronouns_2=p2
                            )

                            m = Measurement(
                                index=idx,
                                context=c,
                                measurement=json_output,
                                probabilities=None,
                                logits=pronoun_logits
                            )

                            measurements_idx.append(m)

                            # Append one-by-one to the single shared file
                            with open(output_fpath, "a") as f:
                                _lock_file(f)
                                f.write(json.dumps(asdict(m)) + "\n")
                                f.flush()
                                os.fsync(f.fileno())
                                _unlock_file(f)

                            if error_count > 0:
                                logger.warning(f"{error_count}/{n_runs} not captured.")

        logger.info(f"Successfully collected {idx}. Writing {len(measurements_idx)} measurements to {output_fpath}.")

@app.command()
def generate_one_null_context(mode: str,
                              model_name: str,
                              temperature: float,
                              n_runs: int = 1000,
                              quantized: bool = True,
                              assistant = True,
                              input_dir: pathlib.Path = INTERIM_DATA_DIR ,
                              output_dir: pathlib.Path = INTERIM_DATA_DIR,
                              start: int = 0,
                              end: int | None = None,                 # inclusive
                              input_file: str = "all_sentences.csv",
                              output_file: pathlib.Path | None = None, # single, shared output file
                              prime_column: str | None = None,
                              tag: str = "null"
                              ):
    """
    Collect single-pronoun logits and completions for each sentence under a non-pronoun prime.

    By default every sentence is run against each of the generic primes in
    config.null_sentences. With prime_column set, each row is instead run against its own prime,
    read from that column of the input file (e.g. the referent-only primes built by
    revisions/build_refnull_inputs.py).

    :param prime_column: input column holding a per-row prime; None uses config.null_sentences
    :param tag: label for the prime condition in the output: sent_order is [f"{tag}_{i}", 0],
                and the default output file name starts with f"{tag}_measurements"
    """

    logger.add(LOG_DIR / f"data_collection_{datetime.now()}.log")

    input_fpath = input_dir / input_file

    # single shared output file
    if output_file is None:
        fname_model = model_name.split('/')[-1]
        dt = datetime.now().strftime('%H%M%d%m%y')
        output_fpath = output_dir / f"{tag}_measurements_{fname_model}_{temperature}_{dt}.ndjson"
    else:
        output_fpath = output_file
    output_fpath.parent.mkdir(parents=True, exist_ok=True)

    # file lock for parallel batches
    try:
        import fcntl
        def _lock_file(fh):
            fcntl.flock(fh, fcntl.LOCK_EX)
        def _unlock_file(fh):
            fcntl.flock(fh, fcntl.LOCK_UN)
    except Exception:
        def _lock_file(fh):
            pass
        def _unlock_file(fh):
            pass

    df = pd.read_csv(input_fpath)

    # Normalize slice
    n_rows = len(df)
    start = max(0, start)
    end_exclusive = n_rows if (end is None or end >= n_rows) else end + 1
    if start >= end_exclusive:
        logger.warning(f"No rows to process: start={start}, end={end}. n_rows={n_rows}.")
        return

    indices = df.index[start:end_exclusive]

    mp = ModelProbs(
        mode=mode,
        model_name=model_name,
        key=HF_KEY,
        model_path=MODELS_DIR,
        quantized=quantized
    )
    mp.load_model()

    pbar = tqdm(indices, desc=f"Collecting Null Context Generation"
                              f" [{start}-{(end if end is not None else n_rows - 1)}]")

    for idx in pbar:
        measurements_idx = []
        pronouns = ast.literal_eval(df.differences[idx])
        sentence = df.template[idx]
        case = df.case[idx]

        primes = [df[prime_column][idx]] if prime_column else null_sentences

        for i,prime in enumerate(primes):
            pronoun_lists = [pronouns, pronouns[::-1]]
            for n in range(2):
                error_count = 0
                p_list = pronoun_lists[n]
                for _ in tqdm(range(n_runs)):
                    prompt = role_content_base(*no_game_seq_logit_prompt(
                        option_set=p_list,
                        free_sentence=sentence,
                        fixed_sentence=prime,
                        assistant=assistant,
                    ))

                    model_logits = mp.get_raw_logits(prompt=prompt).cpu()
                    # Here `pronouns` is already the option LIST (e.g. ['he', 'she']), so it is
                    # passed whole. This used to be pronouns[1] -- the string 'she' -- which
                    # tokenised the characters 's', 'h', 'e', so null runs collected before this
                    # fix store 3 logits over those characters and have no usable internal probs.
                    pronoun_idxs = mp.get_token_ids(options=pronouns)
                    pronoun_logits = model_logits[sum(pronoun_idxs, [])].tolist()

                    inputs, output = mp.get_completion(
                        prompt=prompt,
                        temperature=temperature,
                        max_new_tokens=12,
                        continue_final_message=assistant
                    )

                    input_len = inputs.shape[1]

                    if "gemma" in model_name:
                        decoded_output_full = mp.tokenizer.decode(
                            output.sequences[0],
                            skip_special_tokens=True
                        )
                        try:
                            decoded_output = decoded_output_full.split("model")[-1]
                        except Exception as e:
                            error_count += 1
                            json_output = {'BLANK': 'None'}
                            logger.warning(
                                f"Error {e} for output: {decoded_output_full}. Error count {error_count}")
                            decoded_output = "{'BLANK': 'None'}"
                    else:
                        decoded_output = mp.tokenizer.decode(
                            output.sequences[0][input_len - 5:],
                            skip_special_tokens=True
                        )

                    try:
                        json_output = ast.literal_eval(decoded_output)
                    except Exception as e:
                        error_count += 1
                        json_output = {'BLANK': 'None'}
                        logger.warning(f"Error {e} for output: {decoded_output}. Error count {error_count}")

                    c = Context(
                        sent_order=[f"{tag}_{i}", 0],
                        pnoun_order=(tag, n),
                        sentence_1=prime,
                        sentence_2=sentence,
                        pronouns_1=tag,
                        pronouns_2=p_list,
                        case_1=tag,
                        case_2=case
                    )

                    m = Measurement(
                        index=idx,
                        context=c,
                        measurement=json_output,
                        probabilities=None,
                        logits=pronoun_logits
                    )

                    measurements_idx.append(m)

                    # Append one-by-one to the single shared file
                    with open(output_fpath, "a") as f:
                        _lock_file(f)
                        f.write(json.dumps(asdict(m)) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                        _unlock_file(f)

                    if error_count > 0:
                        logger.warning(f"{error_count}/{n_runs} not captured.")

        logger.info(f"Successfully collected {idx}. Writing {len(measurements_idx)} measurements to {output_fpath}.")

    return



if __name__ == "__main__":
    app()

