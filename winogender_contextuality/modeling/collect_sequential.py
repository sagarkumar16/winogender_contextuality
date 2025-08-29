import os
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

# TODO: Batching currently not available for two pronoun generation
@app.command()
def generate_two_pronouns(
        mode: str,
        model_name: str,
        temperature: float,
        n_runs: int = 100,
        quantized: bool = True,
        input_fpath: pathlib.Path = INTERIM_DATA_DIR / "wino_pairs.tsv",
        output_dir: pathlib.Path = INTERIM_DATA_DIR
):

    """

    :param input_fpath:
    :param output_fpath: ndjson file
    :param mode:
    :param model_name:
    :param temperature:
    :param n_runs:
    :return:
    """

    logger.add(LOG_DIR / f"data_collection_{datetime.now()}.log")
    output_fpath = output_dir / f"two_pronouns_measurements_{model_name.split('/')[-1]}_{temperature}.ndjson"
    
    df = pd.read_csv(input_fpath, sep="\t")
    mp = ModelProbs(
        mode=mode,
        model_name=model_name,
        key=HF_KEY,
        model_path=MODELS_DIR,
        quantized=quantized)

    mp.load_model()

    pbar = tqdm(df.index, desc="Collecting sequential pronouns completions for two pronouns.")

    for idx in pbar:
        measurements_idx = []
        # Unpacking CSV -- hard coding for now
        sentences = {0: df.template_1[idx],
                     1: df.template_2[idx]}
        pronouns = {0: ast.literal_eval(df.differences_1[idx]),
                    1: ast.literal_eval(df.differences_2[idx])}

        

        
        for s_perm in permutations(sentences.keys()):

            s1 = sentences[s_perm[0]]
            s2 = sentences[s_perm[1]]
            p1 = pronouns[s_perm[0]]
            p2 = pronouns[s_perm[1]]

            for i, p1_perm in enumerate(permutations(p1)):
                for j, p2_perm in enumerate(permutations(p2)):
                    
                    error_count = 0
                    for n in tqdm(range(n_runs)):

                        s_list = [s1, s2]
                        p_list = [list(p1_perm), list(p2_perm)]
    
                        prompt = role_content_base(*no_game_seq_prompt(p_list, s_list))
    
                        inputs, output = mp.get_completion(prompt=prompt, temperature=temperature, max_new_tokens=12)
    
                        input_len = inputs.shape[1]
    
                        decoded_output = mp.tokenizer.decode(output.sequences[0][input_len - 5:],
                                                             skip_special_tokens=True)
    
                        try:
                            json_output = ast.literal_eval(decoded_output)
    
                            c = Context(sent_order=s_perm,
                                        pnoun_order=(i,j),
                                        sentence_1=s1,
                                        sentence_2=s2,
                                        pronouns_1=p1,
                                        pronouns_2=p2)

                            probs = None

                            try:
                                
                                first_token, first_logits, second_token, second_logits = mp.pronoun_logits(
                                        pronouns_list=p_list,
                                        generated_sequence = output.sequences[0][input_len:],
                                        scores = output.scores
                                    )

                                first_ids = mp.tokenizer(p1[1]).input_ids
                                second_ids = mp.tokenizer(p2[1]).input_ids

                                first_fem_pnoun_token = first_ids[-1]
                                second_fem_pnoun_token = second_ids[-1]

                                first_fem_prob = masked_softmax(first_fem_pnoun_token, first_logits[0])
                                second_fem_prob = masked_softmax(second_fem_pnoun_token, second_logits[0])

                                probs = (first_fem_prob.item(), second_fem_prob.item())

                            except Exception as e:
                                logger.error(f"Probability extraction failed: {e}")

                            m = Measurement(index=idx, context=c, measurement=json_output, probabilities=probs,
                                            logits=None)
                            measurements_idx.append(m)
                            with open(output_fpath, "a") as f:
                                f.write(json.dumps(asdict(m))+"\n")
                                
                        except Exception as e:
                            error_count += 1
                            logger.warning(f"Error {e} for output: {decoded_output}. Error count {error_count}/{n}")

                    logger.warning(f"{error_count}/{n_runs} not captured.")
        logger.info(f"Successfully collected {idx}. "
                    f"Writing {len(measurements_idx)} measurements to {output_fpath}.")

    return


# TODO: Add capability to extract additional pronouns
@app.command()
def generate_one_pronoun(
        mode: str,
        model_name: str,
        temperature: float,
        n_runs: int = 1000,
        quantized: bool = True,
        input_fpath: pathlib.Path = INTERIM_DATA_DIR / "wino_pairs.tsv",
        output_dir: pathlib.Path = INTERIM_DATA_DIR,
        start: int = 0,
        end: int | None = None,                 # inclusive
        output_file: pathlib.Path | None = None # NEW: single, shared output file
):
    """
    Collect single-pronoun logits and completions over a slice of the dataset.

    :param mode:
    :param model_name:
    :param temperature:
    :param n_runs:
    :param quantized:
    :param input_fpath:
    :param output_dir: ndjson file
    :param start: 0-based start index (inclusive)
    :param end: end index (inclusive); if None, processes through last row
    :param output_file: explicit NDJSON file path shared by all batches
    """

    logger.add(LOG_DIR / f"data_collection_{datetime.now()}.log")

    # single shared output file
    if output_file is None:
        output_fpath = output_dir / f"one_pronoun_measurements_{model_name.split('/')[-1]}_{temperature}.ndjson"
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
                            prompt = role_content_base(*no_game_seq_logit_prompt(
                                option_set=p_list,
                                free_sentence=s2,
                                fixed_sentence=first_sentence_filled
                            ))

                            model_logits = mp.get_raw_logits(prompt=prompt).cpu()
                            pronoun_idxs = mp.get_token_ids(options=pronouns[1])
                            pronoun_logits = model_logits[[sum(pronoun_idxs, [])]].tolist()

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





if __name__ == "__main__":
    app()

