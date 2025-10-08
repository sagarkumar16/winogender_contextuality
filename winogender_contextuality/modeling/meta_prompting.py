# %%
import os
from itertools import permutations
import pathlib
from loguru import logger
from datetime import datetime
import tqdm
import winogender_contextuality.modeling.prompting as pr
from winogender_contextuality.utils import *
from winogender_contextuality.config import *
from winogender_contextuality.modeling.ModelProbs import *

# %%

HF_KEY = os.environ.get("HF_KEY")

def get_question(q:str,
                 blank_role: str,
                 other_role: str):

    """

    :param q:
    :param blank_role:
    :param other_role:
    :return:
    """

    if q == 'anaphora':

        question = ("Answer saying who the pronoun replaced by BLANK is referring to. "
                    "Select from one of the following options: "
                    f"[{blank_role}, {other_role}].")

    if q == "pos":
        question = ("Answer saying what part of speech the BLANK should be. "
                    "Select from one of the following options: "
                    "[noun, verb, pronoun, adjective, adverb, preposition, article].")

    if q == "other_gender":
        question = (f"Answer saying the gender of the {other_role}. "
                    "Select from one of the following options: "
                    "[male, female, nonbinary].")
    
    return question


def get_answers(q: str,
                priming_pnoun: str,
                blank_role: str):
    """

    :param q:
    :param priming_pnoun_idx:
    :param blank_role:
    :return:
    """

    ## need genders for all of them
    ## need to both roles
    ## can only run other_gender where there is a priming sentence

    if q == 'anaphora':
        return blank_role

    if q == 'pos':
        return 'pronoun'

    if q == 'other_gender':
        if priming_pnoun in pronoun_genders.get('male'):
            return 'male'
        elif priming_pnoun in pronoun_genders.get('female'):
            return 'female'
        elif priming_pnoun in pronoun_genders.get('nb'):
            return 'nonbinary'
        else:
            raise AttributeError("Invalid pronoun.")

# TODO: how is this reconfigured to work for metaprompting?
def metaprompt_template(option_set: list[str],
                        free_sentence: str,
                        fixed_sentence: None | str,
                        question: str,
                        blank_role: str,
                        other_role: str):
    """
    Outputs a zero-shot user prompt which needs to be fit into a template. This prompt should only be used to probe model state.

    :param option_set: A LIST of pronoun options. MUST be ordered to measure contextuality.
    :param free_sentence: Sentence with a token BLANK in it.
    :param fixed_sentence: Additional sentence to test the effect of adding context.
    :return: user prompt
    """

    SYSTEM_PROMPT = ("Below you will find a passage in *bold* which contains precisely one instance of "
                     "the term BLANK. "
                     "After the passage, two possible options will be provided to replace the BLANK."
                     "After being presented the options, you will be asked to answer a question."
                     "The task is designed to be unambiguous, so please read the passage and question carefully, "
                     "provide only one answer, and do not reorder the data.")

    if fixed_sentence:
        sentence = fixed_sentence + " " + free_sentence
    else:
        sentence = free_sentence


    USER_PROMPT = (f"Given this passage: *{sentence}*\n" 
                   f"One of two options may be chose to replace the BLANK: {option_set}"
                   "Given this information, please do the following: "
                   f"{get_question(question, blank_role, other_role)} "
                   "Respond only in the following format {'ANSWER': '<text>'}")

    ASSISTANT_PROMPT = "{'ANSWER':'"

    return SYSTEM_PROMPT, USER_PROMPT, ASSISTANT_PROMPT

@app.command()
def run_metaprompting(
        mode: str,
        model_name: str,
        temperature: float,
        questions: list[str] | None = None,
        quantized: bool = True,
        input_dir: pathlib.Path = INTERIM_DATA_DIR ,
        output_dir: pathlib.Path = INTERIM_DATA_DIR,
        start: int = 0,
        end: int | None = None,                 # inclusive
        input_file: str = "wino_pairs.tsv",
        output_file: pathlib.Path | None = None # single, shared output file
):

    logger.add(LOG_DIR / f"metaprompting_{datetime.now()}.log")

    input_fpath = input_dir / input_file

    # single shared output file
    if output_file is None:
        fname_model = model_name.split('/')[-1]
        dt = datetime.now().strftime('%H%M%d%m%y')
        output_fpath = output_dir / f"metaprompting_{fname_model}_{temperature}_{dt}.ndjson"
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

    if questions is None:
        questions = ['anaphora', 'pos', 'other_gender']

    role_dict = get_role_dict()
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

    pbar = tqdm(indices, desc=f"Metaprompting [{start}-{(end if end is not None else n_rows-1)}]")

    for idx in pbar:
        measurements_idx = []

        sentences = {0: df.template_1[idx], 1: df.template_2[idx]}
        pronouns = {0: ast.literal_eval(df.differences_1[idx]),
                    1: ast.literal_eval(df.differences_2[idx])}

        orientations = ['forward', 'reverse']

        for direction, s_perm in zip(orientations, permutations(sentences.keys())):
            s1 = sentences[s_perm[0]]
            s2 = sentences[s_perm[1]]
            p1 = pronouns[s_perm[0]]
            p2 = pronouns[s_perm[1]]

            first_sentences = [None, s1]

            blank_role, other_role = role_dict.get(idx).get(direction)

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

                        p_list = list(p2_perm)

                        ## TODO: get question-answer pairs and output a df

                        ## TODO: Turn this into metaprompt
                        for question in questions:
                            prompt = pr.role_content_base(*metaprompt_template(
                                option_set=p_list,
                                free_sentence=s2,
                                fixed_sentence=first_sentence_filled,
                                question=question,
                                blank_role=blank_role,
                                other_role=other_role
                            ))

                            inputs, output = mp.get_completion(
                                prompt=prompt,
                                temperature=temperature,
                                max_new_tokens=12
                            )

                            input_len = inputs.shape[1]
                            decoded_output = mp.tokenizer.decode(
                                output.sequences[0][input_len - 6:],
                                skip_special_tokens=True
                            )

                            try:
                                json_output = ast.literal_eval(decoded_output)
                            except Exception as e:
                                error_count += 1
                                json_output = {'ANSWER': 'None'}
                                logger.warning(f"Error {e} for output: {decoded_output}. Error count {error_count}")

                            qa = MetaQA(
                                index=idx,
                                question=question,
                                response=json_output['ANSWER'],
                                answer=get_answers(q=question, priming_pnoun=first_pronoun, blank_role=blank_role)
                            )

                            measurements_idx.append(idx)

                            # Append one-by-one to the single shared file
                            with open(output_fpath, "a") as f:
                                _lock_file(f)
                                f.write(json.dumps(asdict(qa)) + "\n")
                                f.flush()
                                os.fsync(f.fileno())
                                _unlock_file(f)

                            if error_count > 0:
                                logger.warning(f"{error_count} errors.")

        logger.info(f"Successfully collected {idx}. Writing {len(measurements_idx)} measurements to {output_fpath}.")




