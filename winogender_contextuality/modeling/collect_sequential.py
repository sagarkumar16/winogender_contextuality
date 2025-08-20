import os
import json
import pathlib
import pandas as pd
import ast
import typer
from datetime import datetime
from dataclasses import dataclass, asdict
from itertools import permutations
from winogender_contextuality.modeling.prompting import *
from winogender_contextuality.modeling.ModelProbs import *
from winogender_contextuality.config import *
from winogender_contextuality.utils import *

app = typer.Typer()

@dataclass
class Context:
    sent_order: tuple[int, int] # denotes the order of the sentences (e.g. (1, 0) => the order was reversed)
    pnoun_order: tuple[int, int] # denotes the order of the pronouns (e.g. (0, 1) => second set of pronouns was reversed)
    sentence_1: str
    sentence_2: str
    pronouns_1: list[str]
    pronouns_2: list[str]

@dataclass
class Measurement:
    index: int 
    context: Context
    measurement: dict[str, str]
    probabilities: tuple[float] | None
    logits: tuple[float]

HF_KEY = os.environ.get("HF_KEY")

@app.command()
def generate_two_pronouns(
        mode: str,
        model_name: str,
        temperature: float,
        n_runs: int = 1000,
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
    output_fpath = output_dir / f"measurements_{model_name.split('/')[-1]}_{temperature}.ndjson"
    
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
    
                        decoded_output = mp.tokenizer.decode(output.sequences[0][input_len - 5:], skip_special_tokens=True)
    
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

                            m = Measurement(index=idx, context=c, measurement=json_output, probabilities=probs)
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
        output_dir: pathlib.Path = INTERIM_DATA_DIR
):

    """


    :param mode:
    :param model_name:
    :param temperature:
    :param n_runs:
    :param quantized:
    :param input_fpath:
    :param output_dir:
    :return:
    """

    logger.add(LOG_DIR / f"data_collection_{datetime.now()}.log")
    output_fpath = output_dir / f"measurements_{model_name.split('/')[-1]}_{temperature}.ndjson"
    logits = []

    df = pd.read_csv(input_fpath, sep="\t")
    mp = ModelProbs(
        mode=mode,
        model_name=model_name,
        key=HF_KEY,
        model_path=MODELS_DIR,
        quantized=quantized)

    mp.load_model()

    pbar = tqdm(df.index, desc="Collecting Single Pronoun Logits and Completions")

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

            first_sentences = [None, s1]

            for j, p2_perm in enumerate(permutations(p2)):
                error_count = 0
                for first_sentence in first_sentences:
                    if first_sentence is not None:
                        for first_pronoun in p1:
                            first_sentence = first_sentence.replace('BLANK', first_pronoun)

                            for n in tqdm(range(n_runs)):
                                p_list = list(p2_perm)
                                prompt = role_content_base(*no_game_seq_logit_prompt(option_set=p_list,
                                                                    free_sentence=s2,
                                                                    fixed_sentence=first_sentence))

                                # Model Logits
                                model_logits = mp.get_raw_logits(prompt=prompt)
                                pronoun_idxs = mp.get_token_ids(options=pronouns[1])
                                pronoun_logits = model_logits[pronoun_idxs].cpu().numpy()

                                inputs, output = mp.get_completion(prompt=prompt, temperature=temperature,
                                                                   max_new_tokens=12)

                                # Generation Logits
                                input_len = inputs.shape[1]
                                decoded_output = mp.tokenizer.decode(output.sequences[0][input_len - 5:],
                                                                     skip_special_tokens=True)

                                try:
                                    json_output = ast.literal_eval(decoded_output)

                                    c = Context(sent_order=s_perm,
                                                pnoun_order=j,
                                                sentence_1=s1,
                                                sentence_2=s2,
                                                pronouns_1=p1,
                                                pronouns_2=p2)

                                    probs = None

                                    try:

                                        first_token, first_logits, second_token, second_logits = mp.pronoun_logits(
                                            pronouns_list=p_list,
                                            generated_sequence=output.sequences[0][input_len:],
                                            scores=output.scores
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
                                                    logits=pronoun_logits)
                                    measurements_idx.append(m)
                                    with open(output_fpath, "a") as f:
                                        f.write(json.dumps(asdict(m)) + "\n")

                                except Exception as e:
                                    error_count += 1
                                    logger.warning(f"Error {e} for output: {decoded_output}. "
                                                   f"Error count {error_count}/{n}")

                logger.warning(f"{error_count}/{n_runs} not captured.")
        logger.info(f"Successfully collected {idx}. "
                    f"Writing {len(measurements_idx)} measurements to {output_fpath}.")

    return




if __name__ == "__main__":
    app()

