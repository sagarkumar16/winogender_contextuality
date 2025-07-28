import os
import json
import pathlib
import pandas as pd
import ast
import typer
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
    context: Context
    measurement: dict[str, str]

HF_KEY = os.environ.get("HF_KEY")

@app.command()
def simulate(
        mode: str,
        model_name: str,
        temperature: float,
        sim: bool,
        n_runs: int = 1000,
        input_fpath: pathlib.Path = PROCESSED_DATA_DIR / "wino_pairs.tsv",
        output_dir: pathlib.Path = PROCESSED_DATA_DIR
):

    """

    :param input_fpath:
    :param output_fpath: ndjson file
    :param mode:
    :param model_name:
    :param temperature:
    :param sim:
    :param n_runs:
    :return:
    """

    df = pd.read_csv(input_fpath, sep="\t")
    mp = ModelProbs(
        mode=mode,
        model_name=model_name,
        key=HF_KEY,
        model_path=MODELS_DIR)

    mp.load_model()

    output_fpath = output_dir / f"measurements_{model_name.split('/')[-1]}_{temperature}.ndjson"

    pbar = tqdm(df.index, desc="Simulating")

    for idx in pbar:
        measurements_idx = []
        # Unpacking CSV -- hard coding for now
        sentences = {0: df.template_1[idx],
                     1: df.template_2[idx]}
        pronouns = {0: ast.literal_eval(df.differences_1[idx]),
                    1: ast.literal_eval(df.differences_2[idx])}

        error_count = 0

        #for _ in range(n_runs):
        for s_perm in permutations(sentences.keys()):

            s1 = sentences[s_perm[0]]
            s2 = sentences[s_perm[1]]
            p1 = pronouns[s_perm[0]]
            p2 = pronouns[s_perm[1]]

            for i, p1_perm in enumerate(permutations(p1)):
                for j, p2_perm in enumerate(permutations(p2)):

                    s_list = [s1, s2]
                    p_list = [p1_perm, p2_perm]

                    prompt = role_content_base(*no_game_seq_prompt(p_list, s_list))

                    input, output = mp.get_completion(prompt=prompt, temperature=temperature)

                    input_len = input.shape[1]

                    try:
                        json_output = ast.literal_eval(
                            mp.tokenizer.decode(output.sequences[0][input_len - 5:], skip_special_tokens=True)
                        )

                        c = Context(sent_order=s_perm,
                                    pnoun_order=(i,j),
                                    sentence_1=s1,
                                    sentence_2=s2,
                                    pronouns_1=p1,
                                    pronouns_2=p2)

                        measurements_idx.append(Measurement(context=c, measurement=json_output))
                    except Exception as e:
                        error_count += 1

                    logger.warning(f"{error_count}/{n_runs} not captured.")
                    logger.info(f"Successfully collected {idx}. "
                                f"Writing {len(measurements_idx)} measurements to {output_fpath}.")


        with open(output_fpath, "a") as f:
            f.write(json.dumps([asdict(m) for m in measurements_idx]) + "\n")


if __name__ == "__main__":
    app()

