import os
import pathlib
import pandas as pd
import typer
import ast
from itertools import chain
from datetime import datetime
import torch.nn.functional as F
from winogender_contextuality.modeling.prompting import *
from winogender_contextuality.modeling.ModelProbs import *
from winogender_contextuality.modeling.contextuality import *
from winogender_contextuality.config import *
from winogender_contextuality.utils import *


app = typer.Typer()
HF_KEY = os.environ.get("HF_KEY")

def compute_joint(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the full joint probability distribution matrix for multiple observations.

    :param matrix: shape (n_observations, n_outcomes), row-stochastic.

    :returns joint_array: N-dimensional tensor of joint probabilities with shape
    (n_outcomes, n_outcomes, ..., n_outcomes) [n_trials times].
    """
    n_observations, n_outcomes = matrix.shape

    # Generate all possible outcome combinations for n trials
    all_combinations = product(range(n_outcomes), repeat=n_observations)

    # Compute joint probabilities for each combination
    joint_probs = [
        np.prod([matrix[obs_idx, outcome_idx] for obs_idx, outcome_idx in enumerate(combo)])
        for combo in all_combinations
    ]

    # Reshape into an N-D tensor
    joint_array = np.array(joint_probs).reshape([n_outcomes] * n_observations)

    return joint_array

@app.command()
def get_contextuality(
        mode: str,
        model_name: str,
        generation: bool,
        game: bool,
        input_path: pathlib.Path = PROCESSED_DATA_DIR / "wino_pairs.tsv"
) -> None:

    """
    Calculates contextuality for a pair of sentences, formatted like wino_pairs.tsv.

    :param mode: 'gpu' or 'api'
    :param model_name: name of huggingface model for download
    :param generation: whether to use generation mode
    :param game: Game prompt or not
    :param input_path: path to input TSV
    :param kwargs: additional arguments for generation -- NOT CURRENTLY IMPLEMENTED BC TYPER
    :return: None
    """


    df = pd.read_csv(input_path, sep="\t")
    mp = ModelProbs(
        mode=mode,
        model_name=model_name,
        key=HF_KEY,
        model_path=MODELS_DIR)

    mp.load_model()

    contextuality_list = []
    for row_idx in tqdm(df.index):
        ms = MeasurementScenario(
            observations=['template_1', 'template_2'],
            measurements=['he_first', 'she_first'],
            outcomes=[0,1]
        )

        for arr_idx, pair in enumerate(ms.context_pairs):
            arr = np.zeros((len(ms.observations), len(ms.outcomes)))
            for pair_idx, oc_idx in enumerate(pair):
                oc_pair = ms.reverse_context_idx.get(oc_idx)
                obs, ctx = oc_pair
                obs_index = obs[-1]
                sent = df[obs][row_idx]  # the 0 is iterated index
                pnouns = reverse_pronouns(df[f"differences_{obs_index}"][row_idx], ctx)
                prompt = get_role_content_prompt(game=game, options=pnouns, sentence=sent)
                if generation:
                    raise NotImplementedError
                else:
                    logits = mp.get_raw_logits(prompt=prompt).to('cpu')
                # For now, we just use the two tokens
                token = mp.get_token_ids(options=[" " + s for s in ast.literal_eval(
                    df[f"differences_{obs_index}"][row_idx])])[1][0]  # in order [m, f]
                softmax = F.softmax(logits, dim=0)
                prob = softmax[token]
                probs = np.array([1-prob, prob])
                arr[pair_idx] = probs
            ms.scenario[arr_idx] = compute_joint(arr).reshape(-1)

        contextuality = check_feasibility(ms)
        if contextuality[1].status != 2:
            logger.warning(f"Context {row_idx} status {contextuality[1].status}.")

        contextuality_list.append(contextuality[0])

    logger.success(f"Completed all contextuality calculations.")

    out_df = df
    out_df['Contextuality'] = [not b for b in contextuality_list]

    model_fname = model_name.split("/")[-1]
    output_path = PROCESSED_DATA_DIR / f"boolean_contextuality_{model_fname}_game-{game}_{datetime.now()}.tsv"
    out_df.to_csv(output_path, index=False)
    return


if __name__ == "__main__":
    app()