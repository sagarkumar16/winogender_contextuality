import os
import pathlib
import pandas as pd
import typer
import ast
from itertools import chain
from datetime import datetime
from winogender_contextuality.modeling.prompting import *
from winogender_contextuality.modeling.ModelProbs import *
from winogender_contextuality.modeling.contextuality import *
from winogender_contextuality.config import *
from winogender_contextuality.utils import *


app = typer.Typer()
HF_KEY = os.environ.get("HF_KEY")

@app.command()
def get_contextuality(
        mode: str,
        model_name: str,
        game: bool,
        input_path: pathlib.Path = PROCESSED_DATA_DIR / "wino_pairs.tsv"
) -> None:

    """
    Calculates contextuality for a pair of sentences, formatted like wino_pairs.tsv.

    :param mode: 'gpu' or 'api'
    :param model_name: name of huggingface model for download
    :param game: Game prompt or not
    :param input_path: path to input TSV
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
                logits = mp.get_raw_logits(prompt=prompt).to('cpu')
                # For now, we just use the two tokens
                tokens = mp.get_token_ids(options=[" " + s for s in ast.literal_eval(
                    df[f"differences_{obs_index}"][row_idx])])  # in order [m, f]
                softmax = masked_softmax(list(chain.from_iterable(tokens)), logits)
                probs = softmax / torch.sum(softmax)
                arr[pair_idx] = probs.detach().numpy()
            ms.scenario[arr_idx] = arr.reshape(-1)  # does this work

        contextuality = check_feasibility(ms)
        if contextuality[1].status != 2:
            logger.warning(f"Context {row_idx} status {contextuality[1].status}.")

        contextuality_list.append(contextuality[0])

    logger.success(f"Completed all contextuality calculations.")

    out_df = df
    out_df['Contextuality'] = [not b for b in contextuality_list]

    output_path = PROCESSED_DATA_DIR / f"contextuality_{model_name}_{datetime.now()}.tsv"
    out_df.to_csv(output_path, index=False)

    return


if __name__ == "__main__":
    app()