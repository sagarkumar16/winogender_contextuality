import os

import typer
from winogender_contextuality.modeling.prompting import *
from winogender_contextuality.modeling.ModelProbs import *
from winogender_contextuality.modeling.contextuality import *
from winogender_contextuality.config import *


app = typer.Typer()

HF_KEY = os.environ.get("HF_KEY")

# TODO: is this the right option? Or should it be a class of its own?
def create_prompt_pair():
    return

# TODO: Run single iteration -- how do we want to break this up, tho?
@app.command()
def get_contextuality(
        prompt: str,
        options: list[str],
        mode: str,
        model_name: str,
        observations: list[str],
        measurements: list[list[str]],
        outcomes: list[str] | list[bool], # This will become redundant
        input_path: str = PROCESSED_DATA_DIR / "wino_pairs.tsv", # path to the tsv
        generation: bool = False, # this is a new var that decides which one to run
        key: str = HF_KEY,
        model_path: str = None,
        quantized: bool = True,
        game: bool = False, # whether it is the game prompt
        rewards: list = [],
        mem_str: str = "",
        **kwargs
):

    # TODO: WRONG -- this has to take in the prompt pairs

    model_probs = ModelProbs(mode, model_name, key, model_path, quantized)

    # Load model
    model_probs.load_model()



    # Get logits
    if generation:
        logits = model_probs.get_completed_logits(prompt, **kwargs)
    else:
        logits = model_probs.get_raw_logits(prompt)

    # Softmax over the logits


    # Get token ids -- how do we make sure we're getting all the possible ones?


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()