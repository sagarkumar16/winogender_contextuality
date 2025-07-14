from winogender_contextuality.config import *
from winogender_contextuality.utils import *
from loguru import logger
from pathlib import Path
import typer

app = typer.Typer()

"""
Assesses contextuality by calling a ModeProbs class to (1) determine if a joint space can be constructed
and (2) measure the contextual fraction.

- Sagar Kumar, 2025
"""

# TODO: HERE is where everything can get filled in via either typer or config.py

# TODO: Create function to output all relevant probabilities

# TODO: Create function to calculate contextuality based on probabilities

# TODO: Create function to calculate contextual fraction based on probabilities


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