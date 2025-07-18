from math import perm
import numpy as np
from typing import Callable
from itertools import product
from winogender_contextuality.config import *
from winogender_contextuality.modeling.ModelProbs import ModelProbs
from winogender_contextuality.utils import *
from loguru import logger
from pathlib import Path
import typer

app = typer.Typer()

"""
Assesses contextuality by calling a ModelProbs class to (1) determine if a joint space can be constructed
and (2) measure the contextual fraction.

- Sagar Kumar, 2025
"""

# TODO: HERE is where everything can get filled in via either typer or config.py

# TODO: Create function to output all relevant probabilities
class MeasurementScenario:

    def __init__(self,
                 observations: list[str],
                 measurements: list[list[str]],
                 outcomes: list[str] | list[bool]):

        """

        :param observations: Prompts
        :param measurements: List of contexts (i.e. [['her', 'his'], ['his', 'her']])
        :param outcomes: Possible outcomes either as output strings or boolean values
        """

        self.observations = observations
        self.measurements = measurements
        self.outcomes = outcomes

        # Calculated Attributes
        self.incidence_matrix = None
        self.measurement_map = {}
        self.outcome_map = {}


    def incidence_matrix(self):

        num_rows = int((len(self.measurements) * len(self.outcomes)) ** len(self.observations))
        num_columns = int(len(self.outcomes) ** (len(self.measurements) * len(self.observations)))

        # All possible global assignments (a, a', b, b') -> 16 combinations of 0/1
        global_assignments = list(product([0, 1], repeat=len(self.observations)*len(self.measurements)))

        arr = np.zeros(shape=(num_rows, num_columns))

        context_pairs = list(product(self.measurements, repeat=len(self.measurements)))
        outcome_pairs = list(product(self.outcomes, repeat=len(self.observations)))

        sections = []
        idx = 0
        for i, measurement in enumerate(context_pairs):
            self.measurement_map[i] = measurement
            for j, outcome in enumerate(outcome_pairs):
                self.outcome_map[j] = outcome
                sections.append((idx, i, j))
                idx += 1

        for row_idx, measurement_idx, outcome_idx in sections:
            for col_idx, global_assignments in enumerate(global_assignments):

                # TODO: Figure out how to finish this








# TODO: strict contextuality
def check_feasibility(measurement_scenario: MeasurementScenario) -> bool:
    mat = measurement_scenario.matrix

    # should return whether or not res.status = 2 (slightly more robust than just that it was unsuccessful)
    return


# TODO: This function simply takes in the matrix and outputs the contextuality
def calculate_contextuality(measurement_scenario: MeasurementScenario) -> tuple[bool, float]:



    return ()

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