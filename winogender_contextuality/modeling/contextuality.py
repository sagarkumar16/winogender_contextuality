from math import perm
import numpy as np
import xarray as xr
from scipy.optimize import linprog
from typing import Callable
from itertools import product, combinations
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

        :param observations: Sentence pair (/list)
        :param measurements: Pair (/list) of contexts (i.e. [['her', 'his'], ['his', 'her']])
        :param outcomes: List of possible outcomes either as strings or boolean values
        """

        self.observations = observations
        self.measurements = measurements
        self.outcomes = outcomes

        # Calculated Attributes
        self.incidence_matrix = None

        ## all possible sentence-input prompt pairs
        self.contexts = list(product(self.observations, self.measurements))
        self.context_idx = {pair: n for n, pair in enumerate(self.contexts)}

        sentence_pairs = []
        for sentence in self.observations:
            pairs = []
            for measurement in self.measurements:
                pairs.append(self.context_idx[(sentence, measurement)])

        # All possible tuples of sentence-input pairs i.e. (a,b), (a',b), etc.
        self.context_pairs = list(product(*sentence_pairs))

        # All possible outcomes tuples i.e. (0,0), (0,1), etc.
        self.outcome_pairs = list(product(self.outcomes, repeat=len(self.measurements)))

        # Empty measurement scenario matrix
        self.scenario = xr.DataArray(np.zeros((len(self.context_pairs), len(self.outcome_pairs))),
                                     dims=['context_pair', 'outcome_pair'],
                                     coords=[self.context_pairs, self.outcome_pairs])

    def incidence_matrix(self):

        # Dimensions from Abramsky and Brandenburger (2011)
        num_rows = int((len(self.measurements) * len(self.outcomes)) ** len(self.observations))
        num_columns = int(len(self.outcomes) ** (len(self.measurements) * len(self.observations)))


        # All possible global assignments e.g. (a, a', b, b') [order comes from product function in self.contexts]
        global_assignments = list(product([0, 1], repeat=len(self.observations)*len(self.measurements)))

        # All pairs of context pairs and outcome pairs
        context_outcome_pairs = product(self.context_pairs, self.outcome_pairs)

        arr = xr.DataArray(np.zeros(shape=(num_rows, num_columns)),
                           dims=['s', 't'],
                           coords=[context_outcome_pairs, global_assignments]
                           )

        for p in arr.s:
            context_pair, outcome_pair = p.item()
            for b in arr.t:
                bool_str = b.item()
                bool_vals = []
                for dim in range(len(self.measurements)):
                    bool_vals.append(bool_str[context_pair[dim]] == outcome_pair[dim])
                arr[p,b] = all(bool_vals)

        return arr

# TODO: finish this code
def check_feasibility(measurement_scenario: MeasurementScenario) -> bool:
    m = measurement_scenario.incidence_matrix()
    vals = measurement_scenario.scenario.values.reshape(-1)

    m_prime = np.vstack([m.values, np.ones(m.shape[1])])
    v_prime = np.hstack([vals, np.ones(1)])

    # dummy objective
    c = np.zeros(m.shape[1])

    res = np.linprog(c=c, A_eq=m_prime, b_eq=v_prime, bounds=[(0,1)]*m.shape[1])

    # should return whether res.status = 2 (slightly more robust than just that it was unsuccessful)
    if res.status == 2:
        return False
    else:
        return res.status


# TODO: This function simply takes in the matrix and outputs the contextuality
def calculate_contextuality(measurement_scenario: MeasurementScenario) -> tuple[bool, float]:
    return

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