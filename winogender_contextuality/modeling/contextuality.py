import numpy as np
import xarray as xr
from scipy.optimize import linprog, OptimizeResult
from itertools import product
from loguru import logger

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
                 measurements: list[str],
                 outcomes: list[str] | list[bool] | list[int]):

        """

        :param observations: Sentence pair (/list)
        :param measurements: Pair (/list) of contexts (i.e. [['her', 'his'], ['his', 'her']])
        :param outcomes: List of possible outcomes either as strings or boolean values
        """

        self.observations = observations
        self.measurements = measurements
        self.outcomes = outcomes


        ## all possible sentence-input prompt pairs
        self.contexts = list(product(self.observations, self.measurements))
        self.context_idx = {pair: n for n, pair in enumerate(self.contexts)}
        self.reverse_context_idx = {n: pair for n, pair in enumerate(self.contexts)}

        sentence_pairs = []
        for sentence in self.observations:
            pairs = []
            for measurement in self.measurements:
                pairs.append(self.context_idx[(sentence, measurement)])
            sentence_pairs.append(pairs)

        # All possible tuples of sentence-input pairs i.e. (a,b), (a',b), etc.
        self.context_pairs = list(product(*sentence_pairs))
        self.context_pair_map = {i: pair for i, pair in enumerate(self.context_pairs)}

        # All possible outcomes tuples i.e. (0,0), (0,1), etc.
        self.outcome_pairs = list(product(self.outcomes, repeat=len(self.measurements)))
        self.outcome_pair_map = {j: pair for j, pair in enumerate(self.outcome_pairs)}

        # Empty measurement scenario matrix
        self.scenario = xr.DataArray(np.zeros((len(self.context_pairs), len(self.outcome_pairs))),
                                     dims=['context_pair', 'outcome_pair'],
                                     coords=[list(self.context_pair_map.keys()), list(self.outcome_pair_map.keys())])

    def incidence_matrix(self):

        # Dimensions from Abramsky and Brandenburger (2011)
        num_rows = int((len(self.measurements) * len(self.outcomes)) ** len(self.observations))
        num_columns = int(len(self.outcomes) ** (len(self.measurements) * len(self.observations)))


        # All possible global assignments e.g. (a, a', b, b') [order comes from product function in self.contexts]
        global_assignments = list(product([0, 1], repeat=len(self.observations)*len(self.measurements)))
        global_assignments_map = {t: bstr for t,bstr in enumerate(global_assignments)}

        # All pairs of context pairs and outcome pairs
        context_outcome_pairs = product(self.context_pairs, self.outcome_pairs)
        context_outcome_pairs_map = {p: pair for p, pair in enumerate(context_outcome_pairs)}

        arr = xr.DataArray(np.zeros(shape=(num_rows, num_columns)),
                           dims=['s', 't'],
                           coords=[list(context_outcome_pairs_map.keys()), list(global_assignments_map.keys())]
                           )

        for p in arr.s:
            context_pair, outcome_pair = context_outcome_pairs_map[p.item()]
            for b in arr.t:
                bool_str = global_assignments_map[b.item()]
                bool_vals = []
                for dim in range(len(self.measurements)):
                    bool_vals.append(bool_str[context_pair[dim]] == outcome_pair[dim])
                arr[p,b] = all(bool_vals)

        return arr


def check_feasibility(measurement_scenario: MeasurementScenario) -> tuple[bool, OptimizeResult]:
    m = measurement_scenario.incidence_matrix()
    vals = measurement_scenario.scenario.values.reshape(-1)

    m_prime = np.vstack([m.values, np.ones(m.shape[1])])
    v_prime = np.hstack([vals, np.ones(1)])

    # dummy objective
    c = np.zeros(m.shape[1])

    res = linprog(c=c, A_eq=m_prime, b_eq=v_prime, bounds=[(0,1)]*m.shape[1], method='highs')

    # should return whether res.status = 2 (slightly more robust than just that it was unsuccessful)
    status = res.status
    logger.info(f'Measurement status: {status}')

    return res.success, res


# TODO: Create function to calculate contextual fraction based on probabilities
# New from Claude
def calculate_contextual_fraction_abramsky(measurement_scenario: MeasurementScenario) -> float:
    """
    Calculate contextual fraction using the method from Abramsky, Barbosa, and Mansfield (2017).

    This implements the linear programming approach described in their PRL paper.
    The contextual fraction CF(e) = 1 - NCF(e), where NCF(e) is the noncontextual fraction
    calculated by solving the LP in equation (3) of their paper.

    Returns:
        float: Contextual fraction between 0 and 1, where 0 means non-contextual
               and 1 means strongly contextual.
    """

    # Get the incidence matrix M
    M = measurement_scenario.incidence_matrix()

    # Get the empirical model as a vector v_e
    # This flattens the scenario probabilities into a vector
    v_e = measurement_scenario.scenario.values.reshape(-1)

    # Ensure probabilities are normalized (should sum to number of context pairs)
    # Each context pair should have probabilities that sum to 1
    n_context_pairs = len(measurement_scenario.context_pairs)
    if np.abs(np.sum(v_e) - n_context_pairs) > 1e-10:
        logger.warning(f"Probabilities don't sum to {n_context_pairs} (sum = {np.sum(v_e)})")
        logger.warning("This may indicate an issue with the probability distribution")

    # Set up the linear program from equation (3) in Abramsky et al.
    # Find b ∈ R^n maximizing 1·b subject to M·b ≤ v_e and b ≥ 0

    n_columns = M.shape[1]  # number of global assignments

    # Objective: maximize 1·b (equivalently, minimize -1·b)
    c = -np.ones(n_columns)

    # Inequality constraints: M·b ≤ v_e
    A_ub = M.values
    b_ub = v_e

    # Bounds: b ≥ 0 (each component of b is non-negative)
    bounds = [(0, None)] * n_columns

    try:
        # Solve the linear program
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if res.success:
            # The noncontextual fraction is 1·b* where b* is the optimal solution
            noncontextual_fraction = -res.fun  # negative because we minimized -1·b

            # Contextual fraction = 1 - noncontextual fraction
            contextual_fraction = 1.0 - noncontextual_fraction

            # Ensure the result is in [0, 1] due to numerical precision
            contextual_fraction = max(0.0, min(1.0, contextual_fraction))

            logger.info(f"Noncontextual fraction: {noncontextual_fraction:.6f}")
            logger.info(f"Contextual fraction: {contextual_fraction:.6f}")

            # Additional diagnostic information
            if contextual_fraction < 1e-10:
                logger.info("Model is non-contextual (within numerical precision)")
            elif abs(contextual_fraction - 1.0) < 1e-10:
                logger.info("Model is strongly contextual")
            else:
                logger.info(f"Model has partial contextuality: {contextual_fraction:.4f}")

            return float(contextual_fraction)

        else:
            logger.error(f"Linear programming failed: {res.message}")
            logger.error("This may indicate an infeasible problem or numerical issues")

            # If LP fails, try to determine if it's due to infeasibility
            # An infeasible problem might indicate issues with the setup
            return np.nan

    except Exception as e:
        logger.error(f"Error in linear programming: {e}")
        return np.nan