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
def calculate_contextual_fraction(measurement_scenario: MeasurementScenario) -> float:
    """
    Calculates the contextual fraction of a measurement scenario.

    The contextual fraction is defined as the minimum weight of the contextual part
    when decomposing the probability distribution as:
    P = (1-w) * P_nc + w * P_c

    where P_nc is non-contextual, P_c is contextual, and w is the contextual fraction.

    Returns: Contextual fraction between 0 and 1, where 0 means fully non-contextual and 1 means fully contextual.
    """

    # Get the incidence matrix and scenario probabilities
    m = measurement_scenario.incidence_matrix()
    p_obs = measurement_scenario.scenario.values.reshape(-1)

    # First check if the scenario is feasible (non-contextual)
    is_feasible, _ = check_feasibility(measurement_scenario)

    if is_feasible:
        logger.info("Scenario is non-contextual (contextual fraction = 0)")
        return 0.0

    # If not feasible, calculate the contextual fraction
    # We solve: minimize w such that (p_obs - w * p_c) is non-contextual
    # This is equivalent to: minimize w such that M * lambda = p_obs - w * p_c
    # where lambda >= 0, sum(lambda) = 1 - w

    # Set up the optimization problem
    # Variables: [lambda_1, ..., lambda_n, w] where n is number of columns in M
    n_lambda = m.shape[1]
    n_vars = n_lambda + 1  # lambda variables + w

    # Objective: minimize w (last variable)
    c = np.zeros(n_vars)
    c[-1] = 1.0  # minimize w

    # Constraints: M * lambda + w * p_obs = p_obs
    # Rearranged: M * lambda - w * p_obs = 0
    # But we want: M * lambda = p_obs - w * p_c
    # For simplicity, we'll use p_c = p_obs (worst case contextual distribution)

    # Equality constraints: M * lambda + w * p_obs = p_obs
    A_eq = np.hstack([m.values, -p_obs.reshape(-1, 1)])
    b_eq = np.zeros(len(p_obs))

    # Additional constraint: sum(lambda) + w = 1 (normalization)
    A_eq_norm = np.zeros((1, n_vars))
    A_eq_norm[0, :n_lambda] = 1.0  # sum of lambdas
    A_eq_norm[0, -1] = 1.0  # plus w
    b_eq_norm = np.array([1.0])

    # Combine equality constraints
    A_eq_combined = np.vstack([A_eq, A_eq_norm])
    b_eq_combined = np.hstack([b_eq, b_eq_norm])

    # Bounds: lambda_i >= 0 for all i, 0 <= w <= 1
    bounds = [(0, None)] * n_lambda + [(0, 1)]

    # Solve the optimization problem
    try:
        res = linprog(c=c, A_eq=A_eq_combined, b_eq=b_eq_combined,
                      bounds=bounds, method='highs')

        if res.success:
            contextual_fraction = res.x[-1]  # w is the last variable
            logger.info(f"Contextual fraction calculated: {contextual_fraction:.4f}")
            return float(contextual_fraction)
        else:
            logger.warning(f"Optimization failed: {res.message}")

    except Exception as e:
        logger.error(f"Error in contextual fraction calculation: {e}")


def calculate_contextual_fraction_alternative(measurement_scenario: MeasurementScenario) -> float:
    """
    Alternative method to calculate contextual fraction using distance-based approach.

    This method finds the closest non-contextual distribution and calculates
    the fraction based on the L1 distance.
    """

    m = measurement_scenario.incidence_matrix()
    p_obs = measurement_scenario.scenario.values.reshape(-1)

    # Find the closest non-contextual distribution
    # Minimize ||p_obs - p_nc||_1 subject to M * lambda = p_nc, lambda >= 0, sum(lambda) = 1

    n_lambda = m.shape[1]
    n_p = len(p_obs)

    # Variables: [lambda_1, ..., lambda_n, p_nc_1, ..., p_nc_m, t_1, ..., t_m]
    # where t_i are auxiliary variables for L1 norm: |p_obs_i - p_nc_i| <= t_i
    n_vars = n_lambda + n_p + n_p  # lambda + p_nc + t

    # Objective: minimize sum(t_i)
    c = np.zeros(n_vars)
    c[n_lambda + n_p:] = 1.0  # minimize sum of t variables

    # Constraint 1: M * lambda = p_nc
    A_eq1 = np.zeros((n_p, n_vars))
    A_eq1[:, :n_lambda] = m.values
    A_eq1[:, n_lambda:n_lambda + n_p] = -np.eye(n_p)
    b_eq1 = np.zeros(n_p)

    # Constraint 2: sum(lambda) = 1
    A_eq2 = np.zeros((1, n_vars))
    A_eq2[0, :n_lambda] = 1.0
    b_eq2 = np.array([1.0])

    # Combine equality constraints
    A_eq = np.vstack([A_eq1, A_eq2])
    b_eq = np.hstack([b_eq1, b_eq2])

    # Inequality constraints for L1 norm: -t_i <= p_obs_i - p_nc_i <= t_i
    # This gives us: p_nc_i - t_i <= p_obs_i and p_obs_i <= p_nc_i + t_i
    A_ub = np.zeros((2 * n_p, n_vars))
    b_ub = np.zeros(2 * n_p)

    for i in range(n_p):
        # p_nc_i - t_i <= p_obs_i  =>  p_nc_i - t_i - p_obs_i <= 0
        A_ub[2 * i, n_lambda + i] = 1.0  # p_nc_i
        A_ub[2 * i, n_lambda + n_p + i] = -1.0  # -t_i
        b_ub[2 * i] = p_obs[i]

        # p_obs_i <= p_nc_i + t_i  =>  -p_nc_i - t_i + p_obs_i <= 0
        A_ub[2 * i + 1, n_lambda + i] = -1.0  # -p_nc_i
        A_ub[2 * i + 1, n_lambda + n_p + i] = -1.0  # -t_i
        b_ub[2 * i + 1] = -p_obs[i]

    # Bounds: lambda_i >= 0, p_nc_i >= 0, t_i >= 0
    bounds = [(0, None)] * n_vars

    try:
        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                      bounds=bounds, method='highs')

        if res.success:
            # Extract the closest non-contextual distribution
            p_nc = res.x[n_lambda:n_lambda + n_p]

            # Calculate L1 distance
            l1_distance = np.sum(np.abs(p_obs - p_nc))

            # Contextual fraction is the L1 distance (normalized)
            # Since probabilities sum to 1, the maximum L1 distance is 2
            contextual_fraction = l1_distance / 2.0

            logger.info(f"Alternative contextual fraction: {contextual_fraction:.4f}")
            return float(contextual_fraction)
        else:
            logger.error(f"Alternative optimization failed: {res.message}")
            return 1.0  # Assume fully contextual if calculation fails

    except Exception as e:
        logger.error(f"Error in alternative contextual fraction calculation: {e}")
        return 1.0


