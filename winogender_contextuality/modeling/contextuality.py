import numpy as np
import xarray as xr
from scipy.optimize import linprog, OptimizeResult
from itertools import product
from loguru import logger
from winogender_contextuality.utils import Measurement, get_index
from collections import Counter
from scipy.special import softmax

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

        return

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


def cbd_expectation(px: float):

    """
    Returns the expectation value of a random variable, given its marginal probability (as per Dzhafarov)
    :param px: marginal probability
    :return: expectation value
    """
    return 2*px - 1

def cbd_correlation(px: float,
                    py: float,
                    pxy: float):
    """
    Returns the correlation of two random variables, given their marginals and joint (as per Dzhafarov)
    :param px: marginal probability of X
    :param py: marginal probability of Y
    :param pxy: joint probability
    :return:
    """
    return 4*pxy - 2*px - 2*py + 1

def cbd_s1_4cycle(w,x,y,z):
    """
    The s1 term of a cyclic system of rank 4 (Dzhafarov & Kujala):

        s1 = max over sign vectors with an ODD number of -1s of  sum_i (sign_i * x_i)

    For n = 4 the odd-sign vectors come in +/- pairs, so the maximum over all eight of them is
    the maximum of the absolute values of the four one-minus-sign patterns.

    NOTE: the term |w + x - y - z| has TWO minus signs (even), so it is not admissible; it was
    used here in place of |w + x - y + z| and could overstate s1, and hence delta_c.
    """
    term1 = abs(w+x+y-z)
    term2 = abs(w+x-y+z)
    term3 = abs(w-x+y+z)
    term4 = abs(-w+x+y+z)
    return max(term1,term2,term3,term4)


def pronoun_context_array(
        index: int,
        data: list[dict] | list[Measurement],
        forward: bool = True
) -> np.ndarray:
    """
    Given data in the form of a list of Measurement objects (see collect_sequential.py) OR equivalent dictionaries,
    this function outputs the measurement scenario array, as described in sheaf-contextual approaches to contextuality.

    :param index: integer index sentence pair
    :param data: list of Measurement objects (or equivalently structured dictionaries)
    :param forward: Whether sentences should be in forward or reverse order (default=True)
    :return: measurement scenario array
    """
    # Constants
    female_pnouns = {'she', 'her', 'hers'}
    orientations = [[0, 0], [0, 1], [1, 0], [1, 1]]

    # Sentence order control
    if forward == True:
        sentence_order = [0, 1]
    else:
        sentence_order = [1, 0]

    # Loading data and creating empty dictionary
    measurement_data = [d for d in data if (d['index'] == index) and (d['context']['sent_order'] == sentence_order)]
    measurement_dict = {tuple(o): {tuple(o): 0 for o in orientations} for o in orientations}
    print(f"Checking contextuality on basis of {len(measurement_data)} measurements.")

    # Collecting empirical probabilities
    for d in measurement_data:
        orientation = d['context']['pnoun_order']
        measurement = d['measurement']
        encoded_m = tuple([val in female_pnouns for val in measurement.values()])
        measurement_dict[tuple(orientation)][encoded_m] += 1

    # Converting dictionary of dictionaries to array
    arr_list = []
    for outcome_dict in measurement_dict.values():
        ctx_total = sum(outcome_dict.values())
        ctx_list = []
        for val in outcome_dict.values():
            try:
                ctx_list.append(val / ctx_total)
            except ZeroDivisionError:
                ctx_list.append(0)

        arr_list.append(ctx_list)
    arr = np.array(arr_list)

    return arr

def calculate_pronouns_nc_fraction(
        arr: np.ndarray
) -> float:

    """
    Degree of contextuality (CbD) for the rank-4 cyclic system over pronoun-option orders.

    The four content variables are
        q0 = sentence-1 pronoun, options presented male-first
        q1 = sentence-2 pronoun, male-first
        q2 = sentence-1 pronoun, female-first
        q3 = sentence-2 pronoun, female-first
    and a cyclic system of rank 4 requires context i to measure (q_i, q_{i+1 mod 4}), i.e.
    CONSECUTIVE CONTEXTS MUST SHARE A CONTENT VARIABLE:

        c0 = (q0, q1) = options (0,0)     c1 = (q1, q2) = options (1,0)
        c2 = (q2, q3) = options (1,1)     c3 = (q3, q0) = options (0,1)

    `arr` arrives in the order MeasurementScenario uses -- the product order
    (0,0), (0,1), (1,0), (1,1) -- which is correct for the sheaf/linear-programming code but is
    NOT a cycle: it places (0,1) next to (1,0), two contexts that share no content variable.
    Consuming it row-by-row as if it were a cycle mispairs the disturbance terms. We therefore
    reindex into cycle order here rather than changing the array, which the sheaf code depends on.

    In contexts c1 and c3 the lower-indexed content variable is the SECOND-sentence pronoun, so
    the roles of the two marginals swap. The correlation is symmetric in its marginals and so is
    unaffected; only the disturbance pairing changes.

    :param arr: MeasurementScenario.scenario.values array, rows in product order
                (0,0), (0,1), (1,0), (1,1)
    :return: delta_c; the system is contextual iff delta_c > 0
    """

    # product-order row index -> (cycle position, swap the two marginals?)
    # (0,0) -> row 0, (0,1) -> row 1, (1,0) -> row 2, (1,1) -> row 3
    cycle = [(0, False),   # c0 = options (0,0): first = q0 (sentence 1), second = q1
             (2, True),    # c1 = options (1,0): first = q1 (sentence 2), second = q2
             (3, False),   # c2 = options (1,1): first = q2 (sentence 1), second = q3
             (1, True)]    # c3 = options (0,1): first = q3 (sentence 2), second = q0

    correlations = []
    firsts = []
    seconds = []

    for row_idx, swap in cycle:
        dzhafarov_arr = arr[row_idx].reshape(2, 2)
        # index 1 = female pronoun probability
        px = np.sum(dzhafarov_arr, axis=1)[1]   # sentence-1 pronoun marginal
        py = np.sum(dzhafarov_arr, axis=0)[1]   # sentence-2 pronoun marginal
        pxy = dzhafarov_arr[1, 1]

        # cbd_correlation is symmetric in px and py, so the swap does not touch it
        correlations.append(cbd_correlation(px, py, pxy))

        p_first, p_second = (py, px) if swap else (px, py)
        firsts.append(cbd_expectation(p_first))
        seconds.append(cbd_expectation(p_second))

    # Calculating the S1 term
    s1_term = cbd_s1_4cycle(*correlations)

    # Inconsistent connectedness: content variable q_i is the FIRST of context i and the SECOND
    # of context i-1, so each term compares its two contexts.
    sum_term = np.sum([abs(firsts[i] - seconds[(i - 1) % 4]) for i in range(4)])

    delta_c = s1_term - 2 - sum_term

    return delta_c

# Sentence-order contextuality via CBD
def sentence_order_results(idx: int,
                           model_measurements: list[Measurement],
                           pnoun_order: list[int] = [0,0],
                           default_pronoun: int = 1  # 0 for male, 1 for female
                           ) -> dict:
    """
    Creates a dictionary given a subset of measurements for a single pair of sentences which summarizes the results of
    all runs, separated on the basis of sentence order, filtered for only one pronoun order.

    :param idx: integer index sentence pair
    :param model_measurements: list of Measurement objects (or equivalently structured dictionaries)
    :param pnoun_order: filtered pronoun order index
    :param default_pronoun: pronoun index for probabiity calculations (default = 1 for female)
    :return:
    """
    sentence_measurements = get_index(idx, model_measurements)
    test_measurements = [d for d in sentence_measurements if d['context']['pnoun_order'] == pnoun_order]

    data_dict = {'forward': {'pnoun_1': [], 'pnoun_2': [], 'pronouns': []},
                 'reverse': {'pnoun_1': [], 'pnoun_2': [], 'pronouns': []}}

    for d in test_measurements:
        context = d['context']['sent_order']
        if context == [0, 1]:
            dict_key = 'forward'
            pnoun_1 = d['measurement']['BLANK1']
            pnoun_2 = d['measurement']['BLANK2']
            default_p1 = d['context']['pronouns_1'][default_pronoun]
            default_p2 = d['context']['pronouns_2'][default_pronoun]
        else:
            dict_key = 'reverse'
            pnoun_1 = d['measurement']['BLANK2']
            pnoun_2 = d['measurement']['BLANK1']
            default_p1 = d['context']['pronouns_2'][default_pronoun]
            default_p2 = d['context']['pronouns_1'][default_pronoun]

        data_dict[dict_key]['pnoun_1'].append(pnoun_1.lower())
        data_dict[dict_key]['pnoun_2'].append(pnoun_2.lower())
        data_dict[dict_key]['pronouns'] = [default_p1, default_p2]

    return data_dict


def sentence_order_single_results(idx: int,
                                  model_measurements: list[dict],
                                  mode: str,
                                  pnoun_order: int,  # 0 for [m, f], 1 for [f, m]
                                  default_pronoun: int = 1  # 0 for male, 1 for female
                                  ) -> dict:
    """
    Creates a dictionary given a subset of measurements for a single pair of sentences which summarizes the results of
    all runs, separated on the basis of sentence order, filtered for only one pronoun order.

    :param idx: integer index sentence pair
    :param model_measurements: list of Measurement objects (or equivalently structured dictionaries)
    :param mode: 'internal' or 'generation'
    :param pnoun_order: filtered pronoun order--NOTE: this uses default_pronoun indexing
    :param default_pronoun: pronoun index for probabiity calculations (default = 1 for female)
    :return:
    """
    sentence_measurements = get_index(idx, model_measurements)
    test_measurements = [d for d in sentence_measurements if (d['context']['pnoun_order'][1] == pnoun_order) and
                         (d['context']['pnoun_order'][0] is not None)]

    data_dict = {'forward': {'fixed_pnoun': [], 'free_pnoun': [], 'pronouns': []},
                 'reverse': {'fixed_pnoun': [], 'free_pnoun': [], 'pronouns': []}}

    if mode=='internal':

        for d in test_measurements:
        
            context = d['context']['sent_order'] 
                
            if context == [0, 1]:
                dict_key = 'forward'
                fixed_pnoun = d['context']['pnoun_order'][0]
                free_pnoun = d['logits']
                default_p1 = d['context']['pronouns_1'][default_pronoun]
                default_p2 = d['context']['pronouns_2'][default_pronoun]
            elif context == [1, 0]:
                dict_key = 'reverse'
                fixed_pnoun = d['context']['pnoun_order'][0]
                free_pnoun = d['logits']
                default_p1 = d['context']['pronouns_2'][default_pronoun]
                default_p2 = d['context']['pronouns_1'][default_pronoun]
            else:
                raise AttributeError
    
            data_dict[dict_key]['fixed_pnoun'].append(fixed_pnoun.lower())
            data_dict[dict_key]['free_pnoun'].append(free_pnoun)
            data_dict[dict_key]['pronouns'] = [default_p1, default_p2]

    elif mode=='generation':
    
        for d in test_measurements:
    
            context = d['context']['sent_order']
    
            if context == [0, 1]:
                dict_key = 'forward'
                fixed_pnoun = d['context']['pnoun_order'][0]
                try:
                    free_pnoun = d['measurement']['BLANK']
                except TypeError:
                    continue
                default_p1 = d['context']['pronouns_1'][default_pronoun]
                default_p2 = d['context']['pronouns_2'][default_pronoun]
            elif context == [1, 0]:
                dict_key = 'reverse'
                fixed_pnoun = d['context']['pnoun_order'][0]
                try:
                    free_pnoun = d['measurement']['BLANK']
                except TypeError:
                    continue
                default_p1 = d['context']['pronouns_2'][default_pronoun]
                default_p2 = d['context']['pronouns_1'][default_pronoun]
            else:
                raise AttributeError
    
            data_dict[dict_key]['fixed_pnoun'].append(fixed_pnoun.lower())
            data_dict[dict_key]['free_pnoun'].append(free_pnoun.lower())
            data_dict[dict_key]['pronouns'] = [default_p1, default_p2]

    else: 
        raise ValueError("Invalid mode. Must be 'internal' or 'generation'")

    return data_dict

def calculate_sentence_nc_fraction(data_dict: dict) -> float:

    """
    Degree of contextuality (CbD, cyclic system of rank 2) from the output of
    sentence_order_results(), i.e. from JOINT two-pronoun measurements.

    Content variables: V = the sentence-1 pronoun, W = the sentence-2 pronoun. Contexts: the two
    sentence orders. sentence_order_results() already remaps BLANK1/BLANK2 so that 'pnoun_1'
    always refers to sentence 1 in both orders.

        delta_c = |<V W>_fwd - <V W>_rev| - |<V>_fwd - <V>_rev| - |<W>_rev - <W>_fwd|

    contextual iff delta_c > 0.

    :param data_dict: output of sentence_order_results()
    :return: delta_c
    """
    C1_size = len(data_dict['forward']['pnoun_1'])
    C2_size = len(data_dict['reverse']['pnoun_1'])

    V1_dict = Counter(data_dict['forward']['pnoun_1'])
    W2_dict = Counter(data_dict['forward']['pnoun_2'])
    W1_dict = Counter(data_dict['reverse']['pnoun_1'])
    V2_dict = Counter(data_dict['reverse']['pnoun_2'])

    target_f = data_dict['forward']['pronouns']
    target_r = data_dict['reverse']['pronouns']

    V1 = V1_dict.get(target_f[0], 0) / C1_size
    W2 = W2_dict.get(target_f[1], 0) / C1_size
    W1 = W1_dict.get(target_r[0], 0) / C2_size
    V2 = V2_dict.get(target_r[1], 0) / C2_size

    # Compute joint probabilities
    forward_trials = zip(data_dict['forward']['pnoun_1'], data_dict['forward']['pnoun_2'])
    count_c1 = sum(1 for x, y in forward_trials if x == target_f[0] and y == target_f[1])
    V1W2 = count_c1 / C1_size

    reverse_trials = zip(data_dict['reverse']['pnoun_1'], data_dict['reverse']['pnoun_2'])
    count_c2 = sum(1 for x, y in reverse_trials if x == target_r[0] and y == target_r[1])
    V2W1 = count_c2 / C2_size

    # Each correlation uses the two marginals OF ITS OWN CONTEXT: (V1, W2) forward and
    # (W1, V2) reverse. The forward term previously used V2 -- the sentence-2 marginal of the
    # REVERSE context -- in place of W2. cbd_correlation is symmetric in its first two
    # arguments, so the reverse term was already correct.
    delta_c = (
            abs(cbd_correlation(V1, W2, V1W2) - cbd_correlation(W1, V2, V2W1))
            - (abs(cbd_expectation(V1) - cbd_expectation(W1))
               + abs(cbd_expectation(V2) - cbd_expectation(W2)))
    )

    return delta_c

def _free_pronoun_prob(logits) -> float:
    """
    P(female pronoun) for a single trial, from that trial's stored logits.

    Logits are written by collect_sequential in canonical option order, so index 1 is the
    female pronoun.
    """
    return float(softmax(np.asarray(logits, dtype=float).ravel())[1])


def calculate_sentence_dc_fraction(data_dict: dict,
                                   mode: str) -> float:
    """
    Calculates degree of contextuality (CbD, cyclic system of rank 2) from the output of
    sentence_order_single_results().

    The system has two content variables -- the pronoun of sentence A and the pronoun of
    sentence B -- and two contexts, the two sentence orders. In each context one variable is
    the *prime* (fixed by the experimenter: V1 forward, V2 reverse) and the other is the
    model's *free* response (W2 forward, W1 reverse). With +-1-valued outcomes,

        delta_c = |<R_A R_B>_fwd - <R_A R_B>_rev|
                  - |<R_A>_fwd - <R_A>_rev| - |<R_B>_rev - <R_B>_fwd|

    and the system is contextual iff delta_c > 0.

    :param data_dict: output of sentence_order_single_results()
    :param mode: 'internal' (logit-derived probabilities) or 'generation' (empirical frequencies)
    :return: degree of contextuality, or nan if either context has no trials
    """

    if mode not in ('internal', 'generation'):
        raise AttributeError("Mode must be either 'internal' or 'generation'")

    fwd = data_dict['forward']
    rev = data_dict['reverse']

    C1_size = len(fwd['fixed_pnoun'])
    C2_size = len(rev['fixed_pnoun'])

    if C1_size == 0 or C2_size == 0 or not fwd['pronouns'] or not rev['pronouns']:
        logger.warning("Empty context; cannot compute degree of contextuality.")
        return np.nan

    # sentence_order_single_results stores 'pronouns' as [sentence-A female, sentence-B female]
    # *relative to the presentation order*, so in the forward context the prime's target is
    # element 0 and the free sentence's is element 1, and in the reverse context they swap.
    prime_target_f, free_target_f = fwd['pronouns'][0], fwd['pronouns'][1]
    free_target_r, prime_target_r = rev['pronouns'][0], rev['pronouns'][1]

    # Prime marginals: the prime is a string in both modes.
    V1 = Counter(fwd['fixed_pnoun']).get(prime_target_f, 0) / C1_size
    V2 = Counter(rev['fixed_pnoun']).get(prime_target_r, 0) / C2_size

    if mode == 'internal':
        # Each trial's logits give P(free = female | that trial's prime). The marginal is the
        # mean of those per-trial probabilities, and the joint follows p(x,y) = p(y|x)p(x):
        #     V1W2 = (1/N) * sum_t  1{prime_t = female} * p_t
        fwd_probs = [_free_pronoun_prob(z) for z in fwd['free_pnoun']]
        rev_probs = [_free_pronoun_prob(z) for z in rev['free_pnoun']]

        W2 = float(np.mean(fwd_probs))
        W1 = float(np.mean(rev_probs))

        V1W2 = float(np.mean([
            p if x == prime_target_f else 0.0
            for x, p in zip(fwd['fixed_pnoun'], fwd_probs)
        ]))
        V2W1 = float(np.mean([
            p if x == prime_target_r else 0.0
            for x, p in zip(rev['fixed_pnoun'], rev_probs)
        ]))

    else:  # generation
        W2 = Counter(fwd['free_pnoun']).get(free_target_f, 0) / C1_size
        W1 = Counter(rev['free_pnoun']).get(free_target_r, 0) / C2_size

        V1W2 = sum(
            1 for x, y in zip(fwd['fixed_pnoun'], fwd['free_pnoun'])
            if x == prime_target_f and y == free_target_f
        ) / C1_size
        V2W1 = sum(
            1 for x, y in zip(rev['fixed_pnoun'], rev['free_pnoun'])
            if x == prime_target_r and y == free_target_r
        ) / C2_size

    # Each correlation is built from the two marginals OF ITS OWN CONTEXT: (V1, W2) forward and
    # (V2, W1) reverse.
    delta_c = (
            abs(cbd_correlation(V1, W2, V1W2) - cbd_correlation(V2, W1, V2W1))
            - (abs(cbd_expectation(V1) - cbd_expectation(W1))
               + abs(cbd_expectation(V2) - cbd_expectation(W2)))
    )

    return float(delta_c)

