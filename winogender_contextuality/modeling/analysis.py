import numpy as np
import pandas as pd
from winogender_contextuality.config import *
from winogender_contextuality.utils import *


def primed_completion_differences(data: list[Measurement],
                                  mode: str,
                                  max_index: int = 60,
                                  plot: bool = False):
    """
    Plots the probability of female pronoun generation for each sentence pair and its reverse.

    :param data: list of Measurement objects (or equivalently structured dictionaries)
    :param mode: 'internal' or 'generation'
    :param max_index: maximum index in data (last index is thrown away)
    :param plot: whether to plot results

    :returns: list of tuples with (unprimed fpronoun prob, mprimed fpronoun production prob, fprimed fpronoun production prob)
    """

    differences = []

    if mode == 'internal':
        prob_func = get_internal_probs
    elif mode == 'generation':
        prob_func = get_generation_probs
    else:
        raise AttributeError

    for idx in tqdm(range(max_index)):
        all_index_data = get_index(idx, data)

        unprimed_1 = prob_func(get_sent_order([0, 1], get_single_sentences(all_index_data)))
        unprimed_2 = prob_func(get_sent_order([1, 0], get_single_sentences(all_index_data)))

        primed_m1 = prob_func(get_sent_order([0, 1], get_filled_pnoun(0, all_index_data)))
        primed_f1 = prob_func(get_sent_order([0, 1], get_filled_pnoun(1, all_index_data)))
        primed_m2 = prob_func(get_sent_order([1, 0], get_filled_pnoun(0, all_index_data)))
        primed_f2 = prob_func(get_sent_order([1, 0], get_filled_pnoun(1, all_index_data)))

        tup1 = (unprimed_1, primed_m1, primed_f1)
        tup2 = (unprimed_2, primed_m2, primed_f2)

        try:
            differences.append(tup1)
            differences.append(tup2)
        except ValueError:
            print(tup1, tup2)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 4))
        pruned_differences = [(n, t) for n, t in enumerate(differences) if is_valid_tuple(t)]
        collected_indices = [p[0] for p in pruned_differences]

        ax.plot(collected_indices, [p[1][0][1] for p in pruned_differences], label='unprimed', marker='x',
                linestyle='None', color='gray',
                markersize=5)

        ax.plot(collected_indices, [p[1][1][1] for p in pruned_differences], label='m primed', marker='o',
                linestyle='None', color='#E69F00',
                markersize=5)

        ax.plot(collected_indices, [p[1][2][1] for p in pruned_differences], label='f primed', marker='o',
                linestyle='None', color='#009E73',
                markersize=5)

        ax.legend()

        for subidx in collected_indices:
            try:
                min_idx, med_idx, max_idx = np.argsort([i[1] for i in differences[subidx]])
            except IndexError:
                continue

            if med_idx != 0:
                color = 'red'
            elif min_idx != 2:
                color = '#0072B2'
            else:
                color = 'gray'

            lb = differences[subidx][min_idx][1]
            ub = differences[subidx][max_idx][1]
            ax.vlines(subidx, ymin=lb, ymax=ub, alpha=0.5, color=color, linewidth=0.7)

        ax.set_xticks([])

    return differences


def mean_completion_shift(data: list[Measurement] | list[tuple],
                          mode: str | None = None,
                          max_index: int = 60):
    """
    Calculates the mean shift (across all ordered sentences pairs) in the probability of female pronoun production for
    mprimed and fprimed sentences.

    :param data: list of Measurement objects (or equivalently structured dictionaries) OR
         list of tuples from primed_completion_differences()
    :param mode: 'internal' or 'generation' (only necessary if data is a list of dicts/Measurements)
    :param max_index: maximum index in data (last index is thrown away)

    :returns: (mean shift mprimed, mean shift fprimed, array of shifts with mprimed, array of shifts with fprimed)

    """

    if isinstance(data[0], dict) or isinstance(data[0], Measurement):
        if mode is None:
            raise AttributeError("Must define mode if data is not a list of probability tupes")
        else:
            tuples = primed_completion_differences(data, mode, max_index, plot=False)
    else:
        tuples = data

    pruned_tuples = [t for t in tuples if is_valid_tuple(t)]

    unprimed_fprobs = np.array([t[0][1] for t in pruned_tuples])
    mprimed_fprobs = np.array([t[1][1] for t in pruned_tuples])
    fprimed_fprobs = np.array([t[2][1] for t in pruned_tuples])

    mshift = mprimed_fprobs - unprimed_fprobs
    fshift = fprimed_fprobs - unprimed_fprobs

    mean_mshift = np.nanmean(mshift)
    mean_fshift = np.nanmean(fshift)

    return mean_mshift, mean_fshift, mshift, fshift


def create_feature_table(index: int,
                         data: list[Measurement] | list[dict],
                         role_dict: dict,
                         ratio_df: pd.DataFrame):
    """
    Creates a DataFrame with columns [Measurement, Role, GPrime, POrder] for a *pair of sentences* using measurements provided.

    ONLY OUTPUTS PRIMED INSTANCES.

    :param ratio: whether to exclude instances where target role has no ratio.

    :param data: List of Measurements (or equivalent dictionaries)

    """

    ## TODO: Add case

    fwd_list = get_sent_order([0, 1], get_index(index, data, filter_none=True))
    bwd_list = get_sent_order([1, 0], get_index(index, data, filter_none=True))

    forward_occ = role_dict.get(index)['forward'][0]
    reverse_occ = role_dict.get(index)['reverse'][0]

    fwd_prime_list = [str(m['context']['pnoun_order'][0]) for m in fwd_list]
    bwd_prime_list = [str(m['context']['pnoun_order'][0]) for m in bwd_list]

    fwd_measurement_list = [m['measurement']['BLANK'] for m in fwd_list if m['measurement']['BLANK'] is not None]
    bwd_measurement_list = [m['measurement']['BLANK'] for m in bwd_list if m['measurement']['BLANK'] is not None]

    fwd_measurement_pnoun_order = [str(m['context']['pnoun_order'][1]) for m in fwd_list]
    bwd_measurement_pnoun_order = [str(m['context']['pnoun_order'][1]) for m in bwd_list]

    measurements_df = pd.DataFrame(data={
        'Measurement': fwd_measurement_list + bwd_measurement_list,
        'Role': [forward_occ] * len(fwd_list) + [reverse_occ] * len(bwd_list),
        'GPrime': fwd_prime_list + bwd_prime_list,
        'POrder': fwd_measurement_pnoun_order + bwd_measurement_pnoun_order
    })

    measurements_df["X"] = (measurements_df["Measurement"] == "she").astype(int)
    measurements_df["y_role"] = (measurements_df["Role"] == forward_occ).astype(int)
    measurements_df["y_prime"] = (measurements_df["GPrime"] == "she").astype(int)

    measurements_df["y_ratio"] = [ratio_df['english_mean'][o] for o in measurements_df["Role"]]
    # measurements_df["y_stats"] = [stats["bls_pct_female"][o] for o in measurements_df["Role"]]

    measurements_df["y_order"] = (measurements_df["POrder"]).astype(int)

    measurements_df = measurements_df[measurements_df["GPrime"] != "None"]

    return measurements_df



