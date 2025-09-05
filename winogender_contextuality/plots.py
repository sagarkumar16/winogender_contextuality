from pathlib import Path
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
import typer

from winogender_contextuality.config import FIGURES_DIR, PROCESSED_DATA_DIR
from winogender_contextuality.utils import *

app = typer.Typer()


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

        differences.append(tup1)
        differences.append(tup2)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(2 * max_index), [t[0][1] for t in differences], label='unprimed', marker='x',
                linestyle='None', color='gray',
                markersize=5)
        ax.plot(range(2 * max_index), [t[1][1] for t in differences], label='m primed', marker='o',
                linestyle='None', color='#E69F00',
                markersize=5)
        ax.plot(range(2 * max_index), [t[2][1] for t in differences], label='f primed', marker='o',
                linestyle='None', color='#009E73',
                markersize=5)
        ax.legend()
        for subidx in range(2 * max_index):
            min_idx, med_idx, max_idx = np.argsort([i[1] for i in differences[subidx]])

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

    unprimed_fprobs = np.array([t[0][1] for t in tuples])
    mprimed_fprobs = np.array([t[1][1] for t in tuples])
    fprimed_fprobs = np.array([t[2][1] for t in tuples])

    mshift = mprimed_fprobs - unprimed_fprobs
    fshift = fprimed_fprobs - unprimed_fprobs

    mean_mshift = np.nanmean(mshift)
    mean_fshift = np.nanmean(fshift)

    return mean_mshift, mean_fshift, mshift, fshift

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
