import gc
import torch
import typer
import torch.nn.functional as F
import ast
import json
from collections import defaultdict, Counter
import numpy as np
from scipy.special import softmax
from scipy.spatial import distance
from dataclasses import dataclass, asdict
from loguru import logger

app = typer.Typer()

@dataclass
class Context:
    sent_order: tuple[int, int] # denotes the order of the sentences (e.g. (1, 0) => the order was reversed)
    pnoun_order: tuple[int, int] # denotes the order of the pronouns (e.g. (0, 1) => second set of pronouns was reversed)
    sentence_1: str
    sentence_2: str
    pronouns_1: list[str]
    pronouns_2: list[str]

@dataclass
class Measurement:
    index: int
    context: Context
    measurement: dict[str, str]
    probabilities: tuple[float] | None
    logits: tuple[float] | None

@app.command()
def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()

def masked_softmax(token_ids: list[int],
                   logits: torch.Tensor) -> torch.Tensor:
    """
    Calculates probabilities based on a masked softmax which only considers the tokens in token_ids so that the sum
    of their probabilities is equal to 1.

    :param token_ids: list of selected token IDs
    :param logits: logits tensor from get_raw_logits() or get_completed_logits()
    :return: tensor containing normalized probabilities
    """

    z = logits[token_ids]
    probs = F.softmax(z, dim=0)

    return probs

def reverse_pronouns(options: str, 
                     measurement: str,
                     prime: str = "she_first") -> list[str]:

    """
    Takes in a pronoun ist string from the processed TSV file and returns the pronouns as a list in the correct order.
    
    :param options: list stored as a string (e.g. "["he", "she"]")
    :param measurement: measurement context (e.g. "he_first")
    :param prime: which measurement context needs to be reversed
    """

    pronouns = ast.literal_eval(options)
    if measurement==prime:
        return pronouns[::-1]
    else:
        return pronouns

def load_ndjson(data_path) -> list[dict]:

    """
    Loads ndjson files into memory as dictionaries.
    :param data_path: path to ndjson file
    :return: list of dictionaries
    """

    data = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():  # avoid empty lines
                data.append(json.loads(line))

    return data


# First we get a partition
def get_index(index: int,
              data: list[Measurement] | list[dict]) -> list[Measurement] | list[dict]:
    """
    Filters list of Measurements (or equivalent dictionaries) by index.

    :param index: index of measurement to return
    :param data: list of Measurements
    :return: list of Measurements
    """

    return [d for d in data if d['index'] == index]


def get_sent_order(order: list[int],
                   data: list[Measurement] | list[dict]) -> list[Measurement] | list[dict]:
    """
    Filters list of Measurements (or equivalent dictionaries) by sentence order

    :param order: sentence order as a list (e.g. [0,1] for forward and [0,1] for backwards)
    :param data: list of Measurements
    :return: list of Measurements
    """

    return [d for d in data if d['context']['sent_order'] == order]


def get_single_sentences(data: list[Measurement] | list[dict]) -> list[Measurement] | list[dict]:

    """
    Filters list of Measurements (or equivalent dictionaries) to get instances with no first sentence.

    :param data: list of Measurements
    :return: list of Measurements
    """

    return [d for d in data if d['context']['sentence_1'] == None]


def get_filled_pnoun(pnoun_index: str,
                     data: list[Measurement] | list[dict]) -> list[Measurement] | list[dict]:
    """
    Filters list of Measurements (or equivalent dictionaries) to get instances where first sentence is filled with the
    designated pronoun.

    :param pnoun: pronoun as a string with no spaces
    :param data: list of Measurements
    :return: list of Measurements.
    """

    return [d for d in data if d['context']['pnoun_order'][0] == d['context']['pronouns_1'][pnoun_index]]


def get_pnoun_order(pnoun_order_index: int,
                    data: list[Measurement] | list[dict]) -> list[Measurement] | list[dict]:
    """
    Filters list of Measurements (or equivalent dictionaries) to get instances where pronouns are presented in
    the order designated by pronoun_order_index (i.e. 0 = male pronoun first)
    """

    return [d for d in data if d['context']['pnoun_order'][1] == pnoun_order_index]


def get_internal_probs(measurements: list[Measurement] | list[dict]) -> np.ndarray:
    """
    Calculates probabilities based on mean logits from a list of Measurements (or equivalent dictionaries).

    :param measurements: list of Measurement objects (or equivalant dictionaries)
    :return: array of probabilities
    """

    # calculate probs from mean logits (should be all the same, though)
    internal_probs = softmax(np.mean([np.array(l['logits']) for l in measurements], axis=0))

    return internal_probs


def get_generation_probs(measurements: list[Measurement] | list[dict]) -> np.ndarray:
    """
    Calculates probabilities based on empirical generation frequencies from a list of
     Measurements (or equivalent dictionaries).

    :param measurements: list of Measurement objects (or equivelant dictionaries)
    :return: array of probabilities
    """
    # pronoun set is determined by the first measurement
    pnouns = measurements[0]['context']['pronouns_2']

    # calculate empirical generation probabilities (remove anything not in the list of pronouns)
    generated_pnouns = []
    for m in measurements:
        try:
            generated_pnouns.append(m['measurement']['BLANK'])
        except Exception as e:
            logger.debug(f"Exception {e} raised for item {m}")
    generation_counter = Counter(generated_pnouns)
    generation_counter_clean = {k: generation_counter[k] for k in pnouns}
    num_valid_measurements = np.sum(list(generation_counter_clean.values()))
    generation_probs = np.array(list(generation_counter_clean.values())) / num_valid_measurements

    return generation_probs


def get_generation_logit_distortion(measurements: list[Measurement] | list[dict]):
    """
    Measure of the difference between the (deterministic) internal beliefs of a model and the (observed) generation
    frequency of those same tokens.

    Distance is measured as L1 (Manhattan) Distance because this a discrete, unordered distribution.

    :param measurements: list of Measurement objects (or equivelant dictionaries)
    """

    internal_probs = get_internal_probs(measurements)
    generation_probs = get_generation_probs(measurements)

    # L1 distance of the two arrays
    return distance.cityblock(internal_probs, generation_probs)
    