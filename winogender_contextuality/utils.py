import gc
import torch
import typer
from scipy.special import softmax
import ast
import json
from collections import defaultdict, Counter
import numpy as np
from scipy.special import softmax
from scipy.spatial import distance
from dataclasses import dataclass, asdict
from loguru import logger
import pandas as pd
from winogender_contextuality.config import *

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
    probs = softmax(z, dim=0)

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

def get_role_dict(
        templates_path: str = RAW_DATA_DIR / "templates.tsv") -> dict:

    """
    Creates a dictionary with role (i.e. occupation or roles like "customer") for the referent in the blank sentence
    and priming sentence.

    :param templates_path: path to templates file
    :param pairs_path: path to pairs file
    :return: dict with shape {idx: {'forward': (referent occupation, priming occupation),
         'reverse': (referent occupation, priming occupation)}
    """

    templates = pd.read_csv(templates_path, sep='\t')
    max_index = max(tempates.index)

    pnoun_role_dict = defaultdict(dict)
    for idx in range(max_index):
        forward_sentence_idx = int((2 * idx) + 1)
        reverse_sentence_idx = int(2 * idx)

        answer_mapping = {0: 'occupation(0)', 1: 'other-participant(1)'}

        forward_ref_col = answer_mapping[templates['answer'][forward_sentence_idx]]
        forward_other_col = answer_mapping[(templates['answer'][forward_sentence_idx] + 1) % 2]
        reverse_ref_col = answer_mapping[templates['answer'][reverse_sentence_idx]]
        reverse_other_col = answer_mapping[(templates['answer'][reverse_sentence_idx] + 1) % 2]

        pnoun_role_dict[idx]['forward'] = (templates[forward_ref_col][forward_sentence_idx],
                                           templates[forward_other_col][forward_sentence_idx])
        pnoun_role_dict[idx]['reverse'] = (templates[reverse_ref_col][reverse_sentence_idx],
                                           templates[reverse_other_col][reverse_sentence_idx])

    return pnoun_role_dict


# First we get a partition
def get_index(index: int,
              data: list[Measurement] | list[dict],
              filter_none: bool= True) -> list[Measurement] | list[dict]:
    """
    Filters list of Measurements (or equivalent dictionaries) by index.

    :param index: index of measurement to return
    :param data: list of Measurements
    :param filter_none: whether to filter out instances with erroneous responses
    :return: list of Measurements
    """

    all_measurements = [d for d in data if d['index'] == index]

    if filter_none:
        string_only = [d for d in all_measurements if isinstance(d['measurement'], dict)]
        no_none = [d for d in string_only if d['measurement']['BLANK'] != "None"]
        return no_none
    else:
        return all_measurements


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


def get_filled_pnoun(pnoun_index: int,
                     data: list[Measurement] | list[dict]) -> list[Measurement] | list[dict]:
    """
    Filters list of Measurements (or equivalent dictionaries) to get instances where first sentence is filled with the
    designated pronoun.

    :param pnoun_index: pronoun as a string with no spaces
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
    try:
        pnouns = measurements[0]['context']['pronouns_2']
    except Exception as e:
        logger.debug(f"Likely no measurements found. Exception raised: {e}")
        return np.nan

    # calculate empirical generation probabilities (remove anything not in the list of pronouns)
    generated_pnouns = []
    for m in measurements:
        try:
            generated_pnouns.append(m['measurement']['BLANK'])
        except Exception as e:
            logger.debug(f"Exception {e} raised for item {m}")
            pass
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


def get_sentence_pnoun_order(idx: int,
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
    :param mode: either 'internal' or 'generation'
    :param pnoun_order: filtered pronoun order--NOTE: this uses default_pronoun indexing
    :param default_pronoun: pronoun index for probabiity calculations (default = 1 for female)
    :return:
    """
    sentence_measurements = [d for d in model_measurements if d['index'] == idx]
    test_measurements = [d for d in sentence_measurements if (d['context']['pnoun_order'][1] == pnoun_order) and
                         (d['context']['pnoun_order'][0] is not None)]

    data_dict = {'forward': {'fixed_pnoun': [], 'free_pnoun': [], 'pronouns': []},
                 'reverse': {'fixed_pnoun': [], 'free_pnoun': [], 'pronouns': []}}

    if mode == 'internal':
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

        return data_dict

    elif mode == 'generation':
        for d in test_measurements:

            context = d['context']['sent_order']

            if context == [0, 1]:
                dict_key = 'forward'
                fixed_pnoun = d['context']['pnoun_order'][0]
                free_pnoun = d['measurement']['BLANK']
                default_p1 = d['context']['pronouns_1'][default_pronoun]
                default_p2 = d['context']['pronouns_2'][default_pronoun]
            elif context == [1, 0]:
                dict_key = 'reverse'
                fixed_pnoun = d['context']['pnoun_order'][0]
                free_pnoun = d['measurement']['BLANK']
                default_p1 = d['context']['pronouns_2'][default_pronoun]
                default_p2 = d['context']['pronouns_1'][default_pronoun]
            else:
                raise AttributeError

            data_dict[dict_key]['fixed_pnoun'].append(fixed_pnoun.lower())
            data_dict[dict_key]['free_pnoun'].append(free_pnoun.lower())
            data_dict[dict_key]['pronouns'] = [default_p1, default_p2]

        return data_dict

    else:
        raise AttributeError



# Sentence-order contextuality via CBD
def sentence_order_results_single_gen(idx: int,
                                      model_measurements: list[dict],
                                      pnoun_order: list[int] = [0, 0],
                                      default_pronoun: int = 1  # 0 for male, 1 for female
                                      ) -> dict:
    """
    Creates a dictionary given a subset of measurements for a single pair of sentences which summarizes the results of
    all runs, separated on the basis of sentence order, filtered for only one pronoun order.

    :param idx: integer index sentence pair
    :param model_measurements: list of Measurement objects (or equivalently structured dictionaries)
    :param pnoun_order: filtered pronoun order--NOTE: this uses default_pronoun indexing
    :param default_pronoun: pronoun index for probabiity calculations (default = 1 for female)
    :return:
    """
    sentence_measurements = [d for d in model_measurements if d['index'] == idx]

    pnoun_filled_fwd = [sentence_measurements[0]['context']['pronouns_1'][pnoun_order[0]], pnoun_order[1]]
    pnoun_filled_bwd = [sentence_measurements[0]['context']['pronouns_2'][pnoun_order[0]], pnoun_order[1]]

    test_measurements = [d for d in sentence_measurements if d['context']['pnoun_order'] == pnoun_filled_fwd or
                         d['context']['pnoun_order'] == pnoun_filled_bwd]

    data_dict = {'forward': {'fixed_pnoun': [], 'free_pnoun': [], 'pronouns': []},
                 'reverse': {'fixed_pnoun': [], 'free_pnoun': [], 'pronouns': []}}

    for d in test_measurements:

        context = d['context']['sent_order']

        if context == [0, 1]:
            dict_key = 'forward'
            fixed_pnoun = d['context']['pnoun_order'][0]
            free_pnoun = d['measurement']['BLANK']
            default_p1 = d['context']['pronouns_1'][default_pronoun]
            default_p2 = d['context']['pronouns_2'][default_pronoun]
        elif context == [1, 0]:
            dict_key = 'reverse'
            fixed_pnoun = d['context']['pnoun_order'][0]
            free_pnoun = d['measurement']['BLANK']
            default_p1 = d['context']['pronouns_2'][default_pronoun]
            default_p2 = d['context']['pronouns_1'][default_pronoun]
        else:
            raise AttributeError

        data_dict[dict_key]['fixed_pnoun'].append(fixed_pnoun.lower())
        data_dict[dict_key]['free_pnoun'].append(free_pnoun.lower())
        data_dict[dict_key]['pronouns'] = [default_p1, default_p2]

    return data_dict


