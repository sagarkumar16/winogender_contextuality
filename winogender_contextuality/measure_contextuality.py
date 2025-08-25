import os
from pathlib import Path
import pandas as pd
import typer
import ast
from itertools import chain, product
from datetime import datetime
from winogender_contextuality.modeling.contextuality import *
from winogender_contextuality.config import *
from winogender_contextuality.utils import *


app = typer.Typer()
HF_KEY = os.environ.get("HF_KEY")

pronoun_order_dict = {
    'mf': 0,
    'fm': 1
}

@app.command()
def measure_contextualities(
        input_fname: str,
        output_fname: str
) -> None:

    """
    Inputs JSON with data formatted like Measurement objects and calculates sentence-order CbD contextuality for all
    pronoun orders, pronoun-order sheaf contextuality & contextual fraction, and pronoun-order CbD contextuality.
    Outputs results as a TSV.

    :param input_fname: filename for input JSON in INTERIM_DATA_DIR
    :return: None
    """

    INPUT_PATH = INTERIM_DATA_DIR / input_fname
    if not INPUT_PATH.exists():
        logger.error(f"Input file {input_fname} does not exist in {INTERIM_DATA_DIR}.")

    OUTPUT_PATH = PROCESSED_DATA_DIR / output_fname
    extension = OUTPUT_PATH.suffix
    if extension not in ['.csv', '.tsv']:
        logger.error(f"Output file {OUTPUT_PATH} must be CSV or TSV.")

    df = pd.DataFrame()

    data = load_ndjson(INPUT_PATH)

    last_index = max([d['index'] for d in data])
    pbar = tqdm(range(last_index))

    sentence_cbd = {'mfmf': [], 'mffm': [], 'fmmf': [], 'fmfm': []}
    pronoun_sheaf_contextuality_bool = []
    pronoun_sheaf_contextuality_status = []
    pronoun_contextual_fraction = []
    pronoun_cbd = []

    processed_indices = []
    results_tracker = {
        'sentence_order': {},  # idx -> {key: result, ...}
        'sheaf_bool': {},  # idx -> result
        'sheaf_status': {},  # idx -> result
        'contextual_fraction': {},  # idx -> result
        'pronoun_nc': {}  # idx -> result
    }

    for idx in pbar:
        processed_indices.append(idx)

        # Initialize tracking for this index
        results_tracker['sentence_order'][idx] = {}

        # Sentence Order Contextualities
        order_keys = ["".join(s) for s in product(pronoun_order_dict.keys(), repeat=2)]
        order_vals = [list(o) for o in product(pronoun_order_dict.values(), repeat=2)]

        for key, val in zip(order_keys, order_vals):
            try:
                measurement_data = sentence_order_results(idx, data, pnoun_order=val)
                sentence_nc_frac = calculate_sentence_nc_fraction(measurement_data)
                results_tracker['sentence_order'][idx][key] = sentence_nc_frac

            except Exception as e:
                logger.error(f"Error on index {idx} for sentence order key {key}: {e}")
                results_tracker['sentence_order'][idx][key] = None  # Mark as failed

        # Pronoun Contextualities
        try:
            ms = MeasurementScenario(observations=['sentence_1', 'sentence_2'],
                                     measurements=['m_first', 'f_first'],
                                     outcomes=[0, 1])
            ms.scenario.values = pronoun_context_array(idx, data)

            # Sheaf contextuality
            try:
                feasibility = check_feasibility(ms)
                results_tracker['sheaf_bool'][idx] = feasibility[0]
                results_tracker['sheaf_status'][idx] = feasibility[1].status
            except Exception as e:
                logger.error(f"Error on index {idx} during sheaf contextuality measurement: {e}")
                results_tracker['sheaf_bool'][idx] = None
                results_tracker['sheaf_status'][idx] = None

            # Contextual fraction
            try:
                contextual_fraction = calculate_contextual_fraction_abramsky(ms)
                results_tracker['contextual_fraction'][idx] = contextual_fraction
            except Exception as e:
                logger.error(f"Error on index {idx} during contextual fraction calculation: {e}")
                results_tracker['contextual_fraction'][idx] = None

            # Pronoun NC fraction
            try:
                pronoun_nc_frac = calculate_pronouns_nc_fraction(ms.scenario.values)
                results_tracker['pronoun_nc'][idx] = pronoun_nc_frac
            except Exception as e:
                logger.error(f"Error on index {idx} during pronoun CbD calculation: {e}")
                results_tracker['pronoun_nc'][idx] = None

        except Exception as e:
            logger.error(f"Error on index {idx} during construction of measurement scenario: {e}")
            # Mark all pronoun-related measurements as failed for this index
            results_tracker['sheaf_bool'][idx] = None
            results_tracker['sheaf_status'][idx] = None
            results_tracker['contextual_fraction'][idx] = None
            results_tracker['pronoun_nc'][idx] = None

    logger.success(f"Completed all contextuality calculations.")

    # Sentence order results
    for key in sentence_cbd.keys():
        sentence_cbd[key] = [
            results_tracker['sentence_order'][idx].get(key, None)
            for idx in processed_indices
        ]

    # Pronoun results
    pronoun_sheaf_contextuality_bool = [
        results_tracker['sheaf_bool'].get(idx, None)
        for idx in processed_indices
    ]

    pronoun_sheaf_contextuality_status = [
        results_tracker['sheaf_status'].get(idx, None)
        for idx in processed_indices
    ]

    pronoun_contextual_fraction = [
        results_tracker['contextual_fraction'].get(idx, None)
        for idx in processed_indices
    ]

    pronoun_cbd = [
        results_tracker['pronoun_nc'].get(idx, None)
        for idx in processed_indices
    ]

    # Add to dataframe - all lists now have the same length
    for key, val in sentence_cbd.items():
        df[f"sentence_{key}"] = val

    df['pronoun_sheaf_bool'] = pronoun_sheaf_contextuality_bool
    df['pronoun_sheaf_status'] = pronoun_sheaf_contextuality_status
    df['pronoun_cf'] = pronoun_contextual_fraction
    df['pronoun_nc'] = pronoun_cbd

    # Log summary of failures
    total_processed = len(processed_indices)
    for measurement_type, results in results_tracker.items():
        if measurement_type == 'sentence_order':
            for key in sentence_cbd.keys():
                failed_count = sum(1 for idx in processed_indices
                                   if results[idx].get(key) is None)
                if failed_count > 0:
                    logger.info(f"Sentence order {key}: {failed_count}/{total_processed} failures")
        else:
            failed_count = sum(1 for idx in processed_indices
                               if results.get(idx) is None)
            if failed_count > 0:
                logger.info(f"{measurement_type}: {failed_count}/{total_processed} failures")

    if extension == '.csv':
        df.to_csv(OUTPUT_PATH, index=True)
    elif extension == '.tsv':
        df.to_csv(OUTPUT_PATH, sep='\t', index=True)
    else:
        raise Exception(f"Extension {extension} is not supported.")

    return


if __name__ == "__main__":
    app()