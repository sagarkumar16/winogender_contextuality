import os
from pathlib import Path
import pandas as pd
import typer
import ast
from itertools import chain, product
from datetime import datetime
from winogender_contextuality.modeling.prompting import *
from winogender_contextuality.modeling.ModelProbs import *
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

    last_index = data[-1]['index']
    pbar = tqdm(range(last_index))

    sentence_cbd = {'mfmf': [], 'mffm': [], 'fmmf': [], 'fmfm': []}
    pronoun_sheaf_contextuality_bool = []
    pronoun_sheaf_contextuality_status = []
    pronoun_contextual_fraction = []
    pronoun_cbd = []

    for idx in pbar:

        # Sentence Order Contextualities
        order_keys = ["".join(s) for s in product(pronoun_order_dict.keys(), repeat=2)]
        order_vals = [list(o) for o in product(pronoun_order_dict.values(), repeat=2)]
        try:
            for key,val in zip(order_keys, order_vals):
                measurement_data = sentence_order_results(idx, data, pnoun_order=val)
                sentence_nc_frac = calculate_sentence_nc_fraction(measurement_data)
                sentence_cbd[key].append(sentence_nc_frac)

        except Exception as e:
            logger.error(f"Error on index {idx} during sentence order calculation: {e}")

        try:
            # Pronoun Contextualities
            ms = MeasurementScenario(observations=['sentence_1', 'sentence_2'],
                                     measurements=['m_first', 'f_first'],
                                     outcomes=[0,1])
            ms.scenario.values = pronoun_context_array(idx, data)

            try:
                # Sheaf
                feasibility = check_feasibility(ms)
                pronoun_sheaf_contextuality_bool.append(feasibility[0])
                pronoun_sheaf_contextuality_status.append(feasibility[1].status)

                contextual_fraction = calculate_contextual_fraction_abramsky(ms)
                pronoun_contextual_fraction.append(contextual_fraction)

            except Exception as e:
                logger.error(f"Error on index {idx} during sheaf contextuality measurement: {e}")

            try:
                pronoun_nc_frac = calculate_pronouns_nc_fraction(ms.scenario.values)
                pronoun_cbd.append(pronoun_nc_frac)

            except Exception as e:
                logger.error(f"Error on index {idx} during pronoun CbD calculation: {e}")

        except Exception as e:
            logger.error(f"Error on index {idx} during construction of measurement scenario: {e}")

    for key,val in sentence_cbd.items():
        df[f"sentence_{key}"] = val
    df['pronoun_sheaf_bool'] = pronoun_sheaf_contextuality_bool
    df['pronoun_sheaf_status'] = pronoun_sheaf_contextuality_status
    df['pronoun_cf'] = pronoun_contextual_fraction
    df['pronoun_nc'] = pronoun_cbd

    logger.success(f"Completed all contextuality calculations.")

    if extension == '.csv':
        df.to_csv(OUTPUT_PATH, index=True)
    elif extension == '.tsv':
        df.to_csv(OUTPUT_PATH, sep='\t', index=True)
    else:
        raise Exception(f"Extension {extension} is not supported.")

    return


if __name__ == "__main__":
    app()