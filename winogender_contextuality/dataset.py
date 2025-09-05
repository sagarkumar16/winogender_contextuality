from pathlib import Path
from loguru import logger
from tqdm import tqdm
from collections import namedtuple
import pandas as pd
import re
import pickle
import typer

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

gendered_pronouns_dict = {
    '$POSS_PRONOUN': ['his', 'her'],
    '$NOM_PRONOUN': ['he', 'she'],
    '$ACC_PRONOUN': ['him', 'her']
}

nb_pronouns_dict = {
    '$POSS_PRONOUN': ['their', 'xyr'],
    '$NOM_PRONOUN': ['they', 'xe'],
    '$ACC_PRONOUN': ['them', 'xem']
}


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "templates.tsv",
    output_path: Path = INTERIM_DATA_DIR / "wino_pairs.tsv",
    sentence_col: str = "sentence",
    nb: bool = False
) -> None:
    """
    Takes templates.tsv from WinoGender or WinoPron schemas and writes a TSV formatted to be compatible with local
    scripts
    :param input_path: input filename
    :param output_path: output filename
    :param sentence_col: column name of sentences
    :param nb: whether to use nonbinary pronouns
    :return: None
    """
    logger.info(f"Processing {input_path}...")
    df = pd.read_csv(input_path, sep="\t")

    if nb:
        pronouns_dict = nb_pronouns_dict
    else:
        pronouns_dict = gendered_pronouns_dict

    new_rows = []
    pbar = tqdm(range(0, len(df), 2))

    for i in pbar:
        chunk = df[i:i+2].reset_index(drop=True)
        new_row = {}

        for idx in chunk.index:
            s = chunk[sentence_col][idx]
            s = s.replace('$OCCUPATION', f"{chunk['occupation(0)'][idx]}")
            s = s.replace('$PARTICIPANT', f"{chunk['other-participant(1)'][idx]}")
            match = re.search(r'\$[a-zA-Z_]\w*', s)
            if match:
                pronoun_type = match.group(0)
                s = s.replace(pronoun_type, 'BLANK')
                new_row[f"template_{idx+1}"] = s
                new_row[f"differences_{idx+1}"] = pronouns_dict[pronoun_type]
            else:
                logger.warning(f"Could not find a pronoun in {s}")

            if df.answer[idx]:
                new_row[f"referent_{idx+1}"] = chunk['other-participant(1)'][idx]
            else:
                new_row[f"referent_{idx+1}"] = chunk['occupation(0)'][idx]

        new_rows.append(new_row)

    # Adding a 61st test row
    test_row = new_rows[-1].copy()
    test_row['differences_1'] = ['she', 'potato']
    test_row['differences_2'] = ['she', 'potato']
    new_rows.append(test_row)

    output_df = pd.DataFrame(new_rows)
    output_df.to_csv(output_path, sep="\t", index=False)
    logger.success(f"Wrote processed dataset to {output_path}.")

    return



if __name__ == "__main__":
    app()
