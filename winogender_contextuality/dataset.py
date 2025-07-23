from pathlib import Path
from loguru import logger
from tqdm import tqdm
from collections import namedtuple
import pandas as pd
import re
import pickle
import typer

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

pronouns_dict = {
    '$POSS_PRONOUN': ['his', 'her'],
    '$NOM_PRONOUN': ['he', 'she'],
    '$ACC_PRONOUN': ['him', 'her']
}


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "templates.tsv",
    output_path: Path = PROCESSED_DATA_DIR / "wino_pairs.tsv",
):
    logger.info(f"Processing {input_path}...")
    df = pd.read_csv(input_path, sep="\t")

    new_rows = []
    pbar = tqdm(range(0, len(df), 2))

    for i in pbar:
        chunk = df[i:i+2].reset_index(drop=True)
        new_row = {}

        for idx in chunk.index:
            s = df.sentence[idx]
            s = s.replace('$OCCUPATION', f"{df['occupation(0)'][idx]}")
            s = s.replace('$PARTICIPANT', f"{df['other-participant(1)'][idx]}")
            match = re.search(r'\$[a-zA-Z_]\w*', s)
            if match:
                pronoun_type = match.group(0)
                s = s.replace(pronoun_type, 'BLANK')
                new_row[f"template_{idx+1}"] = s
                new_row[f"differences_{idx+1}"] = pronouns_dict[pronoun_type]
            else:
                logger.warning(f"Could not find a pronoun in {s}")

            if df.answer[idx]:
                new_row[f"referent_{idx+1}"] = df['other-participant(1)'][idx]
            else:
                new_row[f"referent_{idx+1}"] = df['occupation(0)'][idx]

        new_rows.append(new_row)

    output_df = pd.DataFrame(new_rows)
    output_df.to_csv(output_path, sep="\t", index=False)
    logger.success(f"Wrote processed dataset to {output_path}.")

    return



if __name__ == "__main__":
    app()
