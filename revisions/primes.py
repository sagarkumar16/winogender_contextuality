"""
Referent-only null primes (Reviewer 3, "Co-Occurrence").

The contextual prime is the paired WinoPron sentence with a gendered pronoun filled in. It
therefore introduces two things at once: a pronoun, and a second mention of the referents.
Reviewer 3 asked whether the shift we attribute to pronoun priming is really just referent
co-occurrence. This module builds a prime that keeps the referents and drops the pronoun,
so the two effects can be separated.

Two styles, both generated programmatically from the existing items:

  repeated_np (default)
      The paired template with its BLANK replaced by the definite NP of its own referent --
      the classic repeated-name control. Lexical content and syntax are held constant with
      respect to the contextual prime; the *only* change is pronoun -> antecedent NP.
          contextual:    "The technician told the customer that she had completed the repair."
          referent-only: "The technician told the customer that the technician had completed the repair."

  conjunction
      A minimal frame naming both referents and nothing else, matched in spirit to the
      generic null primes ("The sky is blue."):
          "The technician and the customer were both present."

Grammatical case is respected via the item's case_* column ($POSS_PRONOUN -> possessive NP).
Every generated prime is checked to contain no pronoun at all; violations are reported, not
silently emitted.

The published prompt-construction code is imported, never edited:
`no_game_seq_logit_prompt` slots these primes in as `fixed_sentence`, exactly where the
contextual and generic-null primes go.

- EMNLP revisions, 2026
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from loguru import logger

# Imported from the published package -- the pronoun inventory and the prompt builder that
# these primes are fed into.
from winogender_contextuality.utils import gendered_pronouns_dict, nb_pronouns_dict
from winogender_contextuality.modeling.prompting import no_game_seq_logit_prompt, role_content_base

from revisions.common import CASE_TO_NP, OUTPUTS_DIR, add_common_args, data_dir, provenance, set_seeds, write_results

PRIME_STYLES = ("repeated_np", "conjunction")

# A personal pronoun in a "referent-only" prime would defeat its purpose: these are the
# tokens that could carry a gender-priming signal, and there must be none.
PERSONAL_PRONOUNS = sorted(
    {p.lower() for pair in gendered_pronouns_dict.values() for p in pair}
    | {p.lower() for pair in nb_pronouns_dict.values() for p in pair}
    | {"hers", "theirs", "himself", "herself", "themselves"}
)
_PRONOUN_RE = re.compile(r"\b(" + "|".join(PERSONAL_PRONOUNS) + r")\b", flags=re.IGNORECASE)

# Inanimate pronouns are a different matter. Some original WinoPron templates contain "it"
# or "its" ("...found it hard to eat enough"), which our primes inherit. They carry no gender
# and appear identically in the contextual prime and in the free sentence, so they are not a
# confound -- we count them for the record rather than treating them as failures.
_INANIMATE_RE = re.compile(r"\b(it|its)\b", flags=re.IGNORECASE)


def referent_np(referent: str, case: str) -> str:
    """Definite NP for a referent in the grammatical slot the pronoun occupied."""
    if case not in CASE_TO_NP:
        raise ValueError(f"Unknown grammatical case {case!r}; expected one of {list(CASE_TO_NP)}")
    return CASE_TO_NP[case].format(referent=str(referent).strip())


def repeated_np_prime(template: str, referent: str, case: str) -> str:
    """Fill the template's BLANK with its referent's NP instead of a pronoun."""
    np_str = referent_np(referent, case)

    # Sentence-initial BLANK needs the NP capitalised.
    if template.lstrip().startswith("BLANK"):
        np_str = np_str[0].upper() + np_str[1:]

    return template.replace("BLANK", np_str)


def conjunction_prime(referent_a: str, referent_b: str) -> str:
    """A minimal sentence naming both referents and nothing else."""
    a, b = str(referent_a).strip(), str(referent_b).strip()
    return f"The {a} and the {b} were both present."


def find_pronouns(sentence: str) -> list[str]:
    """Personal-pronoun tokens left in a sentence. Must be empty for a referent-only prime."""
    return _PRONOUN_RE.findall(sentence)


def find_inanimate_pronouns(sentence: str) -> list[str]:
    """Inanimate 'it'/'its' inherited from the source template. Recorded, not an error."""
    return _INANIMATE_RE.findall(sentence)


def build_referent_primes(
    pairs_df: pd.DataFrame,
    style: str = "repeated_np",
    max_index: int | None = None,
) -> pd.DataFrame:
    """
    Build one row per (pair, free-sentence slot) -- the same unit the null-prime runs use.

    Row order mirrors the paper's aggregation order, which emits the sent_order [0,1] item
    (free sentence = template_2) before the sent_order [1,0] item (free = template_1) for
    each pair. `pair_index` and `free_slot` are carried explicitly so downstream analysis
    joins on metadata rather than trusting row position.

    Columns `template`, `differences` and `case` match the schema that
    generate_one_null_context already consumes; `prime` is the new per-item column.
    """
    if style not in PRIME_STYLES:
        raise ValueError(f"Unknown prime style {style!r}; expected one of {PRIME_STYLES}")

    limit = len(pairs_df) if max_index is None else min(max_index, len(pairs_df))
    rows = []

    for idx in range(limit):
        row = pairs_df.iloc[idx]

        # free_slot = which template is the sentence being completed; the *other* one primes.
        for free_slot, prime_slot in ((2, 1), (1, 2)):
            free_template = row[f"template_{free_slot}"]
            prime_template = row[f"template_{prime_slot}"]
            prime_referent = row[f"referent_{prime_slot}"]
            prime_case = row[f"case_{prime_slot}"]

            if style == "repeated_np":
                prime = repeated_np_prime(prime_template, prime_referent, prime_case)
            else:
                prime = conjunction_prime(row["referent_1"], row["referent_2"])

            leftover = find_pronouns(prime)
            if leftover:
                logger.error(
                    f"pair {idx} slot {free_slot}: referent-only prime still contains personal "
                    f"pronoun(s) {leftover}: {prime!r}"
                )

            rows.append(
                {
                    "pair_index": idx,
                    "free_slot": free_slot,
                    "sent_order": "[0, 1]" if free_slot == 2 else "[1, 0]",
                    "template": free_template,
                    "differences": row[f"differences_{free_slot}"],
                    "case": row[f"case_{free_slot}"],
                    "referent": row[f"referent_{free_slot}"],
                    "prime": prime,
                    "prime_referent": prime_referent,
                    "prime_case": prime_case,
                    "prime_style": style,
                    "has_personal_pronoun": bool(leftover),
                    "has_inanimate_pronoun": bool(find_inanimate_pronouns(prime)),
                }
            )

    out = pd.DataFrame(rows)

    n_bad = int(out.has_personal_pronoun.sum())
    if n_bad:
        logger.error(
            f"{n_bad}/{len(out)} generated primes still contain a personal pronoun -- "
            "inspect before running inference."
        )
    else:
        logger.success(f"All {len(out)} referent-only primes are free of personal pronouns.")

    n_inanimate = int(out.has_inanimate_pronoun.sum())
    if n_inanimate:
        logger.info(
            f"{n_inanimate}/{len(out)} primes contain inanimate 'it'/'its' inherited from the "
            "source template (present in the contextual prime too; not a gender confound)."
        )

    return out


def preview_prompt(row: pd.Series) -> list[dict]:
    """The exact chat prompt a referent-only prime produces, via the published builder."""
    import ast

    options = ast.literal_eval(row["differences"]) if isinstance(row["differences"], str) else row["differences"]
    return role_content_base(
        *no_game_seq_logit_prompt(
            option_set=list(options),
            free_sentence=row["template"],
            fixed_sentence=row["prime"],
        )
    )


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        prog="python -m revisions.primes",
        description="Generate referent-only null primes from the WinoPron items (Experiment 3).",
    )
    add_common_args(parser)
    parser.add_argument("--input-file", default="winopron_pairs.tsv", help="Pairs TSV in --data-dir.")
    parser.add_argument("--style", choices=PRIME_STYLES, default="repeated_np", help="Prime construction style.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV (default: revisions/outputs/primes/referent_primes_<style>.csv).",
    )
    parser.add_argument("--preview", type=int, default=3, help="Log this many example prompts.")
    args = parser.parse_args(argv)

    seed = set_seeds(args.seed)

    in_path = data_dir(args.data_dir) / args.input_file
    if not in_path.exists():
        raise FileNotFoundError(
            f"{in_path} not found. Pass --data-dir (or set $WC_INTERIM_DIR) to the directory "
            "holding winopron_pairs.tsv."
        )

    pairs = pd.read_csv(in_path, sep="\t")
    logger.info(f"Loaded {len(pairs)} pairs from {in_path}")

    out_df = build_referent_primes(pairs, style=args.style, max_index=args.max_index)

    for _, row in out_df.head(args.preview).iterrows():
        logger.info(f"[pair {row.pair_index} slot {row.free_slot}] prime: {row.prime}")
        logger.info(f"    free: {row.template}")

    out_path = Path(args.out) if args.out else OUTPUTS_DIR / "primes" / f"referent_primes_{args.style}.csv"

    meta = provenance(
        {
            "experiment": "3_referent_only_null_primes",
            "prime_style": args.style,
            "input_file": str(in_path),
            "max_index": args.max_index,
            "n_primes_with_personal_pronoun": int(out_df.has_personal_pronoun.sum()),
            "n_primes_with_inanimate_pronoun": int(out_df.has_inanimate_pronoun.sum()),
        },
        seed=seed,
    )
    return write_results(out_df, out_path, meta)


if __name__ == "__main__":
    main()
