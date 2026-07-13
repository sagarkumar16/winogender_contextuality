"""
Experiment 3 (Reviewer 3, "Co-Occurrence") -- collection.

Runs the free sentence against a *referent-only* prime: a prime that names the referents but
contains no pronoun (see revisions/primes.py). Structurally this mirrors
`generate_one_null_context`, but the prime varies per item rather than coming from the three
fixed sentences in config.null_sentences, so it needs its own loop. The prompt itself is
built by the published `no_game_seq_logit_prompt`, unchanged.

Output schema matches the published Measurement NDJSON, so the existing filters work:
    context.sent_order  = ["refnull", 0]
    context.pnoun_order = ("refnull", n)   n = presented option order (0 = male-first)
    context.sentence_1  = the referent-only prime
    context.sentence_2  = the free sentence
    index               = row index of the primes CSV (joins back to pair_index / free_slot)

Unlike the published null run, `logits` here is a genuine 2-vector over the canonical
(male, female) options -- see revisions/collect_common.py.

Usage:
    python -m revisions.collect_referent_null --model google/gemma-3-12b-it \
        --primes revisions/outputs/primes/referent_primes_repeated_np.csv \
        --n-runs 50 --start 0 --end 9

- EMNLP revisions, 2026
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from winogender_contextuality.config import INTERIM_DATA_DIR
from winogender_contextuality.modeling.prompting import no_game_seq_logit_prompt, role_content_base

from revisions.collect_common import (
    Context,
    Measurement,
    append_measurement,
    decode_completion,
    default_output_path,
    load_model,
    model_provenance,
    parse_blank,
    pronoun_logits,
    write_run_manifest,
)
from revisions.common import ModelSpec, OUTPUTS_DIR, REV_LOG_DIR, set_seeds


def collect(
    spec: ModelSpec,
    primes: pd.DataFrame,
    output_fpath: Path,
    temperature: float,
    n_runs: int,
    start: int,
    end: int | None,
    seed: int,
    mode: str = "gpu",
    dry_run: bool = False,
) -> Path:
    """
    One (model x prime-set) collection pass over rows [start, end] of the primes table.

    With dry_run=True no weights are loaded and no logits are produced: the prompts and the
    record schema are still built and written, which is how the pipeline is checked off-GPU.
    """
    output_fpath = Path(output_fpath)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(primes)
    start = max(0, start)
    end_exclusive = n_rows if (end is None or end >= n_rows) else end + 1
    if start >= end_exclusive:
        logger.warning(f"Nothing to do: start={start}, end={end}, n_rows={n_rows}")
        return output_fpath

    mp = None if dry_run else load_model(spec, mode=mode)

    indices = list(range(start, end_exclusive))
    pbar = tqdm(indices, desc=f"referent-null {spec.shorthand} [{start}-{end_exclusive - 1}]")

    for idx in pbar:
        row = primes.iloc[idx]
        canonical_pronouns = ast.literal_eval(row["differences"])  # e.g. ['he', 'she']
        free_sentence = row["template"]
        prime = row["prime"]
        case = row["case"]

        # Both presented option orders, as in the published null run.
        for n, p_list in enumerate([canonical_pronouns, canonical_pronouns[::-1]]):
            error_count = 0

            for _ in range(n_runs):
                prompt = role_content_base(
                    *no_game_seq_logit_prompt(
                        option_set=list(p_list),
                        free_sentence=free_sentence,
                        fixed_sentence=prime,
                        assistant=spec.assistant,
                    )
                )

                if dry_run:
                    logits = None
                    json_output = {"BLANK": "DRYRUN"}
                else:
                    # NOTE: canonical list, so position 1 is female. (The published null run
                    # passes pronouns[1] here, which indexes the characters of "she".)
                    logits = pronoun_logits(mp, prompt, canonical_pronouns)

                    inputs, output = mp.get_completion(
                        prompt=prompt,
                        temperature=temperature,
                        max_new_tokens=12,
                        continue_final_message=spec.assistant,
                    )
                    decoded = decode_completion(mp, spec.name, inputs, output)
                    json_output = parse_blank(decoded)
                    if json_output.get("BLANK") == "None":
                        error_count += 1

                m = Measurement(
                    index=int(idx),
                    context=Context(
                        sent_order=["refnull", 0],
                        pnoun_order=("refnull", n),
                        sentence_1=prime,
                        sentence_2=free_sentence,
                        pronouns_1="refnull",
                        pronouns_2=list(canonical_pronouns),
                        case_1="refnull",
                        case_2=case,
                    ),
                    measurement=json_output,
                    probabilities=None,
                    logits=logits,
                )
                append_measurement(output_fpath, m)

            if error_count:
                logger.warning(f"idx {idx} order {n}: {error_count}/{n_runs} unparseable.")

    logger.success(f"Collected rows {start}-{end_exclusive - 1} into {output_fpath}")
    return output_fpath


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        prog="python -m revisions.collect_referent_null",
        description="Collect referent-only null-prime measurements (Experiment 3).",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model id.")
    parser.add_argument("--shorthand", default=None, help="Short name for filenames.")
    parser.add_argument(
        "--primes",
        default=str(OUTPUTS_DIR / "primes" / "referent_primes_repeated_np.csv"),
        help="Primes CSV from `python -m revisions.primes`.",
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--start", type=int, default=0, help="First primes-CSV row (inclusive).")
    parser.add_argument("--end", type=int, default=None, help="Last primes-CSV row (inclusive).")
    parser.add_argument("--quantized", action="store_true", default=False)
    parser.add_argument("--no-assistant", dest="assistant", action="store_false", default=True)
    parser.add_argument("--mode", default="gpu", choices=["gpu", "api"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=str(INTERIM_DATA_DIR))
    parser.add_argument("--output-file", default=None, help="Explicit shared NDJSON path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and records without loading a model (schema/prompt check).",
    )
    args = parser.parse_args(argv)

    seed = set_seeds(args.seed) if args.seed is not None else set_seeds()
    logger.add(REV_LOG_DIR / "collect_referent_null_{time}.log")

    spec = ModelSpec(
        name=args.model,
        shorthand=args.shorthand or args.model.split("/")[-1],
        quantized=args.quantized,
        assistant=args.assistant,
    )

    primes = pd.read_csv(args.primes)
    logger.info(f"Loaded {len(primes)} referent-only primes from {args.primes}")

    out = (
        Path(args.output_file)
        if args.output_file
        else default_output_path(Path(args.output_dir), "refnull_measurements", spec, args.temperature, tag="wp")
    )

    collect(
        spec=spec,
        primes=primes,
        output_fpath=out,
        temperature=args.temperature,
        n_runs=args.n_runs,
        start=args.start,
        end=args.end,
        seed=seed,
        mode=args.mode,
        dry_run=args.dry_run,
    )

    write_run_manifest(
        out,
        model_provenance(
            spec,
            seed,
            {
                "experiment": "3_referent_only_null_primes",
                "condition": "referent_null",
                "primes_file": str(args.primes),
                "prime_style": str(primes["prime_style"].iloc[0]) if len(primes) else None,
                "temperature": args.temperature,
                "n_runs": args.n_runs,
                "rows": [args.start, args.end],
                "dry_run": args.dry_run,
            },
        ),
    )
    return out


if __name__ == "__main__":
    main()
