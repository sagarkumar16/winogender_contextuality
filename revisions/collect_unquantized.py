"""
Experiment 2 (Reviewer 2, "Quantization") -- collection.

Re-runs the *published* pipeline with unquantized weights. The point of this experiment is
that nothing except the weights changes, so this script does not reimplement the collection
loop: it imports `generate_one_pronoun` / `generate_one_null_context` from
winogender_contextuality.modeling.collect_sequential and calls them with `quantized=False`.
Same items, same prompts, same conditions, same n_runs, same temperature -- by construction.

What it adds on top: model selection from a config file, seeding, and a provenance sidecar
recording the model revision, the quantization setting and library versions next to every
NDJSON.

Sanity check: gemma-3-12b-it was already run unquantized, so re-running it here should
reproduce the existing numbers. `python -m revisions.quantization_compare --sanity-check ...`
does that comparison and flags discrepancies.

Usage:
    python -m revisions.collect_unquantized --config revisions/configs/models_unquantized.yaml \
        --condition primed --start 0 --end 9

- EMNLP revisions, 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from revisions.collect_common import model_provenance, write_run_manifest
from revisions.common import (
    ModelSpec,
    OUTPUTS_DIR,
    REV_LOG_DIR,
    data_dir,
    resolve_models,
    set_seeds,
)

MEASUREMENTS_DIR = OUTPUTS_DIR / "measurements"

CONDITIONS = ("primed", "null")


def run_model(
    spec: ModelSpec,
    condition: str,
    temperature: float,
    n_runs: int,
    start: int,
    end: int | None,
    input_dir: Path,
    output_dir: Path,
    input_file: str,
    seed: int,
    mode: str = "gpu",
) -> Path:
    """Drive one published collection function with this spec's quantization setting."""
    # Imported here, not at module scope: collect_sequential pulls in run_local, which needs a
    # CUDA torch build and transformers >= 4.55. Keeps `--help` usable off-GPU.
    from winogender_contextuality.modeling.collect_sequential import (
        generate_one_null_context,
        generate_one_pronoun,
    )

    tag = "unquant" if not spec.quantized else "quant"
    prefix = "one_pronoun_measurements" if condition == "primed" else "null_measurements"
    output_file = Path(output_dir) / f"{prefix}_{spec.stem}_{temperature}_wp_{tag}.ndjson"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[{condition}] {spec.name} quantized={spec.quantized} rows {start}-{end} -> {output_file}"
    )

    common = dict(
        mode=mode,
        model_name=spec.name,
        temperature=temperature,
        n_runs=n_runs,
        quantized=spec.quantized,
        input_dir=input_dir,
        output_dir=output_dir,
        start=start,
        end=end,
        input_file=input_file,
        output_file=output_file,
    )

    if condition == "primed":
        generate_one_pronoun(assistant=spec.assistant, **common)
    elif condition == "null":
        generate_one_null_context(assistant=spec.assistant, **common)
    else:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}")

    write_run_manifest(
        output_file,
        model_provenance(
            spec,
            seed,
            {
                "experiment": "2_unquantized",
                "condition": condition,
                "temperature": temperature,
                "n_runs": n_runs,
                "rows": [start, end],
                "input_file": input_file,
                "entry_point": (
                    "winogender_contextuality.modeling.collect_sequential."
                    + ("generate_one_pronoun" if condition == "primed" else "generate_one_null_context")
                ),
            },
        ),
    )
    return output_file


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(
        prog="python -m revisions.collect_unquantized",
        description="Re-run the published pipeline on unquantized weights (Experiment 2).",
    )
    parser.add_argument("--config", default=None, help="Model-set YAML (see revisions/configs/).")
    parser.add_argument("--models", nargs="*", default=None, help="Explicit HF model ids.")
    parser.add_argument(
        "--condition",
        default="primed",
        choices=[*CONDITIONS, "both"],
        help="'primed' = the main contextual/unprimed runs; 'null' = the generic null primes.",
    )
    parser.add_argument(
        "--quantized",
        action="store_true",
        default=False,
        help="Collect QUANTIZED instead (for a matched comparison arm on the same seeds).",
    )
    parser.add_argument("--temperature", type=float, default=0.5, help="Matches the published runs.")
    parser.add_argument("--n-runs", type=int, default=50, help="Matches the published runs.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--data-dir", default=None, help="Input dir (pairs TSV / all_sentences CSV).")
    parser.add_argument("--input-file", default=None, help="Defaults per condition.")
    parser.add_argument("--output-dir", default=str(MEASUREMENTS_DIR), help="New results dir.")
    parser.add_argument("--mode", default="gpu", choices=["gpu", "api"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    seed = set_seeds(args.seed) if args.seed is not None else set_seeds()
    logger.add(REV_LOG_DIR / "collect_unquantized_{time}.log")

    specs = resolve_models(args.config, args.models)
    if not specs:
        parser.error("No models selected: pass --config and/or --models.")

    # The flag is the source of truth for this run's arm; a config can still carry it per model.
    if args.quantized:
        specs = [ModelSpec(s.name, s.shorthand, quantized=True, assistant=s.assistant) for s in specs]

    conditions = list(CONDITIONS) if args.condition == "both" else [args.condition]

    written = []
    for spec in specs:
        for condition in conditions:
            input_file = args.input_file or (
                "winopron_pairs.tsv" if condition == "primed" else "all_sentences_wp.csv"
            )
            written.append(
                run_model(
                    spec=spec,
                    condition=condition,
                    temperature=args.temperature,
                    n_runs=args.n_runs,
                    start=args.start,
                    end=args.end,
                    input_dir=data_dir(args.data_dir),
                    output_dir=Path(args.output_dir),
                    input_file=input_file,
                    seed=seed,
                    mode=args.mode,
                )
            )

    logger.success(f"Wrote {len(written)} measurement file(s).")
    return written


if __name__ == "__main__":
    main()
