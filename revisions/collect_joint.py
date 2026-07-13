"""
Experiment 1 (Reviewer 1, "Steering") -- collection of JOINT measurements.

The published runs are *steering* measurements: the priming pronoun is fixed by the
experimenter, so in each context only the free pronoun is actually measured. Here both
pronouns of a sentence pair are measured in a single pass (BLANK1 and BLANK2, filled
simultaneously), which is the joint-measurement setting the QQ equality and the
social-science contextuality literature use.

Why a new collector rather than the published `generate_two_pronouns`: that function builds
its `Context` without `case_1`/`case_2`, which became required fields of the dataclass, so it
raises TypeError before writing a single record. It is left untouched; the prompt itself is
still built by the published `no_game_seq_prompt`.

Output schema (Measurement NDJSON, as usual):
    context.sent_order  = (s0, s1)   presented sentence order, e.g. (0, 1)
    context.pnoun_order = (i, j)     option order for BLANK1 and BLANK2 (0 = male-first)
    measurement         = {'BLANK1': <pronoun>, 'BLANK2': <pronoun>}
    index               = pair index in the pairs TSV

BLANK1 always belongs to the sentence *presented first*, so in sent_order (1,0) BLANK1 is
template_2's pronoun. revisions.joint_measurement relies on that convention.

Usage:
    python -m revisions.collect_joint --model google/gemma-3-12b-it --n-runs 50 --start 0 --end 9

- EMNLP revisions, 2026
"""

from __future__ import annotations

import argparse
import ast
from itertools import permutations
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from winogender_contextuality.config import INTERIM_DATA_DIR
from winogender_contextuality.modeling.prompting import no_game_seq_prompt, role_content_base

from revisions.collect_common import (
    Context,
    Measurement,
    append_measurement,
    decode_completion,
    default_output_path,
    load_model,
    model_provenance,
    parse_blank,
    write_run_manifest,
)
from revisions.common import ModelSpec, REV_LOG_DIR, data_dir, set_seeds

JOINT_KEYS = ("BLANK1", "BLANK2")


def collect(
    spec: ModelSpec,
    pairs: pd.DataFrame,
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
    Joint two-pronoun collection over pair rows [start, end].

    For every pair we cross both sentence orders with both option orders for each blank:
    2 sentence orders x 2 option orders (BLANK1) x 2 option orders (BLANK2). The four
    (i, j) combinations within a sentence order are the contexts of the rank-4 cyclic
    system; the two sentence orders are the contexts of the rank-2 system that lines up
    with the steering estimator.
    """
    output_fpath = Path(output_fpath)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(pairs)
    start = max(0, start)
    end_exclusive = n_rows if (end is None or end >= n_rows) else end + 1
    if start >= end_exclusive:
        logger.warning(f"Nothing to do: start={start}, end={end}, n_rows={n_rows}")
        return output_fpath

    mp = None if dry_run else load_model(spec, mode=mode)

    pbar = tqdm(range(start, end_exclusive), desc=f"joint {spec.shorthand} [{start}-{end_exclusive - 1}]")

    for idx in pbar:
        row = pairs.iloc[idx]
        sentences = {0: row["template_1"], 1: row["template_2"]}
        pronouns = {0: ast.literal_eval(row["differences_1"]), 1: ast.literal_eval(row["differences_2"])}
        cases = {0: row["case_1"], 1: row["case_2"]}

        for s_perm in permutations(sentences.keys()):  # (0,1) then (1,0)
            s1, s2 = sentences[s_perm[0]], sentences[s_perm[1]]
            p1, p2 = pronouns[s_perm[0]], pronouns[s_perm[1]]
            case1, case2 = cases[s_perm[0]], cases[s_perm[1]]

            for i, p1_perm in enumerate(permutations(p1)):
                for j, p2_perm in enumerate(permutations(p2)):
                    error_count = 0

                    for _ in range(n_runs):
                        prompt = role_content_base(
                            *no_game_seq_prompt([list(p1_perm), list(p2_perm)], [s1, s2])
                        )

                        if dry_run:
                            json_output = {"BLANK1": "DRYRUN", "BLANK2": "DRYRUN"}
                        else:
                            inputs, output = mp.get_completion(
                                prompt=prompt, temperature=temperature, max_new_tokens=12
                            )
                            decoded = decode_completion(mp, spec.name, inputs, output)
                            json_output = parse_blank(decoded, keys=JOINT_KEYS)
                            if json_output.get("BLANK1") == "None":
                                error_count += 1

                        m = Measurement(
                            index=int(idx),
                            context=Context(
                                sent_order=list(s_perm),
                                pnoun_order=(i, j),
                                sentence_1=s1,
                                sentence_2=s2,
                                # Canonical option lists, so position 1 is female in analysis.
                                pronouns_1=list(p1),
                                pronouns_2=list(p2),
                                case_1=case1,
                                case_2=case2,
                            ),
                            measurement=json_output,
                            probabilities=None,
                            logits=None,
                        )
                        append_measurement(output_fpath, m)

                    if error_count:
                        logger.warning(
                            f"idx {idx} order {s_perm} options ({i},{j}): "
                            f"{error_count}/{n_runs} unparseable."
                        )

    logger.success(f"Collected pairs {start}-{end_exclusive - 1} into {output_fpath}")
    return output_fpath


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        prog="python -m revisions.collect_joint",
        description="Collect joint (two-pronoun) measurements (Experiment 1).",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model id.")
    parser.add_argument("--shorthand", default=None)
    parser.add_argument("--input-file", default="winopron_pairs.tsv")
    parser.add_argument("--data-dir", default=None, help="Directory holding the pairs TSV.")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--start", type=int, default=0, help="First pair index (inclusive).")
    parser.add_argument("--end", type=int, default=None, help="Last pair index (inclusive).")
    parser.add_argument("--quantized", action="store_true", default=False)
    parser.add_argument("--mode", default="gpu", choices=["gpu", "api"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=str(INTERIM_DATA_DIR))
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Build prompts/records with no model.")
    args = parser.parse_args(argv)

    seed = set_seeds(args.seed) if args.seed is not None else set_seeds()
    logger.add(REV_LOG_DIR / "collect_joint_{time}.log")

    spec = ModelSpec(
        name=args.model,
        shorthand=args.shorthand or args.model.split("/")[-1],
        quantized=args.quantized,
    )

    pairs = pd.read_csv(data_dir(args.data_dir) / args.input_file, sep="\t")
    logger.info(f"Loaded {len(pairs)} pairs")

    out = (
        Path(args.output_file)
        if args.output_file
        else default_output_path(Path(args.output_dir), "joint_measurements", spec, args.temperature, tag="wp")
    )

    collect(
        spec=spec,
        pairs=pairs,
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
                "experiment": "1_joint_measurement",
                "formulation": "joint",
                "input_file": args.input_file,
                "temperature": args.temperature,
                "n_runs": args.n_runs,
                "pairs": [args.start, args.end],
                "dry_run": args.dry_run,
            },
        ),
    )
    return out


if __name__ == "__main__":
    main()
