"""
Shared infrastructure for the revision experiments: seeding, provenance logging,
result IO, and the count/probability helpers used by every analysis.

Nothing here modifies the original package. Helpers that already exist in
``winogender_contextuality`` are imported rather than reimplemented; the few
functions defined here are either (a) new, or (b) ordering-robust variants of
notebook code that never lived in the package to begin with (the Bernoulli-KL
machinery from notebooks/13-sk-paper-figures.ipynb).

- EMNLP revisions, 2026
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from winogender_contextuality.config import INTERIM_DATA_DIR, PROJ_ROOT
from winogender_contextuality.utils import (  # noqa: F401  (re-exported for callers)
    get_index,
    get_sent_order,
    get_single_sentences,
    get_filled_pnoun,
    load_ndjson,
)

# --------------------------------------------------------------------------------------
# Paths. All revision output lands under revisions/ so no existing result is ever touched.
# --------------------------------------------------------------------------------------

REVISIONS_DIR = PROJ_ROOT / "revisions"
OUTPUTS_DIR = REVISIONS_DIR / "outputs"
CONFIGS_DIR = REVISIONS_DIR / "configs"
REV_LOG_DIR = REVISIONS_DIR / "logs"

DEFAULT_SEED = 20260713

# The final row of the pairs TSV is the sentinel row injected by dataset.py
# (differences = ['she', 'potato']). The paper analyses stop before it; so do we.
DEFAULT_MAX_PAIR_INDEX = 180

FEMALE_PRONOUNS = {"she", "her", "hers"}
MALE_PRONOUNS = {"he", "him", "his"}

# Grammatical case -> how a referent noun phrase is realised in that slot.
CASE_TO_NP = {
    "$NOM_PRONOUN": "the {referent}",
    "$ACC_PRONOUN": "the {referent}",
    "$POSS_PRONOUN": "the {referent}'s",
}


def data_dir(override: str | Path | None = None) -> Path:
    """Interim-data directory: CLI override > $WC_INTERIM_DIR > package config."""
    if override is not None:
        return Path(override)
    env = os.environ.get("WC_INTERIM_DIR")
    if env:
        return Path(env)
    return INTERIM_DATA_DIR


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------


def set_seeds(seed: int = DEFAULT_SEED) -> int:
    """Seed every RNG that can influence a run. Torch is optional (analysis-only hosts)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        logger.debug("torch not installed; skipping torch seeding")
    return seed


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _lib_versions() -> dict:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ["numpy", "pandas", "scipy", "torch", "transformers", "statsmodels", "xarray"]:
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "not installed"
    return versions


def provenance(extra: dict | None = None, seed: int | None = None) -> dict:
    """
    Metadata block written alongside every results file: seed, git commit, library
    versions, the exact command line, and whatever the caller adds (model name,
    quantization setting, revision hash, ...).
    """
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "argv": sys.argv,
        "seed": seed,
        "libraries": _lib_versions(),
    }
    if extra:
        meta.update(extra)
    return meta


def write_results(
    df: pd.DataFrame,
    path: str | Path,
    meta: dict,
    index: bool = False,
) -> Path:
    """Write a results table plus a ``<name>.meta.json`` provenance sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sep = "\t" if path.suffix == ".tsv" else ","
    df.to_csv(path, sep=sep, index=index)

    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta = {**meta, "n_rows": int(len(df)), "columns": list(df.columns), "results_file": path.name}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.success(f"Wrote {len(df)} rows to {path} (provenance: {meta_path.name})")
    return path


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Flags every revision CLI shares."""
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (logged).")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory holding the measurement NDJSONs. Defaults to $WC_INTERIM_DIR, "
        "else INTERIM_DATA_DIR from the package config.",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=DEFAULT_MAX_PAIR_INDEX,
        help=f"Exclusive upper bound on pair index (default {DEFAULT_MAX_PAIR_INDEX}: "
        "drops the ['she','potato'] sentinel row).",
    )
    return parser


# --------------------------------------------------------------------------------------
# Pronoun bookkeeping
#
# `context.pronouns_2` is the *canonical* option list in the primed/unprimed runs, but the
# *presented* (possibly reversed) list in the null runs -- see generate_one_null_context in
# collect_sequential.py. The notebook indexes position [1] for "female", which happens to be
# right only because the n=0 records are written to disk first. We resolve the female option
# by string instead, so ordering can never silently flip a result.
# --------------------------------------------------------------------------------------


def female_option(pronouns: list[str]) -> str:
    """The female member of an option pair, resolved by string rather than position."""
    for p in pronouns:
        if str(p).lower() in FEMALE_PRONOUNS:
            return p
    logger.warning(f"No female pronoun in {pronouns}; falling back to position [1].")
    return pronouns[1]


def male_option(pronouns: list[str]) -> str:
    for p in pronouns:
        if str(p).lower() in MALE_PRONOUNS:
            return p
    logger.warning(f"No male pronoun in {pronouns}; falling back to position [0].")
    return pronouns[0]


@dataclass
class FemaleCounts:
    """Female-pronoun successes out of valid (in-option-set) generations."""

    successes: int
    total: int

    @property
    def smoothed_prob(self) -> float:
        """Add-half (Jeffreys) smoothing, matching get_model_divergences in the notebook."""
        return (self.successes + 0.5) / (self.total + 1)

    @property
    def raw_prob(self) -> float:
        return self.successes / self.total if self.total else np.nan


def female_generation_counts(measurements: list[dict]) -> FemaleCounts:
    """
    Count female-pronoun generations across a set of measurements, discarding outputs
    that are not one of the two offered options (same cleaning rule as
    utils.get_generation_details, but keyed by string).
    """
    if not measurements:
        return FemaleCounts(0, 0)

    options = measurements[0]["context"]["pronouns_2"]
    fem = str(female_option(options)).lower()
    valid = {str(o).lower() for o in options}

    generated = []
    for m in measurements:
        try:
            generated.append(str(m["measurement"]["BLANK"]).lower())
        except (KeyError, TypeError):
            continue

    counter = Counter(g for g in generated if g in valid)
    total = sum(counter.values())
    return FemaleCounts(successes=counter[fem], total=total)


def female_internal_prob(measurements: list[dict]) -> float:
    """
    P(female) from the model's internal logits, averaged over trials then softmaxed --
    the same construction as utils.get_internal_probs, but selecting the female slot by
    string. Logit vectors follow the canonical option order used at collection time.
    """
    if not measurements:
        return float("nan")

    logits = [np.asarray(m["logits"], dtype=float).ravel() for m in measurements if m.get("logits")]
    if not logits:
        return float("nan")

    from scipy.special import softmax

    probs = softmax(np.mean(logits, axis=0))

    # Collection always writes logits in canonical (male, female) order: the token ids come
    # from `pronouns[1]`, the unpermuted option list. Position 1 is therefore female.
    return float(probs[1])


# --------------------------------------------------------------------------------------
# Divergences (ported from notebooks/13-sk-paper-figures.ipynb, which is not importable)
# --------------------------------------------------------------------------------------


def bernoulli_kl(p_x: float, p_null: float) -> float:
    """KL(Bernoulli(p_x) || Bernoulli(p_null)) in bits."""
    p_x = float(p_x)
    p_null = float(p_null)
    return p_x * np.log2(p_x / p_null) + (1 - p_x) * np.log2((1 - p_x) / (1 - p_null))


def kl_from_counts(primed: FemaleCounts, baseline: FemaleCounts) -> float:
    """Smoothed Bernoulli KL of a primed condition against its unprimed baseline."""
    if primed.total == 0 or baseline.total == 0:
        return float("nan")
    return float(bernoulli_kl(primed.smoothed_prob, baseline.smoothed_prob))


# --------------------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------------------


@dataclass
class ModelSpec:
    name: str  # HuggingFace id
    shorthand: str  # short name used in filenames
    quantized: bool
    assistant: bool = True

    @property
    def stem(self) -> str:
        return self.name.split("/")[-1]


def load_model_config(path: str | Path) -> list[ModelSpec]:
    """Load a YAML model set (see revisions/configs/)."""
    import yaml

    with open(path) as f:
        doc = yaml.safe_load(f)

    return [
        ModelSpec(
            name=m["model_name"],
            shorthand=m.get("shorthand", m["model_name"].split("/")[-1]),
            quantized=bool(m.get("quantized", True)),
            assistant=bool(m.get("assistant", True)),
        )
        for m in doc["models"]
    ]


def resolve_models(config: str | Path | None, models: list[str] | None) -> list[ModelSpec]:
    """
    Model selection for the CLIs: an explicit --models list wins over a --config YAML.
    Bare --models entries inherit the config's quantization setting when both are given.
    """
    specs = load_model_config(config) if config else []
    if not models:
        return specs

    by_name = {s.name: s for s in specs}
    by_short = {s.shorthand: s for s in specs}

    chosen = []
    for m in models:
        if m in by_name:
            chosen.append(by_name[m])
        elif m in by_short:
            chosen.append(by_short[m])
        else:
            chosen.append(ModelSpec(name=m, shorthand=m.split("/")[-1], quantized=True))
    return chosen
