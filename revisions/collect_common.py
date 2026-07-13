"""
Shared machinery for the revision collection scripts (the parts that need a GPU).

Everything here wraps the published inference code rather than reimplementing it:
ModelProbs for the model, `no_game_seq_logit_prompt` / `no_game_seq_prompt` for the prompts,
and the `Context` / `Measurement` dataclasses for the on-disk schema. The output NDJSON is
therefore readable by the existing analysis helpers (get_index, get_sent_order, ...).

Two things differ deliberately from the originals, both documented in revisions/README.md:

1. Logit token ids are taken from the full option *list*. `generate_one_null_context` passes
   `pronouns[1]` where `pronouns` is a list, so it indexes the string "she" and stores logits
   for the characters 's','h','e'. Our null-style runs store a genuine 2-vector over
   (male, female), in canonical option order, so internal probabilities are usable.

2. Every run writes a provenance sidecar (seed, model revision, quantization, library
   versions, git commit).

- EMNLP revisions, 2026
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from loguru import logger

from winogender_contextuality.config import MODELS_DIR
from winogender_contextuality.utils import Context, Measurement

from revisions.common import ModelSpec, provenance

# ModelProbs is imported lazily inside load_model(). Importing it at module scope pulls in
# winogender_contextuality.modeling.run_local, which needs `Mxfp4Config` (transformers >= 4.55,
# for gpt-oss) and a CUDA build of torch. Keeping it lazy lets the prompt/schema dry-runs and
# the analysis tests run on a laptop; the GPU path is unaffected.

HF_KEY = os.environ.get("HF_KEY")

try:  # parallel batches append to one file, exactly as the published scripts do
    import fcntl

    def _lock(fh):
        fcntl.flock(fh, fcntl.LOCK_EX)

    def _unlock(fh):
        fcntl.flock(fh, fcntl.LOCK_UN)
except Exception:  # pragma: no cover - non-POSIX

    def _lock(fh):
        pass

    def _unlock(fh):
        pass


def append_measurement(path: Path, m: Measurement) -> None:
    """Append one Measurement to the shared NDJSON, with an exclusive lock."""
    with open(path, "a") as f:
        _lock(f)
        f.write(json.dumps(asdict(m)) + "\n")
        f.flush()
        os.fsync(f.fileno())
        _unlock(f)


def load_model(spec: ModelSpec, mode: str = "gpu"):
    """Instantiate and load a model through the published ModelProbs wrapper."""
    from winogender_contextuality.modeling.ModelProbs import ModelProbs

    mp = ModelProbs(
        mode=mode,
        model_name=spec.name,
        key=HF_KEY,
        model_path=MODELS_DIR,
        quantized=spec.quantized,
    )
    mp.load_model()
    logger.info(f"Loaded {spec.name} (quantized={spec.quantized})")
    return mp


def model_provenance(spec: ModelSpec, seed: int, extra: dict | None = None) -> dict:
    """Provenance block including the resolved HF revision hash where obtainable."""
    revision = "unknown"
    try:
        from huggingface_hub import HfApi

        revision = HfApi().model_info(spec.name, token=HF_KEY).sha
    except Exception as e:  # offline cluster nodes, gated repos, ...
        logger.debug(f"Could not resolve HF revision for {spec.name}: {e}")

    meta = {
        "model_name": spec.name,
        "model_shorthand": spec.shorthand,
        "model_revision": revision,
        "quantized": spec.quantized,
        "assistant_prompt": spec.assistant,
    }
    if extra:
        meta.update(extra)
    return provenance(meta, seed=seed)


def write_run_manifest(output_fpath: Path, meta: dict) -> Path:
    """Provenance sidecar next to a measurements NDJSON."""
    meta_path = output_fpath.with_suffix(output_fpath.suffix + ".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Wrote provenance to {meta_path}")
    return meta_path


def pronoun_logits(mp, prompt: list[dict], options: list[str]) -> list[float]:
    """
    Next-token logits restricted to the (male, female) options, in canonical option order.

    `options` must be the canonical, unpermuted list (e.g. ['he','she']) so that position 1
    is always the female pronoun, matching how the primed runs store logits.
    """
    model_logits = mp.get_raw_logits(prompt=prompt).cpu()
    token_ids = mp.get_token_ids(options=options)  # list of id-lists, one per option
    return model_logits[[sum(token_ids, [])]].tolist()


def decode_completion(mp, model_name: str, inputs, output) -> str:
    """Decode a generation, reproducing the published gemma special-case."""
    input_len = inputs.shape[1]

    if "gemma" in model_name:
        full = mp.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        try:
            return full.split("model")[-1]
        except Exception as e:
            logger.warning(f"gemma decode failed ({e}) on: {full!r}")
            return "{'BLANK': 'None'}"

    return mp.tokenizer.decode(output.sequences[0][input_len - 5:], skip_special_tokens=True)


def parse_blank(decoded: str, keys: tuple[str, ...] = ("BLANK",)) -> dict:
    """
    Parse a model completion into its BLANK dict, falling back to the published
    {'BLANK': 'None'} sentinel so that downstream filters (get_index) drop it.
    """
    import ast

    try:
        parsed = ast.literal_eval(decoded)
        if isinstance(parsed, dict) and all(k in parsed for k in keys):
            return parsed
        raise ValueError(f"missing keys {keys}")
    except Exception as e:
        logger.warning(f"Could not parse completion {decoded!r}: {e}")
        return {k: "None" for k in keys}


def default_output_path(
    output_dir: Path, prefix: str, spec: ModelSpec, temperature: float, tag: str = ""
) -> Path:
    stamp = datetime.now().strftime("%H%M%d%m%y")
    suffix = f"_{tag}" if tag else ""
    return Path(output_dir) / f"{prefix}_{spec.stem}_{temperature}{suffix}_{stamp}.ndjson"


__all__ = [
    "append_measurement",
    "load_model",
    "model_provenance",
    "write_run_manifest",
    "pronoun_logits",
    "decode_completion",
    "parse_blank",
    "default_output_path",
    "Context",
    "Measurement",
    "HF_KEY",
]
