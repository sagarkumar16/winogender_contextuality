#!/usr/bin/env bash
#
# Run all four revision analyses once the collection jobs have finished.
#
# CPU only -- no GPU, no SLURM. Run it on the login node or a laptop with the NDJSONs mounted:
#
#   bash revisions/slurm/run_analyses.sh
#   MODELS="gemma llama1b" bash revisions/slurm/run_analyses.sh
#
# Every step is skipped (with a message) if its input files are not there yet, so it is safe to
# re-run as jobs land. Results go to revisions/outputs/ and never overwrite existing results.

set -uo pipefail   # NOT -e: a missing input for one experiment should not kill the rest

DATA_DIR="${DATA_DIR:-/scratch/kumar.sag/data/interim}"
REV_DIR="${REV_DIR:-/scratch/kumar.sag/data/revisions/measurements}"
OUT="${OUT:-revisions/outputs}"
TEMP="${TEMP:-0.5}"
MAX_INDEX="${MAX_INDEX:-180}"

# shorthand -> the model's file stem, as it appears in the NDJSON filenames.
# A case statement rather than an associative array: `declare -A` needs bash 4, and macOS still
# ships bash 3.2, so this keeps the script runnable off the cluster too.
stem_for() {
  case "$1" in
    gemma)   echo "gemma-3-12b-it" ;;
    gpt)     echo "gpt-oss-20b" ;;
    llama1b) echo "Llama-3.2-1B-Instruct" ;;
    llama8b) echo "Llama-3.1-8B-Instruct" ;;
    phi)     echo "phi-4" ;;
    qwen)    echo "Qwen2.5-7B-Instruct" ;;
    *)       echo "" ;;
  esac
}

MODELS="${MODELS:-gemma gpt llama1b llama8b phi qwen}"

# The published primed runs. Filenames are not uniform across models (some carry a _k40 suffix,
# llama1b carries a timestamp), so glob for them rather than assuming a pattern.
primed_file() {
  local stem="$1"
  ls -1 "$DATA_DIR"/one_pronoun_measurements_"$stem"_*.ndjson 2>/dev/null | grep -v unquant | head -1
}
null_file()   { ls -1 "$DATA_DIR"/null_measurements_"$1"_*.ndjson 2>/dev/null | head -1; }
joint_file()  { ls -1 "$DATA_DIR"/joint_measurements_"$1"_*.ndjson 2>/dev/null | head -1; }
refnull_file(){ ls -1 "$DATA_DIR"/refnull_measurements_"$1"_*.ndjson 2>/dev/null | head -1; }
unquant_file(){ ls -1 "$REV_DIR"/one_pronoun_measurements_"$1"_*_unquant.ndjson 2>/dev/null | head -1; }

section() { echo; echo "############ $* ############"; }

# ---------------------------------------------------------------------------------------
section "Experiment 1: joint vs steering"
# ---------------------------------------------------------------------------------------
STEERING=(); JOINT=(); LABELS=()
for m in $MODELS; do
  s=$(primed_file "$(stem_for "$m")"); j=$(joint_file "$(stem_for "$m")")
  if [[ -n "$s" && -n "$j" ]]; then
    STEERING+=("$s"); JOINT+=("$j"); LABELS+=("$m")
  else
    echo "  skip $m (steering='${s:-missing}', joint='${j:-missing}')"
  fi
done
if ((${#LABELS[@]})); then
  python -u -m revisions.joint_measurement \
    --steering "${STEERING[@]}" --joint "${JOINT[@]}" --models "${LABELS[@]}" \
    --max-index "$MAX_INDEX" --bootstrap 1000 --out "$OUT/joint"
else
  echo "  nothing to do -- run revisions/slurm/collect_joint.sh first"
fi

# ---------------------------------------------------------------------------------------
section "Experiment 2: quantized vs unquantized"
# ---------------------------------------------------------------------------------------
QUANT=(); UNQUANT=(); LABELS=()
for m in $MODELS; do
  [[ "$m" == "gemma" ]] && continue          # gemma was already unquantized: it is the sanity check
  q=$(primed_file "$(stem_for "$m")"); u=$(unquant_file "$(stem_for "$m")")
  if [[ -n "$q" && -n "$u" ]]; then
    QUANT+=("$q"); UNQUANT+=("$u"); LABELS+=("$m")
  else
    echo "  skip $m (quantized='${q:-missing}', unquantized='${u:-missing}')"
  fi
done
if ((${#LABELS[@]})); then
  python -u -m revisions.quantization_compare \
    --quantized "${QUANT[@]}" --unquantized "${UNQUANT[@]}" --models "${LABELS[@]}" \
    --max-index "$MAX_INDEX" --out "$OUT/quantization"
else
  echo "  nothing to do -- run revisions/slurm/collect_unquantized.sh first"
fi

section "Experiment 2: gemma reproduction sanity check"
NEW=$(unquant_file "$(stem_for gemma)"); OLD=$(primed_file "$(stem_for gemma)")
if [[ -n "$NEW" && -n "$OLD" ]]; then
  # Uses the PRIMED runs deliberately: the published null runs have unusable logits, so the
  # deterministic half of the check cannot run on them.
  python -u -m revisions.quantization_compare --sanity-check \
    --new "$NEW" --existing "$OLD" --max-index "$MAX_INDEX" --out "$OUT/quantization"
else
  echo "  skip (new='${NEW:-missing}', existing='${OLD:-missing}')"
fi

# ---------------------------------------------------------------------------------------
section "Experiment 3: referent-only null primes"
# ---------------------------------------------------------------------------------------
PRIMES="revisions/outputs/primes/referent_primes_repeated_np.csv"
[[ -f "$PRIMES" ]] || python -u -m revisions.primes --data-dir "$DATA_DIR"

PRIMED=(); NULL=(); REFNULL=(); LABELS=()
for m in $MODELS; do
  p=$(primed_file "$(stem_for "$m")"); n=$(null_file "$(stem_for "$m")"); r=$(refnull_file "$(stem_for "$m")")
  if [[ -n "$p" && -n "$n" && -n "$r" ]]; then
    PRIMED+=("$p"); NULL+=("$n"); REFNULL+=("$r"); LABELS+=("$m")
  else
    echo "  skip $m (primed='${p:-missing}', null='${n:-missing}', refnull='${r:-missing}')"
  fi
done
if ((${#LABELS[@]})); then
  python -u -m revisions.referent_null \
    --primed "${PRIMED[@]}" --null "${NULL[@]}" --refnull "${REFNULL[@]}" \
    --models "${LABELS[@]}" --primes "$PRIMES" \
    --max-index "$MAX_INDEX" --out "$OUT/referent_null"
else
  echo "  nothing to do -- run revisions/slurm/collect_referent_null.sh first"
fi

# ---------------------------------------------------------------------------------------
section "Experiment 4: per-item plain probability table"
# ---------------------------------------------------------------------------------------
# Needs no new collection: it reads the published primed runs.
MEAS=(); LABELS=()
for m in $MODELS; do
  p=$(primed_file "$(stem_for "$m")")
  if [[ -n "$p" ]]; then MEAS+=("$p"); LABELS+=("$m"); else echo "  skip $m (no primed run)"; fi
done
if ((${#LABELS[@]})); then
  python -u -m revisions.per_item_table \
    --measurements "${MEAS[@]}" --models "${LABELS[@]}" \
    --max-index "$MAX_INDEX" --out "$OUT/per_item"
else
  echo "  nothing to do"
fi

echo
echo "Done. Results under $OUT/ -- each with a .meta.json provenance sidecar."
