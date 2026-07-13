#!/usr/bin/env bash
#SBATCH --job-name=rev_refnull
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=07:59:00
#SBATCH --output=/home/kumar.sag/winogender_contextuality/logs/%x_%j.out
#SBATCH --error=/home/kumar.sag/winogender_contextuality/logs/%x_%j.err
#SBATCH --mail-user=kumar.sag@northeastern.edu
#SBATCH --mail-type=ALL

# Experiment 3: referent-only null primes -- primes that name the referents but contain no
# pronoun, isolating co-occurrence from pronoun priming.
#
# Build the primes first (cheap, no GPU):
#   python -m revisions.primes --data-dir "$DATA_DIR"

set -euo pipefail

VENV_DIR="./wcenv"
PYTHON_BIN="python3"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR..."
  $PYTHON_BIN -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

export HF_KEY=""

MODEL="${MODEL:-google/gemma-3-12b-it}"
TEMP="${TEMP:-0.5}"
N_RUNS="${N_RUNS:-50}"
SEED="${SEED:-20260713}"
BATCH_SIZE="${BATCH_SIZE:-10}"
N_ROWS="${N_ROWS:-360}"               # 180 pairs x 2 sentence slots
STYLE="${STYLE:-repeated_np}"
DATA_DIR="${DATA_DIR:-/scratch/kumar.sag/data/interim}"

PRIMES="revisions/outputs/primes/referent_primes_${STYLE}.csv"
OUTPUT_FILE="/scratch/kumar.sag/data/interim/refnull_measurements_${MODEL##*/}_${TEMP}_wp.ndjson"

# Regenerate the primes if they are not already there.
if [[ ! -f "$PRIMES" ]]; then
  echo "Building referent-only primes ($STYLE)..."
  python -u -m revisions.primes --style "$STYLE" --data-dir "$DATA_DIR"
fi

# gemma was run unquantized in the paper; pass --quantized for the models that were not.
QUANT_FLAG=""
if [[ "${QUANTIZED:-false}" == "true" ]]; then
  QUANT_FLAG="--quantized"
fi

for START in $(seq 0 "$BATCH_SIZE" $((N_ROWS - 1))); do
  END=$((START + BATCH_SIZE - 1))
  echo "Running batch: $START to $END"
  python -u -m revisions.collect_referent_null \
    --model "$MODEL" \
    --primes "$PRIMES" \
    --temperature "$TEMP" \
    --n-runs "$N_RUNS" \
    --seed "$SEED" \
    --start "$START" --end "$END" \
    --output-file "$OUTPUT_FILE" \
    $QUANT_FLAG
done

echo "Referent-only null measurements written to $OUTPUT_FILE"
