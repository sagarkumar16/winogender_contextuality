#!/usr/bin/env bash
#SBATCH --job-name=rev_unquant
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --time=07:59:00
#SBATCH --output=/home/kumar.sag/winogender_contextuality/logs/%x_%j.out
#SBATCH --error=/home/kumar.sag/winogender_contextuality/logs/%x_%j.err
#SBATCH --mail-user=kumar.sag@northeastern.edu
#SBATCH --mail-type=ALL

# Experiment 2: re-run the published pipeline on UNQUANTIZED weights.
#
# Note the larger --mem than the quantized runs: unquantized weights need considerably more
# GPU/host memory (a 20B model in bf16 is ~40GB). Raise --mem / use a bigger GPU if a model OOMs.
#
# Same items, prompts, conditions, temperature and n_runs as the published runs -- this script
# calls the published collection functions directly, with quantized=False.
#
# MODEL=google/gemma-3-12b-it is the SANITY CHECK: it was already run unquantized, so the new
# numbers must reproduce the existing ones (see revisions/slurm/../README.md).

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
CONDITION="${CONDITION:-primed}"      # primed | null | both
TEMP="${TEMP:-0.5}"                   # matches the published runs
N_RUNS="${N_RUNS:-50}"                # matches the published runs
SEED="${SEED:-20260713}"
BATCH_SIZE="${BATCH_SIZE:-10}"
DATA_DIR="${DATA_DIR:-/scratch/kumar.sag/data/interim}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/kumar.sag/data/revisions/measurements}"

# 181 pairs for the primed condition; 362 sentence rows for the null condition.
if [[ "$CONDITION" == "null" ]]; then
  N_ROWS="${N_ROWS:-362}"
else
  N_ROWS="${N_ROWS:-181}"
fi

for START in $(seq 0 "$BATCH_SIZE" $((N_ROWS - 1))); do
  END=$((START + BATCH_SIZE - 1))
  echo "Running batch: $START to $END"
  python -u -m revisions.collect_unquantized \
    --models "$MODEL" \
    --condition "$CONDITION" \
    --temperature "$TEMP" \
    --n-runs "$N_RUNS" \
    --seed "$SEED" \
    --start "$START" --end "$END" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR"
done

echo "Unquantized measurements written under $OUTPUT_DIR"
