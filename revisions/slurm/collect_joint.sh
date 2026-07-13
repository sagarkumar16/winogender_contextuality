#!/usr/bin/env bash
#SBATCH --job-name=rev_joint
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

# Experiment 1: joint (two-pronoun) measurements.
# Both pronouns of a pair are generated in one pass, so both random variables are actually
# measured in each context -- the formulation the QQ equality uses.

set -euo pipefail

VENV_DIR="./wcenv"
PYTHON_BIN="python3"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR..."
  $PYTHON_BIN -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

export HF_KEY=""

MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
TEMP="${TEMP:-0.5}"
SEED="${SEED:-20260713}"
N_RUNS="${N_RUNS:-50}"
BATCH_SIZE="${BATCH_SIZE:-10}"
N_ROWS="${N_ROWS:-181}"
DATA_DIR="${DATA_DIR:-/scratch/kumar.sag/data/interim}"

OUTPUT_FILE="/scratch/kumar.sag/data/interim/joint_measurements_${MODEL##*/}_${TEMP}_wp.ndjson"

for START in $(seq 0 "$BATCH_SIZE" $((N_ROWS - 1))); do
  END=$((START + BATCH_SIZE - 1))
  echo "Running batch: $START to $END"
  python -u -m revisions.collect_joint \
    --model "$MODEL" \
    --temperature "$TEMP" \
    --n-runs "$N_RUNS" \
    --seed "$SEED" \
    --start "$START" --end "$END" \
    --data-dir "$DATA_DIR" \
    --input-file "winopron_pairs.tsv" \
    --output-file "$OUTPUT_FILE"
done

echo "Joint measurements written to $OUTPUT_FILE"
