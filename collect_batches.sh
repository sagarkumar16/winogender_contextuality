#!/usr/bin/env bash
#SBATCH --job-name=collect_oss
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

set -euo pipefail

# --- Optional: load modules ---
# module purge
# module load cuda/12.1

# --- Python venv ---
VENV_DIR="./wcenv"
PYTHON_BIN="python3"  # or an absolute path if needed

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR..."
  $PYTHON_BIN -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Optional: install deps (comment out if handled elsewhere)
# pip install -U pip
# pip install -r requirements.txt

# --- Environment variables ---
export HF_KEY=""  

# --- Script arguments ---
DEVICE="gpu"
MODEL="openai/gpt-oss-20b"
TEMP=0.5
STREAM=True

# --- Run in batches of 10 ---
TOTAL_RUNS=50
BATCH_SIZE=10
OUTPUT_FILE="/scratch/kumar.sag/data/interim/one_pronoun_measurements_${MODEL##*/}_${TEMP}_wp.ndjson"
INPUT_FILE="winopron_pairs.tsv"
N_ROWS=181

for START in $(seq 0 $BATCH_SIZE $((N_ROWS-1))); do
  END=$((START + BATCH_SIZE - 1))
  echo "Running batch: $START to $END"
  python -u winogender_contextuality/modeling/collect_sequential.py generate-one-pronoun \
    "$DEVICE" "$MODEL" "$TEMP" --n-runs="$TOTAL_RUNS" \
    --start "$START" --end "$END" \
    --input-file="$INPUT_FILE" --output-file "$OUTPUT_FILE"
done