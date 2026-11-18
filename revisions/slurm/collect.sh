#!/usr/bin/env bash
#SBATCH --job-name=collect_revision
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=07:59:00
#SBATCH --output=/home/kumar.sag/winogender_contextuality/logs/%x_%j.out
#SBATCH --error=/home/kumar.sag/winogender_contextuality/logs/%x_%j.err
#SBATCH --mail-user=kumar.sag@northeastern.edu
#SBATCH --mail-type=ALL

# Collection for the EMNLP revision experiments (see revisions/README.md). Same loop as the
# published collect_*.sh scripts, configured from the environment:
#
#   MODEL       HuggingFace id (required)
#   CONDITION   primed | null | refnull | joint
#   QUANTIZED   true (default) | false
#   SUFFIX      appended to the output file name, e.g. _bf16 (default: none)
#   START_ROW   first row to collect, to resume a job that timed out (default 0). Delete the
#               records of any partially finished batch first, or they will be duplicated.
#   DATA_DIR    directory holding the input files; outputs are written there too
#
# e.g.  MODEL=openai/gpt-oss-20b CONDITION=primed QUANTIZED=false SUFFIX=_bf16 sbatch revisions/slurm/collect.sh

set -euo pipefail

# --- Python venv ---
VENV_DIR="./wcenv"
PYTHON_BIN="python3"  # or an absolute path if needed

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR..."
  $PYTHON_BIN -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# --- Environment variables ---
export HF_KEY="${HF_KEY:-}"

# --- Script arguments ---
DEVICE="gpu"
MODEL="${MODEL:?set MODEL to a HuggingFace model id}"
CONDITION="${CONDITION:?set CONDITION to primed, null, refnull or joint}"
QUANTIZED="${QUANTIZED:-true}"
SUFFIX="${SUFFIX:-}"
START_ROW="${START_ROW:-0}"
DATA_DIR="${DATA_DIR:-/scratch/kumar.sag/data/interim}"
TEMP=0.5
TOTAL_RUNS=50
BATCH_SIZE=10

EXTRA_ARGS=()
case "$CONDITION" in
  primed)  COMMAND=generate-one-pronoun;      INPUT_FILE=winopron_pairs.tsv;           PREFIX=one_pronoun_measurements ;;
  null)    COMMAND=generate-one-null-context; INPUT_FILE=all_sentences_wp.csv;         PREFIX=null_measurements ;;
  refnull) COMMAND=generate-one-null-context; INPUT_FILE=all_sentences_wp_refnull.csv; PREFIX=refnull_measurements
           EXTRA_ARGS=(--prime-column prime --tag refnull) ;;
  joint)   COMMAND=generate-two-pronouns;     INPUT_FILE=winopron_pairs.tsv;           PREFIX=joint_measurements ;;
  *) echo "CONDITION must be primed, null, refnull or joint; got '$CONDITION'" >&2; exit 1 ;;
esac

case "$QUANTIZED" in
  true)  QUANT_FLAG=--quantized ;;
  false) QUANT_FLAG=--no-quantized ;;
  *) echo "QUANTIZED must be true or false; got '$QUANTIZED'" >&2; exit 1 ;;
esac

if [[ "$CONDITION" == "refnull" && ! -f "$DATA_DIR/$INPUT_FILE" ]]; then
  python -m revisions.build_refnull_inputs --pairs "$DATA_DIR/winopron_pairs.tsv" --output "$DATA_DIR/$INPUT_FILE"
fi

# Row count from the input file itself (header excluded), not hardcoded.
N_ROWS=$(python -c "import sys, pandas as pd; f = sys.argv[1]; print(len(pd.read_csv(f, sep='\t' if f.endswith('.tsv') else ',')))" "$DATA_DIR/$INPUT_FILE")
OUTPUT_FILE="$DATA_DIR/${PREFIX}_${MODEL##*/}_${TEMP}_wp${SUFFIX}.ndjson"

echo "$CONDITION | $MODEL | quantized=$QUANTIZED | rows $START_ROW-$((N_ROWS-1)) of $INPUT_FILE -> $OUTPUT_FILE"

# --- Run in batches of 10 ---
for START in $(seq "$START_ROW" "$BATCH_SIZE" $((N_ROWS-1))); do
  END=$((START + BATCH_SIZE - 1))
  echo "Running batch: $START to $END"
  python -u winogender_contextuality/modeling/collect_sequential.py "$COMMAND" \
    "$DEVICE" "$MODEL" "$TEMP" --n-runs="$TOTAL_RUNS" \
    --start "$START" --end "$END" \
    --input-dir "$DATA_DIR" --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" "$QUANT_FLAG" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
done
