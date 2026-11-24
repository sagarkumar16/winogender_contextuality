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
#   START_ROW   0 (default): a new collection; refuses to append to an existing output file.
#               auto: continue the output file from its first incomplete row (does nothing if it
#               is finished). A number: restart at that row. Either way the records of the row it
#               starts at and of later rows are removed first, so nothing is duplicated.
#   MAX_TRIES   attempts per batch (default 3). A failed batch (e.g. a transient CUDA error) is
#               retried from the row it stopped in, RETRY_WAIT seconds later (default 60).
#   DATA_DIR    directory holding the input files; outputs are written there too
#
# e.g.  MODEL=openai/gpt-oss-20b CONDITION=primed QUANTIZED=false SUFFIX=_bf16 sbatch revisions/slurm/collect.sh
#       ... START_ROW=auto sbatch revisions/slurm/collect.sh    # after a job timed out or failed

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
MAX_TRIES="${MAX_TRIES:-3}"
RETRY_WAIT="${RETRY_WAIT:-60}"
DATA_DIR="${DATA_DIR:-/scratch/kumar.sag/data/interim}"
TEMP=0.5
TOTAL_RUNS=50
BATCH_SIZE=10

# PER_ROW: records the collector writes for one input row, i.e. a complete row
EXTRA_ARGS=()
case "$CONDITION" in
  primed)  COMMAND=generate-one-pronoun;      INPUT_FILE=winopron_pairs.tsv;           PREFIX=one_pronoun_measurements
           PER_ROW=$((16 * TOTAL_RUNS)) ;;   # 2 sentence orders x 2 option orders x (unprimed, primed) x 2 pronouns
  null)    COMMAND=generate-one-null-context; INPUT_FILE=all_sentences_wp.csv;         PREFIX=null_measurements
           PER_ROW=$((6 * TOTAL_RUNS)) ;;    # 3 primes (config.null_sentences) x 2 option orders
  refnull) COMMAND=generate-one-null-context; INPUT_FILE=all_sentences_wp_refnull.csv; PREFIX=refnull_measurements
           PER_ROW=$((2 * TOTAL_RUNS))       # 1 prime x 2 option orders
           EXTRA_ARGS=(--prime-column prime --tag refnull) ;;
  joint)   COMMAND=generate-two-pronouns;     INPUT_FILE=winopron_pairs.tsv;           PREFIX=joint_measurements
           PER_ROW=$((4 * TOTAL_RUNS)) ;;    # 2 passage orders x 2 option orders
  *) echo "CONDITION must be primed, null, refnull or joint; got '$CONDITION'" >&2; exit 1 ;;
esac

case "$QUANTIZED" in
  true)  QUANT_FLAG=--quantized ;;
  false) QUANT_FLAG=--no-quantized ;;
  *) echo "QUANTIZED must be true or false; got '$QUANTIZED'" >&2; exit 1 ;;
esac

[[ "$START_ROW" == auto || "$START_ROW" =~ ^[0-9]+$ ]] || { echo "START_ROW must be auto or a row number; got '$START_ROW'" >&2; exit 1; }

if [[ "$CONDITION" == "refnull" && ! -f "$DATA_DIR/$INPUT_FILE" ]]; then
  python -m revisions.build_refnull_inputs --pairs "$DATA_DIR/winopron_pairs.tsv" --output "$DATA_DIR/$INPUT_FILE"
fi

# Row count from the input file itself (header excluded), not hardcoded.
N_ROWS=$(python -c "import sys, pandas as pd; f = sys.argv[1]; print(len(pd.read_csv(f, sep='\t' if f.endswith('.tsv') else ',')))" "$DATA_DIR/$INPUT_FILE")
OUTPUT_FILE="$DATA_DIR/${PREFIX}_${MODEL##*/}_${TEMP}_wp${SUFFIX}.ndjson"

# Records are appended, so a new collection must not start on an existing file.
if [[ "$START_ROW" == 0 && -s "$OUTPUT_FILE" ]]; then
  echo "$OUTPUT_FILE already exists. START_ROW=auto continues it; delete it to start over." >&2
  exit 1
elif [[ "$START_ROW" == auto ]]; then
  START_ROW=$(python -m revisions.resume_point "$OUTPUT_FILE" --per-row "$PER_ROW")
elif [[ -s "$OUTPUT_FILE" ]]; then
  START_ROW=$(python -m revisions.resume_point "$OUTPUT_FILE" --per-row "$PER_ROW" --from-row "$START_ROW")
fi
if (( START_ROW >= N_ROWS )); then
  echo "$CONDITION | $MODEL: $OUTPUT_FILE has all $N_ROWS rows; nothing to do."
  exit 0
fi

echo "$CONDITION | $MODEL | quantized=$QUANTIZED | rows $START_ROW-$((N_ROWS-1)) of $INPUT_FILE -> $OUTPUT_FILE"

# --- Run in batches of 10, each retried from the row it stopped in ---
for START in $(seq "$START_ROW" "$BATCH_SIZE" $((N_ROWS-1))); do
  END=$((START + BATCH_SIZE - 1))
  FROM=$START
  for TRY in $(seq "$MAX_TRIES"); do
    ATTEMPT=""
    if (( TRY > 1 )); then ATTEMPT=" (attempt $TRY of $MAX_TRIES)"; fi
    echo "Running batch: $FROM to $END$ATTEMPT"
    if python -u winogender_contextuality/modeling/collect_sequential.py "$COMMAND" \
        "$DEVICE" "$MODEL" "$TEMP" --n-runs="$TOTAL_RUNS" \
        --start "$FROM" --end "$END" \
        --input-dir "$DATA_DIR" --input-file "$INPUT_FILE" \
        --output-file "$OUTPUT_FILE" "$QUANT_FLAG" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; then
      continue 2
    fi
    FROM=$(python -m revisions.resume_point "$OUTPUT_FILE" --per-row "$PER_ROW" --search-from "$START")
    echo "Batch failed (attempt $TRY of $MAX_TRIES); rows before $FROM are complete." >&2
    if (( FROM > END )); then
      continue 2   # it failed after writing all its rows
    fi
    if (( TRY < MAX_TRIES )); then
      sleep "$RETRY_WAIT"
    fi
  done
  echo "Giving up on rows $FROM-$END after $MAX_TRIES attempts. START_ROW=auto continues from row $FROM." >&2
  exit 1
done
