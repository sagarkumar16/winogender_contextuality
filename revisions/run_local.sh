#!/usr/bin/env bash
# Run revision collections one after another on this machine, through slurm/collect.sh (no SLURM).
# A failed run doesn't stop the rest; each run logs to $LOG_DIR/<run name>.log (default logs/).
# Every run continues its output file where it stops (START_ROW=auto in collect.sh) and skips it
# if it is finished, so after a failure or a reboot, just run this again.
#
#   bash revisions/run_local.sh
#   nohup setsid bash revisions/run_local.sh > logs/run_local.out 2>&1 &    # keeps running after logout
#
# Without HF_KEY the weights must already be in the cache (all four models are on this machine), and
# the Hub is not contacted (HF_HUB_OFFLINE=1): a gated model like Llama is refused without a token.
#
# The runs below are every revision experiment on Llama-3.1-8B-Instruct, about 17 h on an RTX A5000:
#   refnull, joint   4-bit (bitsandbytes nf4), as Llama ran in the paper (2 h + 2.5 h)
#   primed, null     unquantized (bf16): the quantization arm, compared with the paper's 4-bit runs (6.5 h + 5 h)
#
# DATA_DIR defaults to the package's INTERIM_DATA_DIR, where the notebook reads, rather than
# collect.sh's cluster default. FROM=<run name> skips the runs before it (names as in the log
# files, e.g. Llama-3.1-8B-Instruct_primed_bf16); START_ROW=<row> restarts that first run at a
# given row (see collect.sh).

set -euo pipefail
[[ -n "${HF_KEY:-}" ]] || export HF_HUB_OFFLINE=1
cd "$(dirname "$0")/.."

# collect.sh activates ./wcenv; if it is missing it creates an EMPTY venv there and every run fails.
[[ -d wcenv ]] || { echo "No ./wcenv. Point it at the project venv first: ln -s .venv wcenv" >&2; exit 1; }
DATA_DIR="${DATA_DIR:-$(wcenv/bin/python -c 'from winogender_contextuality.config import INTERIM_DATA_DIR as d; print(d)' 2>/dev/null)}"
[[ -n "$DATA_DIR" ]] || { echo "Could not read INTERIM_DATA_DIR; set DATA_DIR" >&2; exit 1; }
export DATA_DIR
start_row="${START_ROW:-auto}"
unset START_ROW
log_dir="${LOG_DIR:-logs}"
mkdir -p "$log_dir"

RUNS=(
  # model                             condition quantized suffix
  "meta-llama/Llama-3.1-8B-Instruct   refnull   true"
  "meta-llama/Llama-3.1-8B-Instruct   joint     true"
  "meta-llama/Llama-3.1-8B-Instruct   primed    false   _bf16"
  "meta-llama/Llama-3.1-8B-Instruct   null      false   _bf16"
)

names=()
for spec in "${RUNS[@]}"; do
  read -r model condition quantized suffix <<< "$spec"
  names+=("${model##*/}_${condition}${suffix}")
done
first=0
if [[ -n "${FROM:-}" ]]; then
  for i in "${!names[@]}"; do [[ "${names[$i]}" == "$FROM" ]] && { first=$i; break; }; done
  [[ "${names[$first]}" == "$FROM" ]] || { echo "FROM=$FROM matches no run. Runs: ${names[*]}" >&2; exit 1; }
fi

echo "=== $(date '+%F %T')  ${#RUNS[@]} runs from ${names[$first]}, DATA_DIR=$DATA_DIR"
failed=()
for ((i = first; i < ${#RUNS[@]}; i++)); do
  read -r model condition quantized suffix <<< "${RUNS[$i]}"
  name=${names[$i]}
  log="$log_dir/$name.log"
  echo "=== $(date '+%F %T')  $name  quantized=$quantized  START_ROW=$start_row  -> $log"
  if MODEL="$model" CONDITION="$condition" QUANTIZED="$quantized" SUFFIX="$suffix" START_ROW="$start_row" \
      bash revisions/slurm/collect.sh 2>&1 | tee -a "$log"; then
    echo "=== $(date '+%F %T')  finished $name"
  else
    echo "=== $(date '+%F %T')  FAILED $name (see $log)"
    failed+=("$name")
  fi
  start_row=auto   # an explicit START_ROW applies to the first run only
done

if (( ${#failed[@]} )); then
  echo "=== $(date '+%F %T')  done; failed: ${failed[*]}"
  exit 1
fi
echo "=== $(date '+%F %T')  all runs finished"
