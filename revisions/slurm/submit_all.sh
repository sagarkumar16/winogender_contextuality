#!/usr/bin/env bash
# Submit every revision collection as its own collect.sh job, chained so they run one after
# another. Each job starts when the previous one ends, whether it succeeded, failed or timed out
# (afterany), so one bad run doesn't stop the rest, and each job gets collect.sh's full time limit.
# Sequential order also matters for refnull: the first refnull job builds
# all_sentences_wp_refnull.csv and the later ones reuse it.
#
#   HF_KEY=... bash revisions/slurm/submit_all.sh            # DATA_DIR is passed through if set
#   DRY_RUN=1 HF_KEY=x bash revisions/slurm/submit_all.sh    # print the sbatch commands only
#
# Resuming, e.g. after hitting the cluster's limit on queued jobs (the script prints this command
# when sbatch refuses a run): FROM is the job name of the first run to submit, AFTER the id of the
# job it should wait for. If AFTER has already left the queue, the run starts without waiting.
#   FROM=gpt-oss-20b_primed_bf16 AFTER=11988646 HF_KEY=... bash revisions/slurm/submit_all.sh
#
# Per-run logs: <collect.sh --output dir>/<model>_<condition><suffix>_<jobid>.out/.err
# Cancel the rest of the chain with: scancel --name=<job name>, or scancel <id> for each.

set -euo pipefail
: "${HF_KEY:?set HF_KEY first}"

# sbatch runs the job in the directory it was submitted from, and collect.sh uses ./wcenv and
# relative script paths, so submit from the repo root wherever this was called from.
cd "$(dirname "$0")/../.."

# SLURM does not create the --output directory and a job whose log dir is missing dies silently.
LOG_DIR=$(sed -n 's|^#SBATCH --output=\(.*\)/[^/]*$|\1|p' revisions/slurm/collect.sh)
[[ -n "${DRY_RUN:-}" ]] || mkdir -p "$LOG_DIR"

RUNS=(
  # model                             condition quantized suffix
  "openai/gpt-oss-20b                 refnull   true"
  "microsoft/phi-4                    refnull   true"
  "google/gemma-3-12b-it              refnull   false"
  "meta-llama/Llama-3.1-8B-Instruct   refnull   true"
  "openai/gpt-oss-20b                 joint     true"
  "microsoft/phi-4                    joint     true"
  "google/gemma-3-12b-it              joint     false"
  "meta-llama/Llama-3.1-8B-Instruct   joint     true"
  "openai/gpt-oss-20b                 primed    false   _bf16"
  "openai/gpt-oss-20b                 null      false   _bf16"
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

# sbatch rejects a dependency on a job SLURM has already forgotten, so only wait on AFTER while it
# is still queued or running.
prev="${AFTER:-}"
if [[ -n "$prev" && -z "${DRY_RUN:-}" ]] && ! squeue -h -j "$prev" -o %i 2>/dev/null | grep -q .; then
  echo "job $prev is no longer queued or running; the first run starts without waiting for it"
  prev=""
fi

n=0
for ((i = first; i < ${#RUNS[@]}; i++)); do
  read -r model condition quantized suffix <<< "${RUNS[$i]}"
  name=${names[$i]}
  cmd=(sbatch --parsable --job-name="$name"
       ${prev:+--dependency=afterany:$prev}
       --export="ALL,MODEL=$model,CONDITION=$condition,QUANTIZED=$quantized,SUFFIX=$suffix"
       revisions/slurm/collect.sh)

  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "${cmd[*]}"
    prev="<job$((n += 1))>"
  else
    if ! out=$("${cmd[@]}"); then
      echo "sbatch refused $name (its message is above). Submit it and the runs after it with:" >&2
      echo "  FROM=$name${prev:+ AFTER=$prev} HF_KEY=... bash revisions/slurm/submit_all.sh" >&2
      exit 1
    fi
    prev=${out%%;*}   # --parsable prints "jobid" or "jobid;cluster"
    echo "submitted $prev  $name"
  fi
done
