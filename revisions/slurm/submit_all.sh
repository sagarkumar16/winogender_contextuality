#!/usr/bin/env bash
#
# Submit every GPU collection job for the revision experiments.
#
# This is NOT a SLURM job itself -- run it on the login node. It sbatch's one job per
# (experiment x model) and prints the job ids.
#
#   bash revisions/slurm/submit_all.sh              # submit everything
#   bash revisions/slurm/submit_all.sh --dry-run    # print the sbatch commands, submit nothing
#   EXPERIMENTS="joint" bash revisions/slurm/submit_all.sh          # just experiment 1
#   MODELS="google/gemma-3-12b-it" bash revisions/slurm/submit_all.sh
#
# Start with the fast set (llama1b) to shake out the pipeline before spending H200-hours on
# the 20B models.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Which experiments to submit: any of "joint unquantized refnull".
EXPERIMENTS="${EXPERIMENTS:-joint unquantized refnull}"

# The paper's six models. `name:quantized_in_the_paper` -- gemma was already run unquantized.
DEFAULT_MODELS=(
  "meta-llama/Llama-3.2-1B-Instruct:true"
  "Qwen/Qwen2.5-7B-Instruct:true"
  "meta-llama/Llama-3.1-8B-Instruct:true"
  "microsoft/phi-4:true"
  "google/gemma-3-12b-it:false"
  "openai/gpt-oss-20b:true"
)

# MODELS="a b" overrides; quantization then defaults to true except for gemma.
if [[ -n "${MODELS:-}" ]]; then
  SELECTED=()
  for m in $MODELS; do
    if [[ "$m" == *"gemma"* ]]; then SELECTED+=("$m:false"); else SELECTED+=("$m:true"); fi
  done
else
  SELECTED=("${DEFAULT_MODELS[@]}")
fi

# submit <script> <sbatch-opts> <ENV=VAL>...
# sbatch CLI options override the #SBATCH directives in the script; environment variables do
# not, which is why --mem is passed as a flag rather than exported.
submit() {
  local script="$1"; shift
  local sbatch_opts="$1"; shift

  if $DRY_RUN; then
    echo "  [dry-run] env $* sbatch $sbatch_opts $(basename "$script")"
  else
    local jid
    # shellcheck disable=SC2086  # sbatch_opts is intentionally word-split
    jid=$(env "$@" sbatch --parsable $sbatch_opts "$script")
    echo "  submitted job $jid: $* $(basename "$script")"
  fi
}

# Bigger models need more memory unquantized (bf16 weights, no 4-bit compression).
mem_for() {
  case "$1" in
    *gpt-oss-20b*)         echo "120GB" ;;
    *phi-4*|*gemma-3-12b*) echo "80GB"  ;;
    *)                     echo "40GB"  ;;
  esac
}

echo "Experiments: $EXPERIMENTS"
echo

for entry in "${SELECTED[@]}"; do
  model="${entry%%:*}"
  quantized="${entry##*:}"
  short="${model##*/}"
  mem="$(mem_for "$model")"

  echo "=== $short (quantized in paper: $quantized) ==="

  for exp in $EXPERIMENTS; do
    case "$exp" in

      joint)
        # Experiment 1. Run at the paper's quantization so the joint ΔC is comparable with the
        # published steering ΔC for the same model.
        submit "$HERE/collect_joint.sh" "--job-name=rev_joint_${short}" MODEL="$model"
        ;;

      unquantized)
        # Experiment 2. Both conditions, unquantized -- so --mem is raised: bf16 weights are
        # several times larger than the 4-bit ones the published runs used.
        submit "$HERE/collect_unquantized.sh" "--mem=$mem --job-name=rev_unq_p_${short}" \
          MODEL="$model" CONDITION=primed
        submit "$HERE/collect_unquantized.sh" "--mem=$mem --job-name=rev_unq_n_${short}" \
          MODEL="$model" CONDITION=null
        ;;

      refnull)
        # Experiment 3. Match the paper's quantization for this model so the referent-only
        # condition sits alongside the published contextual and null conditions.
        submit "$HERE/collect_referent_null.sh" "--job-name=rev_refnull_${short}" \
          MODEL="$model" QUANTIZED="$quantized"
        ;;

      *)
        echo "  unknown experiment: $exp" >&2; exit 1 ;;
    esac
  done
  echo
done

cat <<'EOF'
Once the jobs finish, run the analyses (CPU only, no SLURM needed):

    bash revisions/slurm/run_analyses.sh

Experiment 4 (the per-item table) needs no new collection -- it reads the existing
one_pronoun_measurements_*.ndjson files.
EOF
