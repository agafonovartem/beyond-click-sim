#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${1:?Usage: run_steam_qwen_matrix.sh MODEL_KEY MODE}"
MODE="${2:-full}"

case "${MODEL_KEY}" in
  qwen3_8b)
    MODEL_SLUG="qwen3_8b"
    HISTORY_INTERACTION=(
      "llm_yes_no_litellm_qwen3_8b_with_item_stats"
      "llm_listwise_ranking_litellm_qwen3_8b_with_item_stats"
    )
    AGENT4REC_INTERACTION=(
      "agent4rec_yes_no_litellm_qwen3_8b_traits_taste_gpt4o_mini_no_summary"
      "agent4rec_listwise_ranking_litellm_qwen3_8b_traits_taste_gpt4o_mini_no_summary"
    )
    HISTORY_PREFERENCE=(
      "llm_preference_yes_no_litellm_qwen3_8b"
      "llm_preference_listwise_ranking_litellm_qwen3_8b"
    )
    AGENT4REC_PREFERENCE=(
      "agent4rec_preference_yes_no_litellm_qwen3_8b_traits_taste_gpt4o_mini_no_summary"
      "agent4rec_preference_listwise_ranking_litellm_qwen3_8b_traits_taste_gpt4o_mini_no_summary"
    )
    ;;
  qwen36_27b)
    MODEL_SLUG="qwen36_27b"
    HISTORY_INTERACTION=(
      "llm_yes_no_litellm_qwen36_27b_with_item_stats"
      "llm_listwise_ranking_litellm_qwen36_27b_with_item_stats"
    )
    AGENT4REC_INTERACTION=(
      "agent4rec_yes_no_litellm_qwen36_27b_traits_taste_gpt4o_mini_no_summary"
      "agent4rec_listwise_ranking_litellm_qwen36_27b_traits_taste_gpt4o_mini_no_summary"
    )
    HISTORY_PREFERENCE=(
      "llm_preference_yes_no_litellm_qwen36_27b"
      "llm_preference_listwise_ranking_litellm_qwen36_27b"
    )
    AGENT4REC_PREFERENCE=(
      "agent4rec_preference_yes_no_litellm_qwen36_27b_traits_taste_gpt4o_mini_no_summary"
      "agent4rec_preference_listwise_ranking_litellm_qwen36_27b_traits_taste_gpt4o_mini_no_summary"
    )
    ;;
  *)
    echo "Unknown model key: ${MODEL_KEY}" >&2
    exit 2
    ;;
esac

case "${MODE}" in
  smoke)
    METHOD_SUFFIX="smoke"
    INTERACTION_TASKS=(
      "steam_item_stats_cap20_eval_users1000_cg5_m1_seed0"
    )
    PREFERENCE_TASKS=(
      "steam_preference_cap10_eval_users1000_cg5_m1_seed0"
    )
    ;;
  full)
    METHOD_SUFFIX="full"
    INTERACTION_TASKS=()
    for ratio in 1 3 9 19; do
      for seed in 0 1 2; do
        INTERACTION_TASKS+=(
          "steam_item_stats_cap20_eval_users1000_cg5_m${ratio}_seed${seed}"
        )
      done
    done
    PREFERENCE_TASKS=()
    for ratio in 1 2 3 9; do
      for seed in 0 1 2; do
        PREFERENCE_TASKS+=(
          "steam_preference_cap10_eval_users1000_cg5_m${ratio}_seed${seed}"
        )
      done
    done
    ;;
  *)
    echo "Unknown run mode: ${MODE}" >&2
    exit 2
    ;;
esac

for index in "${!HISTORY_INTERACTION[@]}"; do
  HISTORY_INTERACTION["${index}"]="${HISTORY_INTERACTION[${index}]}_${METHOD_SUFFIX}"
done
for index in "${!AGENT4REC_INTERACTION[@]}"; do
  AGENT4REC_INTERACTION["${index}"]="${AGENT4REC_INTERACTION[${index}]}_${METHOD_SUFFIX}"
done
for index in "${!HISTORY_PREFERENCE[@]}"; do
  HISTORY_PREFERENCE["${index}"]="${HISTORY_PREFERENCE[${index}]}_${METHOD_SUFFIX}"
done
for index in "${!AGENT4REC_PREFERENCE[@]}"; do
  AGENT4REC_PREFERENCE["${index}"]="${AGENT4REC_PREFERENCE[${index}]}_${METHOD_SUFFIX}"
done

REPO_ROOT="${REPO_ROOT:-/llm_storage/beyond-click-sim}"
PYTHON="${PYTHON:-/llm_storage/venvs/bcs-runner-py311/bin/python}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/in_distribution/steam_full_matrix/${RUN_ID}_${MODEL_SLUG}_${MODE}}"
STATUS_FILE="${OUTPUT_ROOT}/queue_status.tsv"

export BEYOND_CLICK_SIM_LITELLM_LOCAL_BASE_URL="${BEYOND_CLICK_SIM_LITELLM_LOCAL_BASE_URL:-http://127.0.0.1:8080/v1}"
export BEYOND_CLICK_SIM_LITELLM_VERSION="${BEYOND_CLICK_SIM_LITELLM_VERSION:-1.91.0}"
export BEYOND_CLICK_SIM_LITELLM_ROUTING_STRATEGY="${BEYOND_CLICK_SIM_LITELLM_ROUTING_STRATEGY:-simple-shuffle}"
export BEYOND_CLICK_SIM_VLLM_VERSION="${BEYOND_CLICK_SIM_VLLM_VERSION:-0.19.1}"
export BEYOND_CLICK_SIM_VLLM_REPLICAS="${BEYOND_CLICK_SIM_VLLM_REPLICAS:-8}"
export BEYOND_CLICK_SIM_VLLM_MAX_MODEL_LEN="${BEYOND_CLICK_SIM_VLLM_MAX_MODEL_LEN:-4096}"
export BEYOND_CLICK_SIM_VLLM_GPU_MEMORY_UTILIZATION="${BEYOND_CLICK_SIM_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

mkdir -p "${OUTPUT_ROOT}"
printf "protocol\tphase\ttask\texit_code\tstarted_at\tfinished_at\n" >"${STATUS_FILE}"

join_by_comma() {
  local IFS=,
  echo "$*"
}

run_batch() {
  local protocol="$1"
  local phase="$2"
  local task="$3"
  local module="$4"
  shift 4
  local methods=("$@")
  local started_at
  local finished_at
  local exit_code

  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  set +e
  "${PYTHON}" -m "${module}" \
    --tasks "${task}" \
    --methods "$(join_by_comma "${methods[@]}")" \
    --output-dir "${OUTPUT_ROOT}/${protocol}/${phase}"
  exit_code=$?
  set -e
  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${protocol}" \
    "${phase}" \
    "${task}" \
    "${exit_code}" \
    "${started_at}" \
    "${finished_at}" \
    >>"${STATUS_FILE}"
  return 0
}

cd "${REPO_ROOT}"

for task in "${INTERACTION_TASKS[@]}"; do
  run_batch \
    interaction \
    history \
    "${task}" \
    runners.in_distribution.interaction_prediction.run \
    "${HISTORY_INTERACTION[@]}"
done

for task in "${PREFERENCE_TASKS[@]}"; do
  run_batch \
    preference \
    history \
    "${task}" \
    runners.in_distribution.preference_prediction.run \
    "${HISTORY_PREFERENCE[@]}"
done

for task in "${INTERACTION_TASKS[@]}"; do
  run_batch \
    interaction \
    agent4rec \
    "${task}" \
    runners.in_distribution.interaction_prediction.run \
    "${AGENT4REC_INTERACTION[@]}"
done

for task in "${PREFERENCE_TASKS[@]}"; do
  run_batch \
    preference \
    agent4rec \
    "${task}" \
    runners.in_distribution.preference_prediction.run \
    "${AGENT4REC_PREFERENCE[@]}"
done

if awk -F '\t' 'NR > 1 && $4 != 0 {failed = 1} END {exit failed ? 0 : 1}' "${STATUS_FILE}"; then
  echo "Queue completed with one or more failed batches. See ${STATUS_FILE}." >&2
  exit 1
fi

echo "Queue completed successfully: ${OUTPUT_ROOT}"
