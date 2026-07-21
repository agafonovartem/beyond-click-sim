#!/bin/bash
# Deploys the Qwen vLLM + litellm stack across a set of pods (in parallel), then
# port-forwards each pod's proxy to localhost:$BASE_PORT, $BASE_PORT+1, ...
#
# Usage:
#   scripts/deploy_cluster.sh <pod1> <pod2> ...
#   scripts/deploy_cluster.sh --base-port 8082 <pod1> <pod2> ...
#   scripts/deploy_cluster.sh --model Qwen/Qwen3-8B <pod1> <pod2> ...
#   POD_PREFIX=agent-exp-policy-ranking POD_COUNT=4 scripts/deploy_cluster.sh
#
# After this completes, run scripts/verify_cluster.sh to sanity-check reachability,
# and scripts/teardown_cluster.sh <pods...> to tear everything down later.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_PORT="${BASE_PORT:-8080}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
PODS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --base-port)
            if [ "$#" -lt 2 ]; then
                echo "Usage: $0 [--base-port <port>] [--model <hf_id>] [pod1 pod2 ...]" >&2
                exit 2
            fi
            BASE_PORT="$2"
            shift 2
            ;;
        --model)
            if [ "$#" -lt 2 ]; then
                echo "Usage: $0 [--base-port <port>] [--model <hf_id>] [pod1 pod2 ...]" >&2
                exit 2
            fi
            MODEL_NAME="$2"
            shift 2
            ;;
        --*)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
        *)
            PODS+=("$1")
            shift
            ;;
    esac
done

if ! [[ "$BASE_PORT" =~ ^[0-9]+$ ]] || [ "$BASE_PORT" -lt 1 ] || [ "$BASE_PORT" -gt 65535 ]; then
    echo "Invalid base port: $BASE_PORT (expected an integer from 1 to 65535)." >&2
    exit 2
fi
export BASE_PORT

if [ -z "$MODEL_NAME" ]; then
    echo "MODEL_NAME must be a non-empty Hugging Face model id (e.g. Qwen/Qwen3.6-27B)." >&2
    exit 2
fi
export MODEL_NAME

if [ "${#PODS[@]}" -eq 0 ]; then
    PREFIX="${POD_PREFIX:-agent-exp-policy-ranking}"
    COUNT="${POD_COUNT:-4}"
    for ((i = 0; i < COUNT; i++)); do PODS+=("${PREFIX}-${i}"); done
fi

LAST_PORT=$((BASE_PORT + ${#PODS[@]} - 1))
if [ "$LAST_PORT" -gt 65535 ]; then
    echo "Port range $BASE_PORT-$LAST_PORT exceeds the maximum TCP port 65535." >&2
    exit 2
fi

echo "Deploying model $MODEL_NAME to pods: ${PODS[*]} (local ports $BASE_PORT-$LAST_PORT)"

pids=()
for pod in "${PODS[@]}"; do
    "$SCRIPT_DIR/deploy_pod.sh" "$pod" &
    pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done

if [ "$fail" -ne 0 ]; then
    echo "One or more pods failed to deploy — see output above." >&2
    exit 1
fi

echo "All pods deployed. Setting up port-forwards..."
"$SCRIPT_DIR/port_forward.sh" "${PODS[@]}"
