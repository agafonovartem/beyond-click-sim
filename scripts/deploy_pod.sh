#!/bin/bash
# Deploys (or redeploys, idempotently) the vLLM + litellm stack onto a single pod:
# install deps + download the model (skipped if already complete), launch one vLLM
# instance per GPU, then launch the litellm proxy in front of them.
#
# Safe to re-run: it skips the model download if already complete, and always
# restarts the vLLM/litellm tmux sessions so it can be used to recover a pod after
# a reload without re-downloading a large model.
#
# Usage: deploy_pod.sh <pod_name>
# Env overrides: GPU_COUNT (default 4), PROXY_PORT (default 8080),
#                MODEL_NAME (default Qwen/Qwen3.6-27B)
set -euo pipefail

POD="$1"
GPU_COUNT="${GPU_COUNT:-4}"
PROXY_PORT="${PROXY_PORT:-8080}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[$POD] $*"; }

# Waits for an HTTP endpoint inside the pod to return 200, or fails fast if the
# backing tmux session dies first (rather than looping forever on a crashed process).
wait_for_health() {
    local port="$1" session="$2" path="$3" max_wait="$4"
    local waited=0
    while true; do
        local code
        code=$(kubectl exec "$POD" -- bash -lc "curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://localhost:$port$path" 2>/dev/null || echo 000)
        [ "$code" = "200" ] && return 0
        if ! kubectl exec "$POD" -- bash -lc "tmux has-session -t $session" 2>/dev/null; then
            log "ERROR: tmux session '$session' died before becoming healthy. Last log lines:"
            kubectl exec "$POD" -- bash -lc "tail -n 30 ~/$session.log" 2>&1 || true
            return 1
        fi
        waited=$((waited + 10))
        if [ "$waited" -ge "$max_wait" ]; then
            log "ERROR: timed out after ${max_wait}s waiting for '$session' health on port $port"
            return 1
        fi
        sleep 10
    done
}

log "deploying model $MODEL_NAME"
log "copying deployment scripts"
kubectl cp "$SCRIPT_DIR/install.sh" "$POD:/home/jovyan/install.sh"
kubectl cp "$SCRIPT_DIR/run_llm.sh" "$POD:/home/jovyan/run_llm.sh"
kubectl cp "$SCRIPT_DIR/run_litellm.sh" "$POD:/home/jovyan/run_litellm.sh"
kubectl cp "$SCRIPT_DIR/model_complete_check.sh" "$POD:/home/jovyan/model_complete_check.sh"

# Rewrite the litellm backend model id to match the selected MODEL_NAME.
LITELLM_CONFIG_TMP="$(mktemp)"
sed "s|Qwen/Qwen3\\.6-27B|$MODEL_NAME|g" "$SCRIPT_DIR/litellm_config.yaml" > "$LITELLM_CONFIG_TMP"
kubectl cp "$LITELLM_CONFIG_TMP" "$POD:/home/jovyan/litellm_config.yaml"
rm -f "$LITELLM_CONFIG_TMP"

kubectl exec "$POD" -- bash -lc 'chmod +x ~/install.sh ~/run_llm.sh ~/run_litellm.sh ~/model_complete_check.sh'

model_complete() {
    kubectl exec "$POD" -- bash -lc "bash ~/model_complete_check.sh ~/$MODEL_NAME" >/dev/null 2>&1
}

if model_complete; then
    log "model already downloaded and complete, skipping install"
else
    for attempt in 1 2 3; do
        log "running install.sh for $MODEL_NAME (attempt $attempt/3)"
        kubectl exec "$POD" -- bash -lc "cd ~ && tmux kill-session -t install 2>/dev/null; tmux new-session -d -s install \"MODEL_NAME='$MODEL_NAME' bash install.sh > install.log 2>&1\""
        while kubectl exec "$POD" -- bash -lc 'tmux has-session -t install' 2>/dev/null; do
            sleep 15
        done
        if model_complete; then
            log "model download verified complete"
            break
        fi
        log "model incomplete after attempt $attempt (hf download resumes from where it left off on retry)"
    done
    model_complete || { log "ERROR: model download still incomplete after 3 install attempts"; exit 1; }
fi

log "launching $GPU_COUNT vLLM instance(s) for $MODEL_NAME"
for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
    kubectl exec "$POD" -- bash -lc "cd ~ && tmux kill-session -t vllm$gpu 2>/dev/null; tmux new-session -d -s vllm$gpu \"MODEL_NAME='$MODEL_NAME' bash run_llm.sh $gpu > vllm$gpu.log 2>&1\""
done

for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
    port=$((8000 + gpu))
    log "waiting for vllm gpu $gpu (port $port) to become healthy"
    wait_for_health "$port" "vllm$gpu" "/health" 900 || exit 1
    log "vllm gpu $gpu healthy"
done

log "launching litellm proxy on port $PROXY_PORT"
kubectl exec "$POD" -- bash -lc "cd ~ && tmux kill-session -t litellm 2>/dev/null; tmux new-session -d -s litellm 'bash run_litellm.sh $PROXY_PORT > litellm.log 2>&1'"
wait_for_health "$PROXY_PORT" "litellm" "/health/liveliness" 120 || exit 1
log "litellm proxy healthy"

log "pod deployment complete"
