#!/bin/bash
# Sanity-checks a deployed cluster: for each local port, confirms /v1/models responds
# and a real chat completion comes back.
#
# Usage: scripts/verify_cluster.sh [num_ports] [base_port]
# Env overrides: MODEL_NAME (default Qwen/Qwen3.6-27B)
set -euo pipefail

NUM="${1:-4}"
BASE_PORT="${2:-8080}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"

# Qwen thinking kwargs are only needed/safe for Qwen chat templates.
EXTRA_BODY_FIELDS=""
case "$MODEL_NAME" in
    Qwen/*)
        EXTRA_BODY_FIELDS=',"chat_template_kwargs":{"enable_thinking":false}'
        ;;
esac

fail=0
for ((i = 0; i < NUM; i++)); do
    port=$((BASE_PORT + i))
    echo "=== localhost:$port ($MODEL_NAME) ==="
    if ! curl -sf --max-time 5 "http://localhost:$port/v1/models" | grep -q "$MODEL_NAME"; then
        echo "FAIL: /v1/models did not return $MODEL_NAME"
        fail=1
        continue
    fi

    resp=$(curl -sf --max-time 60 "http://localhost:$port/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":10${EXTRA_BODY_FIELDS}}") || resp=""
    if [ -z "$resp" ]; then
        echo "FAIL: empty/errored completion response"
        fail=1
        continue
    fi
    echo "OK: $resp"
done

exit $fail
