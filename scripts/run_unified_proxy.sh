#!/bin/bash
# Starts a local litellm proxy that routes across all 4 pods' forwarded litellm
# proxies (localhost:8080-8083 by default), giving your code ONE endpoint that
# schedules requests across every pod and every GPU on every pod:
#
#   your code -> localhost:$PORT (this proxy) -> localhost:8080-8083 (per-pod
#   proxies, from port_forward.sh) -> localhost:8000-8003 on each pod (vLLM
#   instances per GPU, from run_llm.sh)
#
# Must be run AFTER port_forward.sh / deploy_cluster.sh has already set up the
# per-pod forwards — this just adds one more routing layer on top of those.
#
# Runs in the foreground so litellm logs go to this terminal and Ctrl+C stops it.
# Kills any previous instance of itself first so re-running never leaves orphaned
# processes fighting over the port.
#
# Usage: run_unified_proxy.sh [port]   (default port: 9000)
set -euo pipefail

PORT="${1:-9000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$SCRIPT_DIR/local_to_pod_litellm_config.yaml"

LITELLM_BIN="litellm"
if [ -x "$REPO_ROOT/.venv/bin/litellm" ]; then
    LITELLM_BIN="$REPO_ROOT/.venv/bin/litellm"
fi

echo "Cleaning up any previous instance of this proxy..."
pkill -f "litellm --config $CONFIG" 2>/dev/null || true
sleep 1

echo "Starting unified litellm proxy on http://127.0.0.1:$PORT (Ctrl+C to stop)"
echo "Verify in another terminal with: scripts/verify_cluster.sh 1 $PORT"
exec "$LITELLM_BIN" --config "$CONFIG" --port "$PORT" --host 127.0.0.1
