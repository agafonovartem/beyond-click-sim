#!/bin/bash
# Tears down a deployed cluster: kills the vllm/litellm tmux sessions on each pod
# (leaving the downloaded model in place for a faster redeploy), stops the local
# port-forward watchdog loops (from port_forward.sh) plus any currently-running
# kubectl port-forward processes for these pods, and stops the unified proxy
# (from run_unified_proxy.sh) if one is running.
#
# Usage: scripts/teardown_cluster.sh <pod1> [pod2] ...
# Only the requested pods' forwarding watchdogs are stopped; other batches remain.
# Env overrides: GPU_COUNT (default 4), PORT_FORWARD_PIDFILE, UNIFIED_PROXY_PIDFILE
set -euo pipefail

GPU_COUNT="${GPU_COUNT:-4}"
PIDFILE="${PORT_FORWARD_PIDFILE:-/tmp/qwen_cluster_port_forwards.pid}"
UNIFIED_PIDFILE="${UNIFIED_PROXY_PIDFILE:-/tmp/qwen_unified_proxy.pid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pod_is_requested() {
    local candidate="$1" requested
    shift
    for requested in "$@"; do
        [ "$candidate" = "$requested" ] && return 0
    done
    return 1
}

for pod in "$@"; do
    echo "[$pod] killing tmux sessions"
    sessions="litellm install"
    for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do sessions="$sessions vllm$gpu"; done
    kubectl exec "$pod" -- bash -lc "for s in $sessions; do tmux kill-session -t \"\$s\" 2>/dev/null || true; done" \
        || echo "[$pod] pod unreachable, skipping"
done

echo "Stopping local port-forward watchdogs..."
if [ -f "$PIDFILE" ]; then
    # Kill the watchdog retry loops first, before touching kubectl itself —
    # otherwise a loop would just relaunch a replacement the moment its
    # current kubectl port-forward process is killed below.
    PIDFILE_TMP="${PIDFILE}.tmp.$$"
    : > "$PIDFILE_TMP"
    while read -r tracked_pod tracked_pid; do
        [ -z "$tracked_pod" ] && continue

        # Migrate the old pid-only format when its active kubectl child identifies
        # the pod. Unknown legacy entries are preserved rather than killed globally.
        if [ -z "${tracked_pid:-}" ]; then
            tracked_pid="$tracked_pod"
            tracked_pod=""
            while read -r child_pid; do
                command="$(ps -p "$child_pid" -o command= 2>/dev/null || true)"
                if [[ "$command" =~ kubectl[[:space:]]+port-forward[[:space:]]+pod/([^[:space:]]+) ]]; then
                    tracked_pod="${BASH_REMATCH[1]}"
                    break
                fi
            done < <(pgrep -P "$tracked_pid" 2>/dev/null || true)
        fi

        if [ -n "$tracked_pod" ] && pod_is_requested "$tracked_pod" "$@"; then
            kill "$tracked_pid" 2>/dev/null || true
        elif [ -n "$tracked_pod" ]; then
            echo "$tracked_pod $tracked_pid" >> "$PIDFILE_TMP"
        else
            echo "$tracked_pid" >> "$PIDFILE_TMP"
        fi
    done < "$PIDFILE"
    if [ -s "$PIDFILE_TMP" ]; then
        mv "$PIDFILE_TMP" "$PIDFILE"
    else
        rm -f "$PIDFILE_TMP" "$PIDFILE"
    fi
fi
# Sweep any remaining kubectl port-forward processes for these pods — covers
# both the loop's currently-running child and any orphans from earlier runs
# that a stale/overwritten pidfile lost track of.
for pod in "$@"; do
    pkill -f "kubectl port-forward pod/$pod " 2>/dev/null || true
done

echo "Stopping unified proxy (if running)..."
if [ -f "$UNIFIED_PIDFILE" ]; then
    while read -r pid; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    done < "$UNIFIED_PIDFILE"
    rm -f "$UNIFIED_PIDFILE"
fi
pkill -f "litellm --config $SCRIPT_DIR/local_to_pod_litellm_config.yaml" 2>/dev/null || true

echo "Teardown complete."
