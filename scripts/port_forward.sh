#!/bin/bash
# Starts a self-restarting `kubectl port-forward` for each pod's litellm proxy
# (remote port $PROXY_PORT) to a local port, in the order pods are given: pod 1 ->
# BASE_PORT, pod 2 -> BASE_PORT+1, etc.
#
# kubectl port-forward has no built-in reconnect logic: if the tunnel drops (API
# server idle timeout, a local network blip, etc.) it silently stops working until
# something restarts it. Each pod's forward is wrapped in a retry loop that
# relaunches it automatically when it exits, so a dropped tunnel self-heals
# instead of causing a lasting "pod times out" symptom.
#
# Any pre-existing port-forwards for the given pods (e.g. left over from a prior
# run of this script, or from a prior deploy_cluster.sh run) are killed first —
# re-running this never leaves orphaned processes fighting over the same local
# port (which otherwise fails with "address already in use" and silently loses
# track of which process is actually serving traffic).
#
# Usage:
#   port_forward.sh <pod1> [pod2] ...
#   BASE_PORT=8084 port_forward.sh <pod5> [pod6] ...
# Env overrides: BASE_PORT (default 8080), PROXY_PORT (default 8080),
#                PORT_FORWARD_PIDFILE, PORT_FORWARD_LOGDIR, RETRY_DELAY (default 3)
set -euo pipefail

BASE_PORT="${BASE_PORT:-8080}"
PROXY_PORT="${PROXY_PORT:-8080}"
PIDFILE="${PORT_FORWARD_PIDFILE:-/tmp/qwen_cluster_port_forwards.pid}"
LOGDIR="${PORT_FORWARD_LOGDIR:-/tmp/qwen_cluster_pf_logs}"
RETRY_DELAY="${RETRY_DELAY:-3}"

if ! [[ "$BASE_PORT" =~ ^[0-9]+$ ]] || [ "$BASE_PORT" -lt 1 ] || [ "$BASE_PORT" -gt 65535 ]; then
    echo "Invalid base port: $BASE_PORT (expected an integer from 1 to 65535)." >&2
    exit 2
fi
if [ "$#" -gt 0 ] && [ $((BASE_PORT + $# - 1)) -gt 65535 ]; then
    echo "Port range starting at $BASE_PORT for $# pods exceeds 65535." >&2
    exit 2
fi

mkdir -p "$LOGDIR"

pod_is_requested() {
    local candidate="$1" requested
    shift
    for requested in "$@"; do
        [ "$candidate" = "$requested" ] && return 0
    done
    return 1
}

echo "Cleaning up any pre-existing port-forwards for the given pods..."
PIDFILE_TMP="${PIDFILE}.tmp.$$"
: > "$PIDFILE_TMP"
if [ -f "$PIDFILE" ]; then
    while read -r tracked_pod tracked_pid; do
        [ -z "$tracked_pod" ] && continue

        # Migrate the old pid-only format when its active kubectl child identifies
        # the pod. This keeps forwards started before this change running.
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
            # An old watchdog may be between retries and have no kubectl child.
            # Preserve it rather than disrupting an unrelated active batch.
            echo "$tracked_pid" >> "$PIDFILE_TMP"
        fi
    done < "$PIDFILE"
fi
mv "$PIDFILE_TMP" "$PIDFILE"
for pod in "$@"; do
    pkill -f "kubectl port-forward pod/$pod " 2>/dev/null || true
done
sleep 1

# Runs in a background subshell per pod; loops forever, restarting kubectl
# port-forward whenever it exits (dropped tunnel, transient error, etc.).
watch_port_forward() {
    local pod="$1" local_port="$2" remote_port="$3" logfile="$4"
    while true; do
        echo "$(date '+%Y-%m-%d %H:%M:%S') starting port-forward: $pod -> localhost:$local_port" >> "$logfile"
        kubectl port-forward "pod/$pod" "$local_port:$remote_port" >> "$logfile" 2>&1 || true
        echo "$(date '+%Y-%m-%d %H:%M:%S') port-forward for $pod exited, restarting in ${RETRY_DELAY}s..." >> "$logfile"
        sleep "$RETRY_DELAY"
    done
}

i=0
for pod in "$@"; do
    local_port=$((BASE_PORT + i))
    logfile="$LOGDIR/pf_$pod.log"
    watch_port_forward "$pod" "$local_port" "$PROXY_PORT" "$logfile" &
    loop_pid=$!
    disown "$loop_pid" 2>/dev/null || true
    echo "$pod $loop_pid" >> "$PIDFILE"
    echo "pod $pod -> localhost:$local_port (watchdog pid $loop_pid)"
    i=$((i + 1))
done

sleep 2
echo "Port-forwards started (auto-restarting on drop). Watchdog PIDs recorded in $PIDFILE, logs in $LOGDIR."
