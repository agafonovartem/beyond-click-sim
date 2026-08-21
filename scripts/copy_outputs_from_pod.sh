#!/bin/bash
# Copy outputs_<pod>.zip from agent-exp-<pod> to the current directory.
#
# Usage: copy_outputs_from_pod.sh <pod>
# Example: copy_outputs_from_pod.sh 0
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <pod>" >&2
    exit 1
fi

POD="$1"
LOCAL_ZIP="outputs_${POD}.zip"

rm "$LOCAL_ZIP"
kubectl cp "agent-exp-${POD}:/home/jovyan/outputs_${POD}.zip" "$LOCAL_ZIP"
