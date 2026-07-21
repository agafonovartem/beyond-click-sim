#!/bin/bash
# Checks whether a local vLLM model download directory has all its safetensors shards.
#
# install.sh's `hf download --local-dir` can be interrupted mid-download (observed:
# a transient proxy error left 14/15 shards on one pod). Re-running install.sh resumes
# the download (hf download skips files already present), but something has to detect
# the incomplete state first — that's what this does, by comparing the shard count
# against the "-of-NNNNN" total encoded in the shard filenames.
#
# Usage: model_complete_check.sh <model_dir>
# Exit 0 and print "complete: X/X shards" if all shards are present, exit 1 otherwise.
set -euo pipefail

DIR="$1"
first=$(ls "$DIR"/model-*-of-*.safetensors 2>/dev/null | head -1 || true)
if [ -z "$first" ]; then
    echo "incomplete: no shards found in $DIR"
    exit 1
fi

total=$(echo "$first" | sed -E 's/.*-of-0*([0-9]+)\.safetensors/\1/')
count=$(ls "$DIR"/model-*-of-*.safetensors 2>/dev/null | wc -l | tr -d ' ')

if [ "$count" -eq "$total" ]; then
    echo "complete: $count/$total shards"
    exit 0
else
    echo "incomplete: $count/$total shards"
    exit 1
fi
