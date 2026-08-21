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

if [ "$count" -ne "$total" ]; then
    echo "incomplete: $count/$total shards"
    exit 1
fi

# Shard count alone is not enough. hf download fetches files concurrently, so an
# interrupted download can leave all shards present while the much smaller tokenizer
# files are still missing. That combination passed this check and then crashed vLLM at
# startup with a misleading sentencepiece error ("Either model_file or model_proto must
# be specified"), because transformers falls back to the slow tokenizer when
# tokenizer.json is absent. Verify the small files too.
for required in config.json tokenizer_config.json; do
    if [ ! -f "$DIR/$required" ]; then
        echo "incomplete: $count/$total shards but $required is missing"
        exit 1
    fi
done

if [ ! -f "$DIR/tokenizer.json" ] && [ ! -f "$DIR/tokenizer.model" ]; then
    echo "incomplete: $count/$total shards but no tokenizer.json or tokenizer.model"
    exit 1
fi

echo "complete: $count/$total shards + tokenizer"
exit 0
