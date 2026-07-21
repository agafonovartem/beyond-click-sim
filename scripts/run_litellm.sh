#!/bin/bash
# Runs the litellm proxy for this pod's local vLLM instances.
#
# --host 0.0.0.0 is required: without it, litellm/uvicorn's default host resolution
# fails with a DNS lookup error immediately after "Application startup complete" in
# this cluster's pods, and the process exits.
#
# Usage: run_litellm.sh [port]
set -euo pipefail

PORT="${1:-8080}"
cd ~
litellm --config litellm_config.yaml --port "$PORT" --host 0.0.0.0
