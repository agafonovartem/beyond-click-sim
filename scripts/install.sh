#!/bin/bash

conda deactivate

pip install tokenizers --only-binary=:all:
pip install -U "huggingface_hub[cli]" 
# litellm and fastapi are pinned to the pair uv.lock resolves. Unpinned,
# `litellm[proxy]` installs 1.97.0 against the image's fastapi 0.141.1, and the proxy
# dies at startup: fastapi removed `get_flat_dependant` in 0.141, which litellm's
# management endpoints import. litellm reports that as a misleading
# "ModuleNotFoundError: No module named 'proxy_server'" — the real ImportError is
# swallowed higher in the traceback. 1.91.x also matches the version the runners
# record in run manifests as BEYOND_CLICK_SIM_LITELLM_VERSION.
pip install huggingface-hub==0.34.0 vllm==0.18.1 'litellm[proxy]==1.91.1' 'fastapi==0.139.0'

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
export HF_HUB_DISABLE_XET="1"
export HF_ENDPOINT="http://huggingface.proxy"

# Gated repos (meta-llama/*) need a token even through the internal mirror: the
# mirror proxies the gate, so an unauthenticated resolve of config.json comes back
# 401 while public repos like Qwen/* return 200. HF_TOKEN is passed in by
# deploy_pod.sh when set; ungated models don't need it.
#
# `original/` holds Meta's consolidated .pth checkpoint — a second full copy of the
# weights (another ~132GB for Llama-70B) that vLLM never loads, since it serves from
# the safetensors shards. Downloading it doubles both transfer time and disk, and
# overflows the tighter pods. Excluding it is a no-op for repos that don't have one.
hf download "$MODEL_NAME" --local-dir ~/"$MODEL_NAME" --exclude "original/*"