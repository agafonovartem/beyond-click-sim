#!/bin/bash

conda deactivate

pip install tokenizers --only-binary=:all:
pip install -U "huggingface_hub[cli]" 
pip install huggingface-hub==0.34.0 vllm==0.18.1 'litellm[proxy]'

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
export HF_HUB_DISABLE_XET="1"
export HF_ENDPOINT="http://huggingface.proxy"
hf download "$MODEL_NAME" --local-dir ~/"$MODEL_NAME"