#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <gpu_id>" >&2
    echo "Example: $0 0  # GPU 0, port 8000" >&2
    exit 1
fi

GPU_ID="$1"
if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: gpu_id must be a non-negative integer, got: $GPU_ID" >&2
    exit 1
fi

PORT=$((8000 + GPU_ID))

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
export MODEL_PATH="$HOME/$MODEL_NAME"
export TENSOR_PARALLEL_SIZE=1
export DATA_PARALLEL_SIZE=1
export HF_HUB_DISABLE_XET="1"
export HF_ENDPOINT="http://huggingface.proxy"

# vllm serve requires the model as the first positional argument (not after --port),
# and must point at the local download dir from install.sh's --local-dir download
# rather than the bare repo id, since the default HF endpoint isn't reachable from
# the pod and the --local-dir layout isn't the default HF cache layout vllm expects.
#
# For Qwen3-style models, --reasoning-parser qwen3 + --default-chat-template-kwargs
# '{"enable_thinking": false}' disable thinking/reasoning output by default.
# A client can still opt back in per-request via
# extra_body={"chat_template_kwargs": {"enable_thinking": true}}.
VLLM_EXTRA_ARGS=()
case "$MODEL_NAME" in
    Qwen/*)
        VLLM_EXTRA_ARGS+=(--reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}')
        ;;
esac

CUDA_VISIBLE_DEVICES="$GPU_ID" vllm serve "$MODEL_PATH" --port "$PORT" --served-model-name "$MODEL_NAME" --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" --data-parallel-size "$DATA_PARALLEL_SIZE" --max-model-len 32000 --max-num-seqs=16 --enable-prefix-caching --gpu-memory-utilization 0.95 "${VLLM_EXTRA_ARGS[@]}"

# CUDA_VISIBLE_DEVICES=0 vllm serve --port 8000 $MODEL_NAME --tensor-parallel-size $TENSOR_PARALLEL_SIZE --data-parallel-size $DATA_PARALLEL_SIZE --max-model-len 32000 --max-num-seqs=16 --enable-prefix-caching --gpu-memory-utilization 0.95
# CUDA_VISIBLE_DEVICES=1 vllm serve --port 8001 $MODEL_NAME --tensor-parallel-size $TENSOR_PARALLEL_SIZE --data-parallel-size $DATA_PARALLEL_SIZE --max-model-len 32000 --max-num-seqs=16 --enable-prefix-caching --gpu-memory-utilization 0.95
# CUDA_VISIBLE_DEVICES=2 vllm serve --port 8002 $MODEL_NAME --tensor-parallel-size $TENSOR_PARALLEL_SIZE --data-parallel-size $DATA_PARALLEL_SIZE --max-model-len 32000 --max-num-seqs=16 --enable-prefix-caching --gpu-memory-utilization 0.95
# CUDA_VISIBLE_DEVICES=3 vllm serve --port 8003 $MODEL_NAME --tensor-parallel-size $TENSOR_PARALLEL_SIZE --data-parallel-size $DATA_PARALLEL_SIZE --max-model-len 32000 --max-num-seqs=16 --enable-prefix-caching --gpu-memory-utilization 0.95

# litellm --config litellm_config.yaml --port 8080


# python -m runners.in_distribution.cold_start.run   --methods popularity_f1_threshold,popularity_ranking,item_knn_cold_start,item_knn_cold_start_ranking,llm_yes_no_vllm_qwen36_27b_full   --output-dir outputs/in_distribution/cold_start --resume --tasks ml-1m_cold_start_k10_cap20_eval_users1000_cg5_m1_seed0,ml-1m_cold_start_k10_cap20_eval_users1000_cg5_m1_seed1,ml-1m_cold_start_k10_cap20_eval_users1000_cg5_m1_seed2,ml-1m_cold_start_k10_cap20_eval_users1000_cg5_m2_seed0,ml-1m_cold_start_k10_cap20_eval_users1000_cg5_m2_seed1,ml-1m_cold_start_k10_cap20_eval_users1000_cg5_m2_seed2,ml-1m_cold_start_k10_cap20_eval_users1000_cg5_m3_seed0,ml-1m_cold_start_k10_cap20_eval_users1000_cg5_m3_seed1,ml-1m_cold_start_k10_cap20_eval_users1000_cg5_m3_seed2,ml-1m_cold_start_k10_cap20_eval_users1000_cg5_m9_seed0,ml-1m_cold_start_k10_cap20_eval_users1000_cg5_m9_seed1,ml-1m_cold_start_k10_cap20_eval_users1000_cg5_m9_seed2,ml-1m_cold_start_k10_cap20_eval_users1000_cg5_m19_seed0,ml-1m_cold_start_k10_cap20_eval_users1000_cg5_m19_seed1,ml-1m_cold_start_k10_cap20_eval_users1000_cg5_m19_seed2,ml-1m_cold_start_k15_cap20_eval_users1000_cg5_m1_seed0,ml-1m_cold_start_k15_cap20_eval_users1000_cg5_m1_seed1,ml-1m_cold_start_k15_cap20_eval_users1000_cg5_m1_seed2,ml-1m_cold_start_k15_cap20_eval_users1000_cg5_m2_seed0,ml-1m_cold_start_k15_cap20_eval_users1000_cg5_m2_seed1,ml-1m_cold_start_k15_cap20_eval_users1000_cg5_m2_seed2,ml-1m_cold_start_k15_cap20_eval_users1000_cg5_m3_seed0,ml-1m_cold_start_k15_cap20_eval_users1000_cg5_m3_seed1,ml-1m_cold_start_k15_cap20_eval_users1000_cg5_m3_seed2,ml-1m_cold_start_k15_cap20_eval_users1000_cg5_m9_seed0,ml-1m_cold_start_k15_cap20_eval_users1000_cg5_m9_seed1,ml-1m_cold_start_k15_cap20_eval_users1000_cg5_m9_seed2,ml-1m_cold_start_k15_cap20_eval_users1000_cg5_m19_seed0,ml-1m_cold_start_k15_cap20_eval_users1000_cg5_m19_seed1,ml-1m_cold_start_k15_cap20_eval_users1000_cg5_m19_seed2,ml-1m_cold_start_k20_cap20_eval_users1000_cg5_m1_seed0,ml-1m_cold_start_k20_cap20_eval_users1000_cg5_m1_seed1,ml-1m_cold_start_k20_cap20_eval_users1000_cg5_m1_seed2,ml-1m_cold_start_k20_cap20_eval_users1000_cg5_m2_seed0,ml-1m_cold_start_k20_cap20_eval_users1000_cg5_m2_seed1,ml-1m_cold_start_k20_cap20_eval_users1000_cg5_m2_seed2,ml-1m_cold_start_k20_cap20_eval_users1000_cg5_m3_seed0,ml-1m_cold_start_k20_cap20_eval_users1000_cg5_m3_seed1,ml-1m_cold_start_k20_cap20_eval_users1000_cg5_m3_seed2,ml-1m_cold_start_k20_cap20_eval_users1000_cg5_m9_seed0,ml-1m_cold_start_k20_cap20_eval_users1000_cg5_m9_seed1,ml-1m_cold_start_k20_cap20_eval_users1000_cg5_m9_seed2,ml-1m_cold_start_k20_cap20_eval_users1000_cg5_m19_seed0,ml-1m_cold_start_k20_cap20_eval_users1000_cg5_m19_seed1,ml-1m_cold_start_k20_cap20_eval_users1000_cg5_m19_seed2 --resume