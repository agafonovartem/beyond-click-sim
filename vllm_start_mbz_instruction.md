# vLLM Llama 3.3 70B startup

Known-good startup path for `srv02`. Logs are split by server.

## Paths

- `srv02` script: `/home/artem.agafonov/run_vllm_llama33_70b_server02.sh`
- `srv02` log: `/home/artem.agafonov/vllm_logs/server02/llama33_70b.log`
- `srv01` logs: `/home/artem.agafonov/vllm_logs/server01/`

## Start on srv02

```bash
ssh srv02
tmux new -s vllm-llama33
cd /home/artem.agafonov
mkdir -p /home/artem.agafonov/vllm_logs/server02
./run_vllm_llama33_70b_server02.sh 2>&1 | tee /home/artem.agafonov/vllm_logs/server02/llama33_70b.log
```

If the tmux session already exists:

```bash
tmux attach -t vllm-llama33
```

## Watch logs

```bash
tail -f /home/artem.agafonov/vllm_logs/server02/llama33_70b.log
```

## Verify API

```bash
curl http://127.0.0.1:8000/v1/models
```

Expected served model name:

```text
llama-3.3-70b-instruct
```

## Current srv02 settings

The script uses GPUs `4,5,6,7`, tensor parallel `4`, port `8000`,
`HF_HUB_OFFLINE=1`, `NCCL_P2P_DISABLE=1`, `--enforce-eager`, and
`--disable-custom-all-reduce`.

Do not reuse `server02` log paths on `srv01`; put `srv01` logs under
`/home/artem.agafonov/vllm_logs/server01/`.
