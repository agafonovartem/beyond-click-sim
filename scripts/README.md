# Redeploying the Qwen3.6-27B vLLM + litellm cluster (manual, no LLM needed)

This is a plain step-by-step guide for a human to run the redeploy directly from a
terminal — no need to ask an LLM to do it each time. It uses the same scripts in
this folder that already encode the deployment logic and known fixes; you're just
invoking them yourself.

(There's also `AGENTS.md` in this folder, which is the equivalent reference written
for an LLM agent to follow. This file covers the same ground for a human running
commands directly.)

For recurring cold-start experiment execution (upload/setup/queue/download across
`agent-exp-0..7`), use `COLD_START_POD_QUEUE_RUNBOOK.md`.

## 0. Prerequisites

```bash
kubectl config current-context        # confirm you're pointed at the right cluster
kubectl get pods | grep agent-exp     # confirm the pods are Running
```

Note the pod names — you'll pass them to the scripts below. Everything after this
assumes you're in the repo root (`cd` there first) so the relative `scripts/...`
paths work.

## 1. Deploy

```bash
scripts/deploy_cluster.sh <pod1> <pod2> <pod3> <pod4>
```

Example, matching the current naming convention:

```bash
scripts/deploy_cluster.sh agent-exp-policy-ranking-0 agent-exp-policy-ranking-1 agent-exp-policy-ranking-2 agent-exp-policy-ranking-3
```

Or, if your pods follow a `<prefix>-0..N-1` naming scheme:

```bash
POD_PREFIX=agent-exp-policy-ranking POD_COUNT=4 scripts/deploy_cluster.sh
```

### Base port / multi-batch deploys

By default pods are forwarded to `localhost:8080`, `8081`, … in argument order.
To start another batch without colliding with existing forwards, set the first
local port with `--base-port` (or the `BASE_PORT` env var). The flag can appear
before or after the pod names:

```bash
scripts/deploy_cluster.sh --base-port 8088 pod-a pod-b
scripts/deploy_cluster.sh pod-a pod-b --base-port 8088
BASE_PORT=8088 scripts/deploy_cluster.sh pod-a pod-b

scripts/verify_cluster.sh 2 8088   # checks localhost:8088 and 8089
```

Ports are assigned by integer addition (`BASE_PORT`, `BASE_PORT+1`, …), so this
works for batches larger than 10 pods. Valid values are TCP ports `1–65535`
(the full range for that batch must fit under 65535).

Teardown is pod-scoped: `scripts/teardown_cluster.sh pod-a pod-b` stops only
those pods' forwards and leaves other batches running. The optional unified
proxy remains fixed to the four backends at `8080-8083`.

### Model selection

Default model is `Qwen/Qwen3.6-27B`. Override with `--model` (or `MODEL_NAME`)
to download and serve a different Hugging Face id on those pods:

```bash
scripts/deploy_cluster.sh --model Qwen/Qwen3-8B pod-a pod-b
scripts/deploy_cluster.sh --model meta-llama/Llama-3.3-70B-Instruct --base-port 8090 pod-c pod-d

MODEL_NAME=Qwen/Qwen3-8B scripts/verify_cluster.sh 2 8080
```

The selected id is used for `hf download`, the on-pod path (`~/$MODEL_NAME`),
vLLM `--served-model-name`, and the pod litellm config. Qwen models still get
thinking disabled by default; other families skip those Qwen-specific vLLM flags.
The local unified proxy YAML still hardcodes `Qwen/Qwen3.6-27B` backends — edit
or bypass it when verifying a different model via per-pod ports.

This runs in the foreground and prints one line per pod per stage (copy scripts →
install/download → launch vLLM per GPU → launch litellm). **It blocks until
everything is healthy or a pod fails** — if you want to keep using your terminal,
run it with `nohup ... &` and `tail -f` the output, or just run it in a background
job / separate terminal tab.

### How long this takes

Based on past runs, expect roughly **15-25 minutes total** for a from-scratch
redeploy (pods with no prior install), most of it pip installs + the ~52GB model
download, run in parallel across pods:

- pip install (`vllm`, `litellm[proxy]`, `huggingface_hub`): a few minutes; can
  occasionally take noticeably longer on one pod if dependency resolution is slow
  — that pod will just lag behind the others, it's not stuck.
- Model download (`hf download`, resumable): a few minutes on the internal mirror.
- vLLM load + `torch.compile` warmup per GPU: roughly 2-3 minutes per instance.
- litellm proxy startup: seconds.

If a pod's model was already downloaded from a previous deploy (i.e. only the pod
process was restarted, not truly wiped), that pod skips straight to the vLLM step
and finishes much faster.

## 2. Verify

```bash
scripts/verify_cluster.sh
```

Or a single manual check:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.6-27B","messages":[{"role":"user","content":"Say OK."}],"max_tokens":10,"chat_template_kwargs":{"enable_thinking":false}}'
```

### Reasoning/thinking is off by default

`run_llm.sh` starts vLLM with `--reasoning-parser qwen3 --default-chat-template-kwargs
'{"enable_thinking": false}'`, so responses come back as direct answers, not
prefixed with a "Here's a thinking process..." preamble. To re-enable it for a
specific request, add `"chat_template_kwargs": {"enable_thinking": true}` (via
`extra_body` if using the OpenAI Python client) to that request.

This checks `/v1/models` and sends one real chat completion through each of
`localhost:8080`, `8081`, `8082`, `8083`. All four should print `OK: {...}`. If one
fails, it isolates the problem to that specific pod (see Troubleshooting below).

## 3. Use it

Each of `localhost:8080-8083` is now an OpenAI-compatible endpoint for
`Qwen/Qwen3.6-27B`, load-balanced (least-busy) across that pod's 4 GPUs.

### Steam in-distribution matrix

When one Qwen model and its local LiteLLM proxy are already running inside a
GPU pod, the following launcher runs the matched Steam interaction/preference
matrix and writes every artifact under the repository `outputs/` directory:

```bash
scripts/run_steam_qwen_matrix.sh qwen3_8b smoke
scripts/run_steam_qwen_matrix.sh qwen3_8b full
scripts/run_steam_qwen_matrix.sh qwen36_27b smoke
scripts/run_steam_qwen_matrix.sh qwen36_27b full
```

The interaction grid uses `m={1,3,9,19}`; preference uses
`m={1,2,3,9}`; both use seeds `0,1,2`. Each grid contains History yes/no,
History listwise, Agent4Rec `traits+taste` yes/no, and Agent4Rec
`traits+taste` listwise. History interaction exposes train-only item
statistics; Agent4Rec uses no item summaries. The launcher records each
task/phase exit code in `queue_status.tsv` and continues to later batches after
an isolated failure.

### Optional: one unified endpoint across all pods

If your code should hit a single URL instead of picking between 4, start the
unified proxy (`local_to_pod_litellm_config.yaml`, which points at `localhost:8080-8083`):

```bash
scripts/run_unified_proxy.sh          # default port 9000
scripts/verify_cluster.sh 1 9000      # verify just this one endpoint
```

This adds one more routing layer: your code → unified proxy → one of the 4 per-pod
proxies (least-busy) → one of that pod's 4 GPUs (least-busy). It self-restarts on
crash, same as `port_forward.sh`.

If the default port is already taken by something unrelated on your machine, check
first (`lsof -nP -iTCP:9000 -sTCP:LISTEN`) — don't kill a process you don't
recognize — and just pass a different port: `scripts/run_unified_proxy.sh 8090`.

## 4. Tear down (when done)

```bash
scripts/teardown_cluster.sh <pod1> <pod2> <pod3> <pod4>
```

Kills the tmux sessions on each pod, stops the local `kubectl port-forward`
watchdogs (see below), and stops the unified proxy if one is running. The
downloaded model is left on each pod so a future redeploy (as long as the pod
itself isn't wiped) skips the download.

## A pod "intermittently times out" — is that normal?

`kubectl port-forward` tunnels through the K8s API server and has no built-in
reconnect logic: if the tunnel drops (API server idle timeout, a local network
blip, etc.), it silently stops forwarding until restarted — even though the pod,
vLLM, and litellm are completely healthy. `port_forward.sh` now wraps each pod's
forward in a retry loop that auto-restarts it on drop, and cleans up any stale
forwards before starting new ones (so re-running `deploy_cluster.sh` never leaves
orphaned processes fighting over the same port). If you still notice timeouts,
check the relevant log:

```bash
tail -f /tmp/qwen_cluster_pf_logs/pf_<pod>.log
```

Repeated "exited, restarting" lines confirm it's the tunnel dropping (self-healing
now, but still worth knowing it's happening), not a pod/model problem.

## Monitoring manually while a deploy is running

From another terminal, on any pod:

```bash
kubectl exec <pod> -- tmux ls                     # which stages are running: install, vllm0-3, litellm
kubectl exec <pod> -- tail -n 20 ~/install.log     # pip/download progress
kubectl exec <pod> -- tail -n 20 ~/vllm0.log       # vLLM load/compile progress for GPU 0
kubectl exec <pod> -- tail -n 20 ~/litellm.log     # litellm proxy startup
```

A tmux session disappearing means that stage's script exited (successfully or
not) — check the corresponding `.log` for the outcome.

## Troubleshooting

| Symptom | What it means | What to do |
|---|---|---|
| `verify_cluster.sh` fails on one port, others pass | Problem isolated to one pod | Check that pod's `tmux ls` + logs directly; no need to touch the others |
| One pod's endpoint intermittently times out, others are fine | Almost always the local `kubectl port-forward` tunnel dropping, not the pod itself (see section above) | Check `/tmp/qwen_cluster_pf_logs/pf_<pod>.log` for restart lines; the watchdog should already have reconnected on its own |
| A pod's `install` tmux session ends but the model directory has fewer than 15 `.safetensors` shards | Partial/interrupted download (seen once — a transient proxy hiccup) | Just re-run `scripts/deploy_cluster.sh` for that pod again — `deploy_pod.sh` detects this via `model_complete_check.sh` and retries automatically (up to 3x) |
| `litellm` tmux session disappears shortly after starting | Missing `--host 0.0.0.0` would cause this (already handled in `run_litellm.sh`) — if you see it anyway, something in the script chain didn't get copied/updated | `kubectl exec <pod> -- tail -n 40 ~/litellm.log` and look for a DNS/host resolution error right after "Application startup complete" |
| A vLLM tmux session dies quickly with an argument-parsing error | `vllm serve` was invoked with the model after a flag instead of as the first positional arg | Shouldn't happen with the current `run_llm.sh` — if it does, the copied script is stale; re-run `deploy_cluster.sh` to re-copy it |
| Whole thing seems to hang on one pod for a long time with no log movement | Usually just slow pip dependency resolution, not a hang | Check `tail -n 5 ~/install.log` on that pod — if the last lines change between two checks a minute apart, it's progressing |

Re-running `scripts/deploy_cluster.sh` (or `scripts/deploy_pod.sh <pod>` for a
single pod) is always safe — every step is idempotent.
