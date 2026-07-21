# scripts/ — Qwen3.6-27B vLLM + litellm cluster deployment

This directory deploys `Qwen/Qwen3.6-27B` across a set of Kubernetes pods: each pod
runs one vLLM instance per GPU, fronted by a local litellm proxy (least-busy routing
across the GPUs on that pod), reachable from the local machine via `kubectl
port-forward`. On top of that, `run_unified_proxy.sh` runs one more local litellm
proxy that routes across all the pods' forwarded proxies, so client code only ever
needs to talk to a single endpoint that schedules across every pod and every GPU:

```
your code -> localhost:$UNIFIED_PORT (run_unified_proxy.sh, local_to_pod_litellm_config.yaml)
          -> localhost:8080-8083     (port_forward.sh, one per pod)
          -> localhost:8000-8003 on each pod (run_llm.sh, one per GPU)
```

If the user says something like "look at the scripts folder and run deployment on
these pods: <pod names>", this file is everything you need. Do not re-derive the
approach from scratch — follow it.

## Prerequisites (verify, don't assume)

- `kubectl` must already be pointed at the right context/cluster (`kubectl config
  current-context`). Do not change kubeconfig or context yourself; ask the user if
  it looks wrong.
- Pods must already exist and be `Running` (`kubectl get pods`). This tooling does
  not create pods — only deploys onto existing ones.
- Each pod is assumed to have N GPUs (`kubectl exec <pod> -- nvidia-smi -L` to
  confirm) — default assumed is 4. If a pod has a different GPU count, pass/export
  `GPU_COUNT` accordingly.
- Pods need outbound access to `http://huggingface.proxy` and an internal PyPI
  mirror (both used by `install.sh`); direct `huggingface.co` is normally
  unreachable from these pods — that's expected, not a bug to fix.

## One-shot redeploy

```bash
scripts/deploy_cluster.sh <pod1> <pod2> ...
# or, if pods follow a `<prefix>-0..N-1` naming convention:
POD_PREFIX=agent-exp-policy-ranking POD_COUNT=4 scripts/deploy_cluster.sh
# optional: choose a different HF model (default Qwen/Qwen3.6-27B) and/or local base port:
scripts/deploy_cluster.sh --model Qwen/Qwen3-8B --base-port 8088 <pod1> <pod2> ...

scripts/verify_cluster.sh          # curl /v1/models + a real completion on each local port
# for a non-default model / port offset:
MODEL_NAME=Qwen/Qwen3-8B scripts/verify_cluster.sh 2 8088

scripts/run_unified_proxy.sh [port]   # optional: one endpoint routing across all pods (default port 9000)
scripts/verify_cluster.sh 1 <port>    # verify just that single unified endpoint
```

Additional batches can coexist by choosing a non-overlapping base port:
`scripts/deploy_cluster.sh --base-port 8084 <pod5> <pod6>`, then
`scripts/verify_cluster.sh 2 8084`. Port numbers use integer addition and support
arbitrary batch sizes up to TCP port 65535. Port-forward tracking and teardown are
pod-scoped, so deploying or tearing down one batch leaves the others running.
The unified proxy config is still fixed to the four backends at `8080-8083`.

That's the whole flow. `deploy_cluster.sh` deploys to all given pods in parallel,
then sets up port-forwards automatically. After it returns 0, the cluster is up and
`verify_cluster.sh` should pass. `run_unified_proxy.sh` is optional and separate —
run it whenever client code needs a single endpoint instead of 4 per-pod ones. If
the default port (9000) is already taken by something unrelated on the machine
(check with `lsof -nP -iTCP:<port> -sTCP:LISTEN` first — don't assume it's free,
and don't kill whatever's using it if it isn't yours), just pass a different port.

Tear down when done:

```bash
scripts/teardown_cluster.sh <pod1> <pod2> ...
```

This kills the vLLM/litellm tmux sessions, stops the local port-forwards, and stops
the unified proxy if one is running, but **leaves the downloaded model on each
pod** so the next deploy skips the ~52GB download.

## What each script does (in call order)

| Script | Role |
|---|---|
| `deploy_cluster.sh` | Orchestrator. Runs `deploy_pod.sh` per pod in parallel, then `port_forward.sh`. Entry point for a full redeploy. |
| `deploy_pod.sh <pod>` | Per-pod pipeline: copy scripts → install deps/model (skipped if already complete, retried up to 3x if incomplete) → launch vLLM per GPU → launch litellm → poll health at each step. Idempotent; safe to re-run on a pod that's already up. |
| `install.sh` | Installs `vllm`/`litellm`/`huggingface_hub`, downloads the model to `~/Qwen/Qwen3.6-27B` via `hf download --local-dir`. Runs inside the pod. |
| `run_llm.sh <gpu_id>` | Starts one vLLM instance on `CUDA_VISIBLE_DEVICES=<gpu_id>`, port `8000+gpu_id`. Runs inside the pod. |
| `run_litellm.sh [port]` | Starts the litellm proxy (default port 8080) in front of the pod's local vLLM instances. Runs inside the pod. |
| `model_complete_check.sh <dir>` | Exit 0 iff all safetensors shards for a model dir are present (compares count against the `-of-NNNNN` suffix in filenames). Used by `deploy_pod.sh` to decide whether to (re)download. |
| `port_forward.sh <pods...>` | Kills any pre-existing forwards for these pods, then starts one self-restarting `kubectl port-forward` per pod: pod *i* → `localhost:<BASE_PORT + i>` (default `BASE_PORT=8080`). Pod/PID pairs are recorded in `/tmp/qwen_cluster_port_forwards.pid`. |
| `run_unified_proxy.sh [port]` | Optional extra layer: a local, self-restarting litellm proxy (config: `local_to_pod_litellm_config.yaml`) that load-balances across the 4 per-pod proxies from `port_forward.sh`. Gives client code one endpoint (default port 9000) instead of 4. PID recorded to `/tmp/qwen_unified_proxy.pid`. |
| `verify_cluster.sh [n] [base_port]` | From the local machine: checks `/v1/models` and sends one real chat completion through each local port. |
| `teardown_cluster.sh <pods...>` | Kills the pods' tmux sessions (`install`, `vllm0..N-1`, `litellm`), the port-forward watchdog loops, any remaining `kubectl port-forward` processes for these pods, and the unified proxy if running. |

Deployment state per pod lives in detached `tmux` sessions (`tmux ls` inside the
pod) so it survives `kubectl exec` disconnects: `install`, `vllm0`..`vllm{N-1}`,
`litellm`. Logs are `~/install.log`, `~/vllm{gpu}.log`, `~/litellm.log` on each pod.

## Known gotchas already fixed in these scripts — do not "fix" them again

These were real failures hit during the first deployment. The scripts already
account for them; if something looks "wrong" below, it's intentional:

1. **`vllm serve` needs the model as the first positional arg.** `vllm serve
   --port 8000 <model>` fails on vllm 0.18.1 — must be `vllm serve <model> --port
   8000 ...`. Already correct in `run_llm.sh`.
2. **`run_llm.sh` serves from `$HOME/Qwen/Qwen3.6-27B` (a real local path), not the
   bare `Qwen/Qwen3.6-27B` repo id.** `install.sh` downloads via `--local-dir`,
   which isn't the default HF cache layout, and direct `huggingface.co` is
   unreachable from the pod — passing the bare repo id makes vllm try (and fail)
   to re-resolve/download it. `--served-model-name` is set so litellm still sees
   the model as `Qwen/Qwen3.6-27B`.
3. **`litellm_config.yaml` backends point at ports 8000-8003** (vLLM's actual
   ports), not 8080+ (that's litellm's own proxy port — pointing backends there
   would make litellm proxy to itself).
4. **`run_litellm.sh` always passes `--host 0.0.0.0`.** Without an explicit host,
   litellm/uvicorn's default host resolution fails with a DNS lookup error right
   after "Application startup complete" and the process exits. This is silent
   unless you check the log — if litellm's tmux session is dead shortly after
   launch, check for this first.
5. **Model downloads can land partial** (observed: a transient proxy error left
   14/15 shards on one pod, no error surfaced other than a traceback deep in
   `install.log`). `deploy_pod.sh` always runs `model_complete_check.sh` before
   *and* after install, and retries install.sh (which resumes/fills gaps) up to 3x
   if shards are missing. Don't assume "install.sh exited" means "model is usable."
6. **`run_llm.sh` passes `--reasoning-parser qwen3 --default-chat-template-kwargs
   '{"enable_thinking": false}'`.** Qwen3-style thinking/reasoning output is on by
   default otherwise (responses come back prefixed with a "Here's a thinking
   process..." preamble). This turns it off server-wide for every request. A
   client can still opt back in per-request via `extra_body={"chat_template_kwargs":
   {"enable_thinking": true}}`. This only takes effect for vLLM instances launched
   *after* this flag was added — already-running instances from a prior deploy need
   their `vllmN` tmux session restarted (`deploy_pod.sh`/`deploy_cluster.sh` does
   this automatically since it always kills and relaunches the `vllmN` sessions).
7. **`kubectl port-forward` has no built-in reconnect logic** — if its tunnel drops
   (API server idle timeout, a local network blip, etc.) it silently stops
   forwarding until restarted, which shows up as "this one pod intermittently
   times out" even though the pod/vLLM/litellm are completely healthy. `port_forward.sh`
   wraps each pod's forward in a retry loop that auto-restarts it on drop, and
   always kills any pre-existing forwards (tracked via the pidfile *and* a
   `pkill -f` sweep) before starting new ones, so repeated `deploy_cluster.sh` runs
   never leave orphaned processes silently fighting over the same local port. If
   you still see timeouts after this, check `tail ~/qwen_cluster_pf_logs/pf_<pod>.log`
   (or the configured `PORT_FORWARD_LOGDIR`) for repeated restarts — that confirms
   it's the tunnel, not the pod, and points at how often it's dropping.

## If something fails

- Re-running `deploy_pod.sh`/`deploy_cluster.sh` is always safe — it's idempotent.
- Check `tmux ls` and the relevant `~/*.log` on the pod for the stage that failed;
  `deploy_pod.sh` already prints the last 30 log lines when a tmux session dies
  before becoming healthy.
- If `verify_cluster.sh` fails on one port but not others, that isolates the
  problem to one pod — check that pod's tmux sessions/logs directly rather than
  re-running the whole cluster deploy.
