# Cold-Start Pod Queue Runbook

This runbook documents the repeatable workflow for running the full MovieLens
cold-start batch on `agent-exp-0..7` with one task per pod, automatic queueing,
and immediate result download to local disk.

Primary automation script:

- `scripts/run_cold_start_pod_queue.py`

The script already includes:

- repo zip build and upload to all 8 pods;
- pod setup with `python -m venv` + `pip install` (no `uv` on pods);
- litellm port auto-discovery per pod (fallback to vLLM ports);
- one-task-per-pod queue scheduling;
- immediate result zipping, `kubectl cp` download, and local unzip;
- resume/retry behavior if a pod reloads or a session exits.

---

## Fixed Experiment Scope (current defaults)

Inside `run_cold_start_pod_queue.py`, defaults are pinned to:

- namespace: `a-ashabokov`
- pods: `agent-exp-0..7`
- method: `agent4rec_yes_no_vllm_qwen36_35b_a3b_item_stats_traits_taste_gpt4o_mini_full`
  (batched Agent4Rec; requires `outputs/agent4rec_taste_cache/ml-1m_v1_seed{0,1,2}_*.jsonl`
  in the upload zip)
- output root: `outputs/experimental/cold_start`
- tasks: full 45-task **item_stats** set (required by `*_item_stats_*` Agent4Rec methods):
  - names: `ml-1m_item_stats_cold_start_k{k}_cap20_eval_users1000_cg5_m{m}_seed{s}`
  - `k in {1,3,5}`
  - `m in {1,2,3,9,19}`
  - `seed in {0,1,2}`
  - plain `ml-1m_cold_start_*` tasks lack `item_rating_mean` and will crash these methods

If you need a different task list, method, namespace, or pod names, edit the
constants near the top of `run_cold_start_pod_queue.py`.

---

## Prerequisites

From repo root:

```bash
kubectl config current-context
kubectl get pods -n a-ashabokov | grep '^agent-exp-[0-7] '
```

Pods should be `Running`. The script assumes vLLM/litellm are already deployed on
pods (for deployment commands see `scripts/README.md`).

---

## Standard Run

### 1) Full setup + queue run

```bash
python3 scripts/run_cold_start_pod_queue.py \
  --zip-path outputs/experimental/cold_start/_queue/beyond-click-sim-code.zip
```

This does setup and then starts the task queue.

### 2) Setup only (if you want to stage pods first)

```bash
python3 scripts/run_cold_start_pod_queue.py \
  --setup-only \
  --zip-path outputs/experimental/cold_start/_queue/beyond-click-sim-code.zip
```

### 3) Queue only (reuse already prepared pods)

```bash
python3 scripts/run_cold_start_pod_queue.py \
  --queue-only \
  --zip-path outputs/experimental/cold_start/_queue/beyond-click-sim-code.zip \
  --poll-seconds 45
```

---

## Monitoring During Run

### Queue state

```bash
python3 - <<'PY'
import json
from pathlib import Path
from collections import Counter
st=json.loads(Path("outputs/experimental/cold_start/_queue/state.json").read_text())
print(Counter(v["status"] for v in st["tasks"].values()))
print("retries", len(st.get("retries", [])))
PY
```

### Queue log tail

```bash
tail -n 40 outputs/experimental/cold_start/_queue/queue.log
```

### Pod-side progress for currently running tasks

```bash
for p in 0 1 2 3 4 5 6 7; do
  kubectl -n a-ashabokov exec agent-exp-$p -- bash -lc \
    'tmux ls 2>/dev/null | grep "^cs_" || true; ls -t ~/beyond-click-sim/outputs/experimental/cold_start/*.log 2>/dev/null | head -1'
done
```

---

## Final Verification (must be 45/45)

```bash
python3 - <<'PY'
from pathlib import Path
import json
from collections import Counter

METHOD = "llm_yes_no_itemwise_vllm_qwen36_27b_full"
root = Path("outputs/experimental/cold_start")
state = json.loads((root / "_queue" / "state.json").read_text())
print("queue_state", Counter(v["status"] for v in state["tasks"].values()))

complete = [
    d for d in root.iterdir()
    if d.is_dir()
    and d.name != "_queue"
    and (d / "metrics.json").exists()
    and (d / "metrics_ranking.json").exists()
]
print("complete_dirs", len(complete))
PY
```

Expected:

- `queue_state Counter({'done': 45})`
- `complete_dirs 45`

---

## Resume / Recovery

If the local terminal was closed or the process interrupted:

```bash
python3 scripts/run_cold_start_pod_queue.py \
  --queue-only \
  --zip-path outputs/experimental/cold_start/_queue/beyond-click-sim-code.zip
```

The script resumes from `outputs/experimental/cold_start/_queue/state.json` and
also re-detects already completed local runs.

If a pod reloads, the script re-queues unfinished tasks and can re-setup pods if
`.venv` or dataset files are missing.

---

## Operational Notes Learned

1. Do not assume litellm is always on `8080`; discover per pod before each start.
2. Pod task logs can show `100%` before the orchestrator polls; allow one more
   poll cycle for download/final status update.
3. Result detection in remote shell globs must avoid quoting task/method inside
   glob patterns (`*_<task>_<method>`); quoted globs can block completion pickup.
4. Keep run artifacts under `outputs/experimental/cold_start/_queue/`:
   - `state.json` (source of truth for scheduling)
   - `queue.log` (timeline)
   - `beyond-click-sim-code.zip` (reusable upload payload)

---

## Related Files

- `scripts/run_cold_start_pod_queue.py` (automation)
- `scripts/README.md` (cluster deploy/verify/teardown)
- `scripts/AGENTS.md` (agent-oriented deploy reference)
