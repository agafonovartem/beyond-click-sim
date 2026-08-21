#!/usr/bin/env python3
"""Host control plane for the detached cold-start ablation pod grid.

Modes:
  prepare      — build immutable grid + per-pod manifests locally
  setup        — zip/upload/venv on all pods (no scheduling)
  launch       — upload manifests and start detached per-pod workers
  status       — read-only health/progress snapshot
  retry-failed — reset permanently failed jobs to pending and relaunch workers
  download     — after quiescence, archive selected results into outputs/ablation/
  all          — prepare + setup + launch (status/download remain manual)
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.cold_start_pod_grid_lib import (  # noqa: E402
    ABLATION_METHODS,
    ABLATION_REL,
    DEFAULT_POLL_SECONDS,
    NAMESPACE,
    PIP_DEPS,
    PODS,
    REMOTE_REPO,
    REMOTE_ZIP,
    TASTE_CACHE_FILES,
    ablation_root,
    assert_safe_name,
    build_ablation_grid,
    grid_pods,
    grids_root,
    has_required_artifacts,
    method_requires_taste_cache,
    now_iso,
    parse_pods,
    read_json,
    remote_method_output,
    worker_session_name,
    write_json_atomic,
)


def log(msg: str) -> None:
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


def kubectl_base() -> list[str]:
    return ["kubectl", "-n", NAMESPACE]


def run_cmd(
    args: list[str],
    *,
    check: bool = True,
    capture: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        capture_output=capture,
        text=True,
        timeout=timeout,
    )


def kubectl_exec(
    pod: str,
    remote_cmd: str,
    *,
    check: bool = True,
    timeout: float | None = 120,
) -> str:
    result = run_cmd(
        [*kubectl_base(), "exec", pod, "--", "bash", "-lc", remote_cmd],
        check=check,
        timeout=timeout,
    )
    return (result.stdout or "").strip()


def pod_ready(pod: str) -> bool:
    try:
        out = run_cmd(
            [
                *kubectl_base(),
                "get",
                "pod",
                pod,
                "-o",
                "jsonpath={.status.phase},{.status.containerStatuses[0].ready}",
            ],
            check=False,
            timeout=30,
        )
        return (out.stdout or "").strip() == "Running,true"
    except Exception:
        return False


def probe_pod_llm_endpoint(pod: str) -> tuple[str, str]:
    """Return (endpoint_label, http_code) for the first healthy local LLM endpoint.

    Supports both litellm-fronted setups (8080-8083) and direct vLLM setups
    (8000-8003).
    """
    candidates = [
        ("litellm", 8080),
        ("litellm", 8081),
        ("litellm", 8082),
        ("litellm", 8083),
        ("vllm", 8000),
        ("vllm", 8001),
        ("vllm", 8002),
        ("vllm", 8003),
    ]
    last = ("none", "000")
    for kind, port in candidates:
        code = kubectl_exec(
            pod,
            f"curl -s -o /dev/null -w '%{{http_code}}' --max-time 2 http://127.0.0.1:{port}/v1/models || echo 000",
            check=False,
            timeout=30,
        ).strip()
        last = (f"{kind}:{port}", code)
        if code == "200":
            return last
    return last


def grid_dir(grid_id: str) -> Path:
    return grids_root() / grid_id


def latest_grid_id() -> str:
    root = grids_root()
    if not root.exists():
        raise FileNotFoundError(f"No grids under {root}")
    candidates = sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / "grid.json").exists()],
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError(f"No grid.json under {root}")
    return candidates[-1].name


def resolve_grid_id(grid_id: str | None) -> str:
    return grid_id or latest_grid_id()


def build_zip(zip_path: Path, *, methods: list[str]) -> None:
    root = REPO_ROOT
    exclude_dirs = {
        ".venv",
        "outputs",
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        ".mypy_cache",
        "node_modules",
        # Result archives and scratch, none of it importable by the runners. Left in,
        # tmp_outputs alone added ~900MB to every pod's upload.
        "tmp_outputs",
        "notebooks",
        "logs",
        "_to_delete",
    }
    exclude_names = {".env", ".DS_Store"}
    need_taste = any(method_requires_taste_cache(m) for m in methods)
    taste_paths: list[Path] = []
    if need_taste:
        cache_dir = root / "outputs" / "agent4rec_taste_cache"
        taste_paths = [cache_dir / name for name in TASTE_CACHE_FILES]
        missing = [str(p) for p in taste_paths if not p.is_file() or p.stat().st_size == 0]
        if missing:
            raise FileNotFoundError("Missing taste cache:\n" + "\n".join(missing))
    if zip_path.exists():
        zip_path.unlink()
    log(f"Building zip {zip_path}")
    count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = Path(dirpath).relative_to(root)
            dirnames[:] = [
                d for d in dirnames if d not in exclude_dirs and not d.startswith(".venv")
            ]
            if rel_dir.parts[:1] == ("data",) and rel_dir.parts != ("data",):
                allowed = rel_dir.parts[:3] == ("data", "canonical", "ml-1m") or rel_dir.parts == (
                    "data",
                    "canonical",
                )
                if not allowed and len(rel_dir.parts) >= 2:
                    dirnames[:] = []
                    continue
            for name in filenames:
                if name in exclude_names or name.startswith(".env"):
                    continue
                if name.endswith((".pyc", ".pyo")):
                    continue
                # Every .zip in this repo is an output/artifact archive parked at the
                # root (outputs_backup*.zip are ~2.9GB each); nothing imports one.
                # Excluding `outputs/` as a directory does not catch them, which is how
                # the code archive reached 6.2GB before this check.
                if name.endswith(".zip"):
                    continue
                full = Path(dirpath) / name
                zf.write(full, arcname=full.relative_to(root).as_posix())
                count += 1
        for full in taste_paths:
            zf.write(full, arcname=full.relative_to(root).as_posix())
            count += 1
    log(f"Zip ready: {zip_path} ({count} files, {zip_path.stat().st_size / 1e6:.1f} MB)")


def upload_and_setup_pod(pod: str, zip_path: Path, *, methods: list[str]) -> None:
    log(f"{pod}: uploading zip")
    run_cmd([*kubectl_base(), "cp", str(zip_path), f"{pod}:{REMOTE_ZIP}"], check=True, capture=False)
    deps = " ".join(shlex.quote(d) for d in PIP_DEPS)
    methods_repr = ", ".join(repr(m) for m in methods)
    setup = f"""
set -euo pipefail
rm -rf {REMOTE_REPO}
mkdir -p {REMOTE_REPO}
cd {REMOTE_REPO}
unzip -q -o {REMOTE_ZIP}
/opt/conda/bin/python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q {deps}
# Install the package itself editable so `beyond_click_sim` resolves as an installed
# distribution rather than relying solely on PYTHONPATH. --no-deps keeps the pinned
# set above authoritative: pyproject pulls torch/optuna, which the runners never
# import for these listwise methods and which would add gigabytes per pod.
.venv/bin/pip install -q -e . --no-deps
mkdir -p {ABLATION_REL.as_posix()}
export PYTHONPATH="{REMOTE_REPO}/src:{REMOTE_REPO}"
.venv/bin/python - <<'PY'
from pathlib import Path
from runners.in_distribution.cold_start.methods import METHOD_RUNNERS
from runners.in_distribution.cold_start.task_builders import TASK_BUILDERS
methods = [{methods_repr}]
for m in methods:
    assert m in METHOD_RUNNERS, m
p = Path("data/canonical/ml-1m/v1/interactions.parquet")
assert p.exists(), p
# spot-check one item_stats task exists
assert any("item_stats" in name for name in TASK_BUILDERS)
print("SETUP_OK")
PY
"""
    log(f"{pod}: unzip + venv + pip install")
    out = kubectl_exec(pod, setup, check=True, timeout=1800)
    if "SETUP_OK" not in out:
        raise RuntimeError(f"{pod}: setup failed: {out[-500:]}")
    log(f"{pod}: setup complete")


def cmd_prepare(args: argparse.Namespace) -> int:
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else list(ABLATION_METHODS)
    pods = parse_pods(args.pods, default=PODS)
    grid = build_ablation_grid(methods, pods=pods, grid_id=args.grid_id)
    grid_id = grid["grid_id"]
    assert_safe_name(grid_id, label="grid_id")
    out = grid_dir(grid_id)
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "grid.json", grid)
    pods_dir = out / "pods"
    pods_dir.mkdir(parents=True, exist_ok=True)
    for pod, jobs in grid["pods_jobs"].items():
        pod_manifest = {
            "grid_id": grid_id,
            "pod": pod,
            "created_at": now_iso(),
            "max_tasks_per_pod": grid["max_tasks_per_pod"],
            "max_attempts": grid["max_attempts"],
            "ablation_rel": grid["ablation_rel"],
            "jobs": jobs,
        }
        write_json_atomic(pods_dir / f"{pod}.json", pod_manifest)
        log(f"Wrote {pod}: {len(jobs)} jobs")
    sizes = [len(v) for v in grid["pods_jobs"].values()]
    log(
        f"Prepared grid {grid_id}: methods={len(methods)} tasks={len(grid['tasks'])} "
        f"jobs={grid['job_count']} pods={pods} pod_sizes={sorted(sizes)}"
    )
    print(grid_id)
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    grid_id = resolve_grid_id(args.grid_id)
    grid = read_json(grid_dir(grid_id) / "grid.json")
    pods = grid_pods(grid)
    methods = list(grid["methods"])
    zip_path = grid_dir(grid_id) / "beyond-click-sim-code.zip"
    build_zip(zip_path, methods=methods)
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=len(pods)) as ex:
        futs = {ex.submit(upload_and_setup_pod, pod, zip_path, methods=methods): pod for pod in pods}
        for fut in as_completed(futs):
            pod = futs[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{pod}: {exc}")
                log(f"ERROR setup {pod}: {exc}")
    if errors:
        raise RuntimeError("Setup failed:\n" + "\n".join(errors))
    log(f"Setup complete for grid {grid_id}")
    return 0


def upload_manifests(grid_id: str, pods: list[str]) -> None:
    local_pods = grid_dir(grid_id) / "pods"
    for pod in pods:
        local = local_pods / f"{pod}.json"
        remote_dir = f"{REMOTE_REPO}/outputs/ablation/_grids/{grid_id}/pods"
        kubectl_exec(pod, f"mkdir -p {shlex.quote(remote_dir)}", check=True, timeout=60)
        run_cmd(
            [*kubectl_base(), "cp", str(local), f"{pod}:{remote_dir}/{pod}.json"],
            check=True,
            capture=False,
        )


def launch_worker_on_pod(
    pod: str, grid_id: str, *, poll_seconds: float, max_slots: int | None = None
) -> None:
    session = worker_session_name(grid_id, pod)
    remote_manifest = f"{REMOTE_REPO}/outputs/ablation/_grids/{grid_id}/pods/{pod}.json"
    remote_state = f"{REMOTE_REPO}/outputs/ablation/_grids/{grid_id}/pods/{pod}.state.json"
    remote_log = f"{REMOTE_REPO}/outputs/ablation/_grids/{grid_id}/pods/{pod}.worker.log"
    kubectl_exec(
        pod,
        f"tmux kill-session -t {shlex.quote(session)} 2>/dev/null || true",
        check=False,
        timeout=30,
    )
    inner = (
        f"cd {REMOTE_REPO} && "
        f"source .venv/bin/activate && "
        f'export PYTHONPATH="{REMOTE_REPO}/src:{REMOTE_REPO}" && '
        f"mkdir -p outputs/ablation/_grids/{grid_id}/pods && "
        f"python scripts/cold_start_pod_worker.py "
        f"--manifest {shlex.quote(remote_manifest)} "
        f"--state {shlex.quote(remote_state)} "
        f"--poll-seconds {poll_seconds} "
        # Concurrency has to track the pod's serving topology, not the pod count. A
        # litellm-fronted pod has four engines to absorb a listwise job's 240 worker
        # threads; a tensor-parallel pod has one, so the same slot count buries it.
        + (f"--max-slots {int(max_slots)} " if max_slots is not None else "")
        + (
            f"> {shlex.quote(remote_log)} 2>&1"
        )
    )
    remote = (
        f"tmux new-session -d -s {shlex.quote(session)} "
        f"{shlex.quote('bash -lc ' + shlex.quote(inner))}"
    )
    kubectl_exec(pod, remote, check=True, timeout=60)
    # Confirm session exists
    alive = kubectl_exec(
        pod,
        f"tmux has-session -t {shlex.quote(session)} 2>/dev/null && echo YES || echo NO",
        check=False,
        timeout=30,
    )
    if "YES" not in alive:
        raise RuntimeError(f"{pod}: worker tmux session failed to start")
    log(f"{pod}: launched worker session {session}")


def cmd_launch(args: argparse.Namespace) -> int:
    grid_id = resolve_grid_id(args.grid_id)
    assert_safe_name(grid_id, label="grid_id")
    grid = read_json(grid_dir(grid_id) / "grid.json")
    pods = grid_pods(grid)
    for pod in pods:
        if not pod_ready(pod):
            raise RuntimeError(f"{pod} is not ready")
    upload_manifests(grid_id, pods)
    for pod in pods:
        # Ensure some local LLM endpoint responds before launch.
        endpoint, code = probe_pod_llm_endpoint(pod)
        if code != "200":
            raise RuntimeError(
                f"{pod}: no healthy LLM endpoint (last={endpoint} http={code!r})"
            )
        launch_worker_on_pod(
            pod, grid_id, poll_seconds=args.poll_seconds, max_slots=args.max_slots
        )
    # Brief settle + status snapshot
    time.sleep(3)
    return cmd_status(argparse.Namespace(grid_id=grid_id))


def fetch_pod_state(pod: str, grid_id: str) -> dict | None:
    remote_state = f"{REMOTE_REPO}/outputs/ablation/_grids/{grid_id}/pods/{pod}.state.json"
    out = kubectl_exec(
        pod,
        f"if [ -f {shlex.quote(remote_state)} ]; then cat {shlex.quote(remote_state)}; else echo MISSING; fi",
        check=False,
        timeout=60,
    )
    if not out or out.strip() == "MISSING":
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"error": "invalid_json", "raw": out[:200]}


def write_pod_state(pod: str, grid_id: str, state: dict) -> None:
    """Persist worker state locally for provenance and upload to the pod."""
    local = grid_dir(grid_id) / "pods" / f"{pod}.state.json"
    write_json_atomic(local, state)
    remote_dir = f"{REMOTE_REPO}/outputs/ablation/_grids/{grid_id}/pods"
    kubectl_exec(pod, f"mkdir -p {shlex.quote(remote_dir)}", check=True, timeout=60)
    run_cmd(
        [*kubectl_base(), "cp", str(local), f"{pod}:{remote_dir}/{pod}.state.json"],
        check=True,
        capture=False,
    )


def reset_failed_jobs(state: dict) -> list[str]:
    """Reset permanently failed jobs to pending with a fresh attempt budget.

    Returns the list of reset job_ids. Completed/running/pending jobs are untouched.
    """
    jobs = state.get("jobs") or {}
    reset_ids: list[str] = []
    for job_id, raw in jobs.items():
        if not isinstance(raw, dict):
            continue
        if raw.get("status") != "failed":
            continue
        raw["status"] = "pending"
        raw["attempts"] = 0
        raw["run_token"] = None
        raw["result_dir"] = None
        raw["tmux_session"] = None
        raw["started_at"] = None
        raw["finished_at"] = None
        raw["error"] = "reset_by_retry_failed"
        raw["exit_code"] = None
        raw["log_path"] = None
        reset_ids.append(job_id)
    if reset_ids:
        state["phase"] = "running"
        state["updated_at"] = now_iso()
        # Refresh summary counts if present.
        summary = {
            "pending": 0,
            "running": 0,
            "retry_wait": 0,
            "completed": 0,
            "failed": 0,
        }
        for raw in jobs.values():
            status = str(raw.get("status", "pending"))
            if status in summary:
                summary[status] += 1
        state["summary"] = summary
        state["jobs"] = jobs
    return reset_ids


def cmd_retry_failed(args: argparse.Namespace) -> int:
    """Reset failed jobs to pending and relaunch workers only on affected pods."""
    grid_id = resolve_grid_id(args.grid_id)
    grid = read_json(grid_dir(grid_id) / "grid.json")
    pods = grid_pods(grid)
    affected: list[str] = []
    total_reset = 0
    for pod in pods:
        if not pod_ready(pod):
            raise RuntimeError(f"{pod} is not ready")
        state = fetch_pod_state(pod, grid_id)
        if not state or "jobs" not in state:
            log(f"{pod}: no state; skipping")
            continue
        # Only touch pods that actually have something to retry. Killing a worker we
        # then decline to relaunch strands the pod: its in-flight job sessions keep
        # running but nothing schedules the rest.
        if not any(
            isinstance(raw, dict) and raw.get("status") == "failed"
            for raw in state["jobs"].values()
        ):
            continue
        # Stop the worker BEFORE rewriting state, then re-read. A live worker holds the
        # job table in memory and rewrites the state file every poll, so a reset
        # uploaded underneath it is silently clobbered and the job stays failed with its
        # old attempt count. Job sessions (csj_*) are independent and keep running; the
        # relaunched worker adopts them from state. Every pod killed here is appended to
        # `affected`, so every pod killed is relaunched below.
        kubectl_exec(
            pod,
            f"tmux kill-session -t {shlex.quote(worker_session_name(grid_id, pod))} "
            "2>/dev/null || true",
            check=False,
            timeout=30,
        )
        state = fetch_pod_state(pod, grid_id) or state
        reset_ids = reset_failed_jobs(state)
        if not reset_ids:
            # Worker changed the picture between the two reads; it is already dead, so
            # it still must be relaunched.
            affected.append(pod)
            continue
        write_pod_state(pod, grid_id, state)
        log(f"{pod}: reset {len(reset_ids)} failed jobs -> pending: {reset_ids}")
        affected.append(pod)
        total_reset += len(reset_ids)
    if not affected:
        log(f"No failed jobs to retry for grid {grid_id}")
        return cmd_status(argparse.Namespace(grid_id=grid_id))

    # Ensure manifests/code worker script are present, then relaunch only affected pods.
    upload_manifests(grid_id, affected)
    for pod in affected:
        endpoint, code = probe_pod_llm_endpoint(pod)
        if code != "200":
            raise RuntimeError(
                f"{pod}: no healthy LLM endpoint (last={endpoint} http={code!r})"
            )
        launch_worker_on_pod(
            pod, grid_id, poll_seconds=args.poll_seconds, max_slots=args.max_slots
        )
    log(f"retry-failed: reset {total_reset} jobs on pods {affected}")
    time.sleep(3)
    return cmd_status(argparse.Namespace(grid_id=grid_id))


def cmd_status(args: argparse.Namespace) -> int:
    grid_id = resolve_grid_id(args.grid_id)
    grid = read_json(grid_dir(grid_id) / "grid.json")
    aggregate = {
        "pending": 0,
        "running": 0,
        "retry_wait": 0,
        "completed": 0,
        "failed": 0,
    }
    per_method = {m: dict(aggregate) for m in grid["methods"]}
    pods = grid_pods(grid)
    print(f"grid_id={grid_id} expected_jobs={grid['job_count']} pods={len(pods)}")
    for pod in pods:
        ready = pod_ready(pod)
        session = worker_session_name(grid_id, pod)
        worker_alive = False
        if ready:
            alive = kubectl_exec(
                pod,
                f"tmux has-session -t {shlex.quote(session)} 2>/dev/null && echo YES || echo NO",
                check=False,
                timeout=30,
            )
            worker_alive = "YES" in alive
            endpoint, code = probe_pod_llm_endpoint(pod)
            llm = f"{endpoint}({code})"
        else:
            llm = "pod_not_ready"
        state = fetch_pod_state(pod, grid_id) if ready else None
        summary = (state or {}).get("summary") or {}
        phase = (state or {}).get("phase", "unknown")
        updated = (state or {}).get("updated_at", "")
        for key in aggregate:
            aggregate[key] += int(summary.get(key, 0))
        # Per-method from job states if present
        jobs_state = (state or {}).get("jobs") or {}
        expected_jobs = {
            j["job_id"]: j
            for j in grid["pods_jobs"][pod]
        }
        for job_id, st in jobs_state.items():
            meta = expected_jobs.get(job_id)
            if not meta:
                continue
            status = st.get("status", "pending")
            if status in per_method[meta["method"]]:
                per_method[meta["method"]][status] += 1
        print(
            f"{pod}: ready={ready} worker={worker_alive} llm={llm} "
            f"phase={phase} summary={summary} updated={updated}"
        )
    print(f"aggregate={aggregate}")
    for method, counts in per_method.items():
        print(f"method {method}: {counts}")
    done = aggregate["completed"]
    print(f"progress={done}/{grid['job_count']}")
    return 0


def collect_inventory(grid_id: str, *, allow_failed: bool = False) -> dict[str, list[dict]]:
    """pod -> list of {method, task, result_dir, job_id}."""
    grid = read_json(grid_dir(grid_id) / "grid.json")
    pods = grid_pods(grid)
    inventory: dict[str, list[dict]] = {pod: [] for pod in pods}
    missing: list[str] = []
    failed: list[str] = []
    for pod in pods:
        state = fetch_pod_state(pod, grid_id)
        if not state:
            raise RuntimeError(f"{pod}: missing state")
        jobs_meta = {j["job_id"]: j for j in grid["pods_jobs"][pod]}
        for job_id, st in state.get("jobs", {}).items():
            meta = jobs_meta.get(job_id)
            if not meta:
                continue
            status = st.get("status")
            if status == "failed":
                failed.append(f"{pod}:{job_id}:{st.get('error')}")
                continue
            if status != "completed":
                missing.append(f"{pod}:{job_id}:{status}")
                continue
            result_dir = st.get("result_dir")
            if not result_dir:
                missing.append(f"{pod}:{job_id}:no_result_dir")
                continue
            inventory[pod].append(
                {
                    "job_id": job_id,
                    "method": meta["method"],
                    "task": meta["task"],
                    "result_dir": result_dir,
                    "required_artifacts": meta["required_artifacts"],
                }
            )
    if failed and not allow_failed:
        raise RuntimeError("Cannot download with failures:\n" + "\n".join(failed[:20]))
    if missing:
        raise RuntimeError("Cannot download; incomplete jobs:\n" + "\n".join(missing[:20]))
    total = sum(len(v) for v in inventory.values())
    if allow_failed:
        if total + len(failed) != grid["job_count"]:
            raise RuntimeError(
                f"inventory size {total} + failed {len(failed)} != expected {grid['job_count']}"
            )
        if failed:
            log(f"Allowing download of {total} completed jobs; skipping {len(failed)} failures")
    elif total != grid["job_count"]:
        raise RuntimeError(f"inventory size {total} != expected {grid['job_count']}")
    return inventory


def quiescence_ok(grid_id: str, *, allow_failed: bool = False) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    grid = read_json(grid_dir(grid_id) / "grid.json")
    pods = grid_pods(grid)
    expected = grid["job_count"]
    completed = 0
    failed_total = 0
    for pod in pods:
        if not pod_ready(pod):
            reasons.append(f"{pod}: not ready")
            continue
        session = worker_session_name(grid_id, pod)
        # Worker may have exited after terminal; that is OK if state is terminal.
        state = fetch_pod_state(pod, grid_id)
        if not state:
            reasons.append(f"{pod}: missing state")
            continue
        summary = state.get("summary") or {}
        blocked_keys = ("pending", "running", "retry_wait")
        if not allow_failed:
            blocked_keys = ("pending", "running", "retry_wait", "failed")
        for key in blocked_keys:
            if int(summary.get(key, 0)) != 0:
                reasons.append(f"{pod}: {key}={summary.get(key)}")
        completed += int(summary.get("completed", 0))
        failed_total += int(summary.get("failed", 0))
        # No leftover job tmux sessions with csj_ prefix from this worker's jobs.
        leftover = kubectl_exec(
            pod,
            "tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^csj_' || true",
            check=False,
            timeout=30,
        )
        if leftover.strip():
            reasons.append(f"{pod}: leftover job sessions {leftover.splitlines()[:5]}")
        runners = kubectl_exec(
            pod,
            "ps -eo args= 2>/dev/null | grep -E '[p]ython.*runners\\.in_distribution\\.cold_start\\.run' || true",
            check=False,
            timeout=30,
        )
        if runners.strip():
            reasons.append(f"{pod}: leftover runners")
        # Worker session may still be alive briefly; ignore if terminal phase.
        if state.get("phase") != "terminal":
            alive = kubectl_exec(
                pod,
                f"tmux has-session -t {shlex.quote(session)} 2>/dev/null && echo YES || echo NO",
                check=False,
                timeout=30,
            )
            if "YES" in alive and int(summary.get("running", 0)) == 0:
                # worker loop still sleeping after terminal — acceptable
                pass
            if state.get("phase") not in {"terminal"}:
                reasons.append(f"{pod}: phase={state.get('phase')}")
    if allow_failed:
        if completed + failed_total != expected:
            reasons.append(f"completed+failed={completed}+{failed_total}/{expected}")
        if completed == 0:
            reasons.append("no completed jobs to download")
    elif completed != expected:
        reasons.append(f"completed={completed}/{expected}")
    return (not reasons, reasons)


def fingerprint_inventory(grid_id: str, inventory: dict[str, list[dict]]) -> dict[str, str]:
    fps: dict[str, str] = {}
    for pod, items in inventory.items():
        for item in items:
            method = item["method"]
            result_dir = item["result_dir"]
            artifacts = " ".join(shlex.quote(a) for a in item["required_artifacts"])
            cmd = f"""
set +e
cd {remote_method_output(method)}/{shlex.quote(result_dir)} 2>/dev/null || {{ echo MISSING; exit 0; }}
for f in {artifacts}; do
  if [ ! -f "$f" ]; then echo MISSING; exit 0; fi
  stat -c '%n %s %Y' "$f" 2>/dev/null || stat -f '%N %z %m' "$f"
done
"""
            fps[f"{pod}:{item['job_id']}"] = kubectl_exec(pod, cmd, check=False, timeout=60)
    return fps


def download_pod_archive(pod: str, items: list[dict], *, staging: Path) -> None:
    if not items:
        return
    # Build archive with method/result_dir paths from ablation root.
    remote_zip = f"/tmp/{pod}_ablation_selected.zip"
    # Create a staging dir on pod with symlinks/copies of selected leaves.
    remote_stage = f"/tmp/{pod}_ablation_stage"
    cmds = [
        f"rm -rf {shlex.quote(remote_stage)} {shlex.quote(remote_zip)}",
        f"mkdir -p {shlex.quote(remote_stage)}",
    ]
    for item in items:
        method = item["method"]
        result_dir = item["result_dir"]
        assert_safe_name(method, label="method")
        if not result_dir or any(c in result_dir for c in " \t\n\"'`$;&|<>(){}[]\\/"):
            raise ValueError(f"unsafe result_dir: {result_dir!r}")
        src = f"{remote_method_output(method)}/{result_dir}"
        dest_parent = f"{remote_stage}/{method}"
        cmds.append(f"mkdir -p {shlex.quote(dest_parent)}")
        cmds.append(
            f"cp -a {shlex.quote(src)} {shlex.quote(dest_parent + '/' + result_dir)}"
        )
    cmds.append(
        f"cd {shlex.quote(remote_stage)} && zip -qr {shlex.quote(remote_zip)} ."
    )
    kubectl_exec(pod, " && ".join(cmds), check=True, timeout=1800)
    local_zip = staging / f"{pod}_ablation_selected.zip"
    if local_zip.exists():
        local_zip.unlink()
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            run_cmd(
                [*kubectl_base(), "cp", f"{pod}:{remote_zip}", str(local_zip)],
                check=True,
                capture=False,
            )
            if local_zip.is_file() and local_zip.stat().st_size > 0:
                break
            raise RuntimeError("empty archive")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log(f"{pod}: kubectl cp attempt {attempt}/3 failed: {exc}")
            time.sleep(2 * attempt)
    else:
        raise RuntimeError(f"{pod}: kubectl cp failed: {last_err}")
    with zipfile.ZipFile(local_zip, "r") as zf:
        # Validate members before extract
        for item in items:
            prefix = f"{item['method']}/{item['result_dir']}/"
            for artifact in item["required_artifacts"]:
                member = prefix + artifact
                if member not in zf.namelist() and member.replace("\\", "/") not in {
                    n.replace("\\", "/") for n in zf.namelist()
                }:
                    # zip may store without trailing consistency; check endswith
                    names = {n.replace("\\", "/") for n in zf.namelist()}
                    if member not in names:
                        raise RuntimeError(f"archive missing {member}")
        dest_root = ablation_root()
        dest_root.mkdir(parents=True, exist_ok=True)
        # Extract carefully: refuse overwrite of existing different dirs
        for item in items:
            target = dest_root / item["method"] / item["result_dir"]
            if target.exists():
                if has_required_artifacts(target, item["required_artifacts"]):
                    log(f"skip existing healthy {target}")
                    continue
                raise FileExistsError(f"Refusing to overwrite incomplete/existing {target}")
        zf.extractall(dest_root)
    log(f"{pod}: downloaded {len(items)} result dirs")


def validate_local_ablation(
    grid_id: str,
    inventory: dict[str, list[dict]],
    *,
    allow_failed: bool = False,
) -> None:
    grid = read_json(grid_dir(grid_id) / "grid.json")
    expected_pairs = {(j["method"], j["task"]) for j in grid["jobs"]}
    inventory_pairs = {
        (item["method"], item["task"]) for items in inventory.values() for item in items
    }
    found_pairs: set[tuple[str, str]] = set()
    for items in inventory.values():
        for item in items:
            path = ablation_root() / item["method"] / item["result_dir"]
            if not has_required_artifacts(path, item["required_artifacts"]):
                raise RuntimeError(f"Missing artifacts under {path}")
            manifest = read_json(path / "manifest.json")
            metrics = read_json(path / "metrics_ranking.json")
            # Manifest schema varies; accept method/task at top-level or nested.
            m_method = manifest.get("method") or (manifest.get("scorer") or {}).get("method_name")
            raw_task = manifest.get("task")
            if isinstance(raw_task, dict):
                m_task = raw_task.get("name")
            elif isinstance(raw_task, str):
                m_task = raw_task
            else:
                m_task = (manifest.get("task_manifest") or {}).get("name")
            if m_method and m_method != item["method"]:
                raise RuntimeError(f"method mismatch in {path}: {m_method}")
            if m_task and m_task != item["task"]:
                raise RuntimeError(f"task mismatch in {path}: {m_task}")
            if not isinstance(metrics, dict) or not metrics:
                raise RuntimeError(f"empty metrics in {path}")
            found_pairs.add((item["method"], item["task"]))
    if found_pairs != inventory_pairs:
        missing = inventory_pairs - found_pairs
        extra = found_pairs - inventory_pairs
        raise RuntimeError(f"downloaded pair mismatch missing={len(missing)} extra={len(extra)}")
    if allow_failed:
        skipped = expected_pairs - found_pairs
        log(
            f"Local validation OK (partial): {len(found_pairs)}/{len(expected_pairs)} "
            f"(skipped_failed={len(skipped)})"
        )
        return
    if found_pairs != expected_pairs:
        missing = expected_pairs - found_pairs
        extra = found_pairs - expected_pairs
        raise RuntimeError(f"pair mismatch missing={len(missing)} extra={len(extra)}")
    log(f"Local validation OK: {len(found_pairs)}/{len(expected_pairs)}")


def cmd_download(args: argparse.Namespace) -> int:
    grid_id = resolve_grid_id(args.grid_id)
    allow_failed = bool(getattr(args, "allow_failed", False))
    log(
        f"Waiting for quiescence on grid {grid_id}"
        + (" (allowing failed jobs)" if allow_failed else "")
    )
    prev_fps: dict[str, str] | None = None
    inventory: dict[str, list[dict]] | None = None
    while True:
        ok, reasons = quiescence_ok(grid_id, allow_failed=allow_failed)
        if not ok:
            log(f"quiescence not ready: {reasons[:8]}")
            prev_fps = None
            time.sleep(args.poll_seconds)
            continue
        inventory = collect_inventory(grid_id, allow_failed=allow_failed)
        fps = fingerprint_inventory(grid_id, inventory)
        if any(v.startswith("MISSING") or v.startswith("ERROR") for v in fps.values()):
            log("fingerprints incomplete; retrying")
            prev_fps = None
            time.sleep(args.poll_seconds)
            continue
        if prev_fps is not None and prev_fps == fps:
            log("Quiescence barrier reached")
            break
        log("quiescence first stable poll; confirming once more")
        prev_fps = fps
        time.sleep(args.poll_seconds)

    assert inventory is not None
    staging = grid_dir(grid_id) / "download_staging"
    staging.mkdir(parents=True, exist_ok=True)
    write_json_atomic(grid_dir(grid_id) / "download_inventory.json", inventory)
    for pod, items in inventory.items():
        download_pod_archive(pod, items, staging=staging)
    validate_local_ablation(grid_id, inventory, allow_failed=allow_failed)
    log(f"Download complete for grid {grid_id}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    rc = cmd_prepare(args)
    if rc != 0:
        return rc
    # prepare printed grid_id; re-read latest if not provided
    if not args.grid_id:
        args.grid_id = latest_grid_id()
    rc = cmd_setup(args)
    if rc != 0:
        return rc
    return cmd_launch(args)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "mode",
        choices=["prepare", "setup", "launch", "status", "retry-failed", "download", "all"],
    )
    p.add_argument("--grid-id", default=None)
    p.add_argument(
        "--methods",
        default=None,
        help="Comma-separated methods (prepare/all). Default: fixed ablation set.",
    )
    p.add_argument(
        "--pods",
        default=None,
        help=(
            "Comma-separated pod names (prepare/all). Persisted into grid.json; "
            "later modes read pods from that grid. Default: agent-exp-policy-ranking-0..7."
        ),
    )
    p.add_argument(
        "--allow-failed",
        action="store_true",
        help="download: pull completed results even if some jobs permanently failed",
    )
    p.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    p.add_argument(
        "--max-slots",
        type=int,
        default=None,
        help=(
            "launch/retry-failed: concurrent jobs per pod. Default None keeps the "
            "worker's own MAX_TASKS_PER_POD. Lower it for tensor-parallel pods "
            "serving a single engine."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ablation_root().mkdir(parents=True, exist_ok=True)
    grids_root().mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        return cmd_prepare(args)
    if args.mode == "setup":
        return cmd_setup(args)
    if args.mode == "launch":
        return cmd_launch(args)
    if args.mode == "status":
        return cmd_status(args)
    if args.mode == "retry-failed":
        return cmd_retry_failed(args)
    if args.mode == "download":
        return cmd_download(args)
    if args.mode == "all":
        return cmd_all(args)
    raise ValueError(args.mode)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        log(f"ERROR: {exc}")
        sys.exit(1)
