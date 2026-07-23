#!/usr/bin/env python3
"""Queue cold-start LLM tasks across agent-exp Kubernetes pods.

Upload/setup once per pod, discover litellm port, run one task per free pod in
tmux, and download/unzip results as soon as metrics appear.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


NAMESPACE = "a-ashabokov"
PODS = [f"agent-exp-{i}" for i in range(8)]
METHOD = "agent4rec_yes_no_vllm_qwen36_35b_a3b_item_stats_traits_taste_gpt4o_mini_full"
REMOTE_REPO = "/home/jovyan/beyond-click-sim"
REMOTE_ZIP = "/home/jovyan/beyond-click-sim-code.zip"
REMOTE_OUTPUT = f"{REMOTE_REPO}/outputs/experimental/cold_start"
# Precomputed Agent4Rec taste profiles required on-pod for traits+taste methods.
TASTE_CACHE_FILES = [
    "ml-1m_v1_seed0_gpt-4o-mini_agent4rec_modify_v1.jsonl",
    "ml-1m_v1_seed1_gpt-4o-mini_agent4rec_modify_v1.jsonl",
    "ml-1m_v1_seed2_gpt-4o-mini_agent4rec_modify_v1.jsonl",
]

TASKS = [
    "ml-1m_item_stats_cold_start_k1_cap20_eval_users1000_cg5_m1_seed0",
    "ml-1m_item_stats_cold_start_k1_cap20_eval_users1000_cg5_m2_seed0",
    "ml-1m_item_stats_cold_start_k1_cap20_eval_users1000_cg5_m3_seed0",
    "ml-1m_item_stats_cold_start_k1_cap20_eval_users1000_cg5_m9_seed0",
    "ml-1m_item_stats_cold_start_k1_cap20_eval_users1000_cg5_m19_seed0",
    "ml-1m_item_stats_cold_start_k1_cap20_eval_users1000_cg5_m1_seed1",
    "ml-1m_item_stats_cold_start_k1_cap20_eval_users1000_cg5_m2_seed1",
    "ml-1m_item_stats_cold_start_k1_cap20_eval_users1000_cg5_m3_seed1",
    "ml-1m_item_stats_cold_start_k1_cap20_eval_users1000_cg5_m9_seed1",
    "ml-1m_item_stats_cold_start_k1_cap20_eval_users1000_cg5_m19_seed1",
    "ml-1m_item_stats_cold_start_k1_cap20_eval_users1000_cg5_m1_seed2",
    "ml-1m_item_stats_cold_start_k1_cap20_eval_users1000_cg5_m2_seed2",
    "ml-1m_item_stats_cold_start_k1_cap20_eval_users1000_cg5_m3_seed2",
    "ml-1m_item_stats_cold_start_k1_cap20_eval_users1000_cg5_m9_seed2",
    "ml-1m_item_stats_cold_start_k1_cap20_eval_users1000_cg5_m19_seed2",
    "ml-1m_item_stats_cold_start_k3_cap20_eval_users1000_cg5_m1_seed0",
    "ml-1m_item_stats_cold_start_k3_cap20_eval_users1000_cg5_m2_seed0",
    "ml-1m_item_stats_cold_start_k3_cap20_eval_users1000_cg5_m3_seed0",
    "ml-1m_item_stats_cold_start_k3_cap20_eval_users1000_cg5_m9_seed0",
    "ml-1m_item_stats_cold_start_k3_cap20_eval_users1000_cg5_m19_seed0",
    "ml-1m_item_stats_cold_start_k3_cap20_eval_users1000_cg5_m1_seed1",
    "ml-1m_item_stats_cold_start_k3_cap20_eval_users1000_cg5_m2_seed1",
    "ml-1m_item_stats_cold_start_k3_cap20_eval_users1000_cg5_m3_seed1",
    "ml-1m_item_stats_cold_start_k3_cap20_eval_users1000_cg5_m9_seed1",
    "ml-1m_item_stats_cold_start_k3_cap20_eval_users1000_cg5_m19_seed1",
    "ml-1m_item_stats_cold_start_k3_cap20_eval_users1000_cg5_m1_seed2",
    "ml-1m_item_stats_cold_start_k3_cap20_eval_users1000_cg5_m2_seed2",
    "ml-1m_item_stats_cold_start_k3_cap20_eval_users1000_cg5_m3_seed2",
    "ml-1m_item_stats_cold_start_k3_cap20_eval_users1000_cg5_m9_seed2",
    "ml-1m_item_stats_cold_start_k3_cap20_eval_users1000_cg5_m19_seed2",
    "ml-1m_item_stats_cold_start_k5_cap20_eval_users1000_cg5_m1_seed0",
    "ml-1m_item_stats_cold_start_k5_cap20_eval_users1000_cg5_m2_seed0",
    "ml-1m_item_stats_cold_start_k5_cap20_eval_users1000_cg5_m3_seed0",
    "ml-1m_item_stats_cold_start_k5_cap20_eval_users1000_cg5_m9_seed0",
    "ml-1m_item_stats_cold_start_k5_cap20_eval_users1000_cg5_m19_seed0",
    "ml-1m_item_stats_cold_start_k5_cap20_eval_users1000_cg5_m1_seed1",
    "ml-1m_item_stats_cold_start_k5_cap20_eval_users1000_cg5_m2_seed1",
    "ml-1m_item_stats_cold_start_k5_cap20_eval_users1000_cg5_m3_seed1",
    "ml-1m_item_stats_cold_start_k5_cap20_eval_users1000_cg5_m9_seed1",
    "ml-1m_item_stats_cold_start_k5_cap20_eval_users1000_cg5_m19_seed1",
    "ml-1m_item_stats_cold_start_k5_cap20_eval_users1000_cg5_m1_seed2",
    "ml-1m_item_stats_cold_start_k5_cap20_eval_users1000_cg5_m2_seed2",
    "ml-1m_item_stats_cold_start_k5_cap20_eval_users1000_cg5_m3_seed2",
    "ml-1m_item_stats_cold_start_k5_cap20_eval_users1000_cg5_m9_seed2",
    "ml-1m_item_stats_cold_start_k5_cap20_eval_users1000_cg5_m19_seed2",
]

PIP_DEPS = [
    "implicit>=0.7.2",
    "litellm[proxy]>=1.91.1",
    "matplotlib",
    "openai",
    "pandas",
    "plotly",
    "pyarrow",
    "python-dotenv",
    "scikit-learn",
    "scipy",
    "seaborn",
    "tabulate",
    "tqdm",
]

_log_lock = threading.Lock()


def log(msg: str) -> None:
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _log_lock:
        print(f"[{ts}] {msg}", flush=True)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def local_output_root() -> Path:
    return repo_root() / "outputs" / "experimental" / "cold_start"


def queue_dir() -> Path:
    d = local_output_root() / "_queue"
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_path() -> Path:
    return queue_dir() / "state.json"


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


def kubectl_exec(pod: str, remote_cmd: str, *, check: bool = True, timeout: float | None = 120) -> str:
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
        text = (out.stdout or "").strip()
        return text == "Running,true"
    except Exception:
        return False


def session_name(task: str) -> str:
    # tmux session names: keep short and alphanumeric-ish
    return f"cs_{task[-40:]}"


def discover_llm_base_url(pod: str) -> str:
    """Prefer litellm port from run_litellm.sh; else first healthy vLLM port."""
    script = r"""
set -e
# Prefer litellm launch cmdline
port=$(ps aux 2>/dev/null | grep '[r]un_litellm.sh' | awk '{for(i=1;i<=NF;i++) if($i~/run_litellm\.sh/){print $(i+1); exit}}')
if [ -n "$port" ]; then
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:${port}/v1/models" || echo 000)
  if [ "$code" = "200" ]; then echo "litellm:$port"; exit 0; fi
fi
# litellm.log uvicorn line
if [ -f ~/litellm.log ]; then
  port=$(grep -Eo 'Uvicorn running on http://[^:]+:[0-9]+' ~/litellm.log | tail -1 | grep -Eo '[0-9]+$')
  if [ -n "$port" ]; then
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:${port}/v1/models" || echo 000)
    if [ "$code" = "200" ]; then echo "litellm:$port"; exit 0; fi
  fi
fi
# fallback: probe common litellm then vllm ports
for port in 8080 8081 8082 8083; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 1 "http://127.0.0.1:${port}/v1/models" || echo 000)
  if [ "$code" = "200" ]; then echo "litellm:$port"; exit 0; fi
done
for port in 8000 8001 8002 8003; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 1 "http://127.0.0.1:${port}/v1/models" || echo 000)
  if [ "$code" = "200" ]; then echo "vllm:$port"; exit 0; fi
done
echo "none:0"
"""
    out = kubectl_exec(pod, script, check=False, timeout=60)
    kind, _, port = out.partition(":")
    if kind in {"litellm", "vllm"} and port.isdigit() and int(port) > 0:
        if kind == "vllm":
            log(f"{pod}: litellm unavailable, falling back to vLLM :{port}")
        return f"http://127.0.0.1:{port}/v1"
    raise RuntimeError(f"{pod}: no healthy LLM endpoint found ({out!r})")


def taste_cache_paths() -> list[Path]:
    cache_dir = repo_root() / "outputs" / "agent4rec_taste_cache"
    paths = [cache_dir / name for name in TASTE_CACHE_FILES]
    missing = [str(p) for p in paths if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            "Missing required Agent4Rec taste cache files:\n" + "\n".join(missing)
        )
    return paths


def build_zip(zip_path: Path) -> None:
    root = repo_root()
    exclude_dirs = {
        ".venv",
        "outputs",
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        ".mypy_cache",
        "node_modules",
    }
    exclude_names = {".env", ".DS_Store"}
    taste_paths = taste_cache_paths()
    if zip_path.exists():
        zip_path.unlink()
    log(f"Building zip {zip_path}")
    count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = Path(dirpath).relative_to(root)
            # prune excluded dirs
            dirnames[:] = [
                d
                for d in dirnames
                if d not in exclude_dirs and not d.startswith(".venv")
            ]
            # keep data/canonical; skip other large data if any
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
                full = Path(dirpath) / name
                arc = full.relative_to(root).as_posix()
                zf.write(full, arcname=arc)
                count += 1
        for full in taste_paths:
            arc = full.relative_to(root).as_posix()
            zf.write(full, arcname=arc)
            count += 1
    log(f"Zip ready: {zip_path} ({count} files, {zip_path.stat().st_size / 1e6:.1f} MB)")


def upload_dotenv(pod: str) -> None:
    """Copy local .env for OpenAI taste-cache misses (never pack secrets into the zip)."""
    env_path = repo_root() / ".env"
    if not env_path.is_file():
        log(f"{pod}: WARNING no local .env; taste cache misses will fail without OPENAI_API_KEY")
        return
    run_cmd(
        [*kubectl_base(), "cp", str(env_path), f"{pod}:{REMOTE_REPO}/.env"],
        check=True,
        capture=False,
    )


def upload_and_setup_pod(pod: str, zip_path: Path) -> None:
    log(f"{pod}: uploading zip")
    run_cmd([*kubectl_base(), "cp", str(zip_path), f"{pod}:{REMOTE_ZIP}"], check=True, capture=False)

    deps = " ".join(shlex.quote(d) for d in PIP_DEPS)
    cache_checks = " and ".join(
        f"Path('outputs/agent4rec_taste_cache/{name}').is_file()" for name in TASTE_CACHE_FILES
    )
    setup = f"""
set -euo pipefail
rm -rf {REMOTE_REPO}
mkdir -p {REMOTE_REPO}
cd {REMOTE_REPO}
unzip -q -o {REMOTE_ZIP}
/opt/conda/bin/python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q {deps}
mkdir -p outputs/experimental/cold_start
export PYTHONPATH="{REMOTE_REPO}/src:{REMOTE_REPO}"
.venv/bin/python -c "from runners.in_distribution.cold_start.methods import METHOD_RUNNERS; assert '{METHOD}' in METHOD_RUNNERS; from pathlib import Path; p=Path('data/canonical/ml-1m/v1/interactions.parquet'); assert p.exists(), p; assert {cache_checks}"
echo SETUP_OK
"""
    log(f"{pod}: unzip + venv + pip install")
    out = kubectl_exec(pod, setup, check=True, timeout=1800)
    if "SETUP_OK" not in out:
        raise RuntimeError(f"{pod}: setup failed: {out[-500:]}")
    upload_dotenv(pod)
    log(f"{pod}: setup complete")


def ensure_pod_setup(pod: str, zip_path: Path | None = None) -> None:
    cache_checks = " && ".join(
        f"[ -s {REMOTE_REPO}/outputs/agent4rec_taste_cache/{name} ]" for name in TASTE_CACHE_FILES
    )
    check = f"""
if [ -x {REMOTE_REPO}/.venv/bin/python ] && [ -f {REMOTE_REPO}/data/canonical/ml-1m/v1/interactions.parquet ] && {cache_checks}; then
  export PYTHONPATH="{REMOTE_REPO}/src:{REMOTE_REPO}"
  if {REMOTE_REPO}/.venv/bin/python -c "from runners.in_distribution.cold_start.methods import METHOD_RUNNERS; assert '{METHOD}' in METHOD_RUNNERS" 2>/dev/null; then
    echo READY
    exit 0
  fi
fi
echo NEED_SETUP
"""
    out = kubectl_exec(pod, check, check=False, timeout=60)
    if "READY" in out:
        return
    if zip_path is None:
        raise RuntimeError(f"{pod}: needs setup but no zip provided")
    upload_and_setup_pod(pod, zip_path)


def local_task_done(task: str) -> bool:
    root = local_output_root()
    pattern = f"*_{task}_{METHOD}"
    for d in root.glob(pattern):
        if d.is_dir() and (d / "metrics.json").exists() and (d / "metrics_ranking.json").exists():
            return True
    return False


def _assert_safe_name(name: str) -> str:
    # Task/method names are controlled identifiers; keep shell globs unquoted.
    if not name or any(c in name for c in " \t\n\"'`$;&|<>(){}[]\\"):
        raise ValueError(f"unsafe name for remote shell: {name!r}")
    return name


def remote_result_dir(pod: str, task: str) -> str | None:
    task = _assert_safe_name(task)
    method = _assert_safe_name(METHOD)
    # Do not shlex.quote inside the glob — quotes become literal characters.
    cmd = f"""
set +e
cd {REMOTE_OUTPUT} 2>/dev/null || exit 0
for d in *_{task}_{method}; do
  [ -d "$d" ] || continue
  if [ -f "$d/metrics.json" ] && [ -f "$d/metrics_ranking.json" ]; then
    printf '%s\\n' "$d"
    exit 0
  fi
done
exit 0
"""
    out = kubectl_exec(pod, cmd, check=False, timeout=60)
    return out if out else None


def tmux_has_session(pod: str, name: str) -> bool:
    out = kubectl_exec(
        pod,
        f"tmux has-session -t {shlex.quote(name)} 2>/dev/null && echo YES || echo NO",
        check=False,
        timeout=30,
    )
    return "YES" in out


def start_task(pod: str, task: str, base_url: str) -> None:
    sess = session_name(task)
    # kill stale session with same name
    kubectl_exec(
        pod,
        f"tmux kill-session -t {shlex.quote(sess)} 2>/dev/null || true",
        check=False,
        timeout=30,
    )
    log_path = f"{REMOTE_OUTPUT}/{task}.log"
    inner = (
        f"cd {REMOTE_REPO} && "
        f"source .venv/bin/activate && "
        f'export PYTHONPATH="{REMOTE_REPO}/src:{REMOTE_REPO}" && '
        f'export BEYOND_CLICK_SIM_VLLM_LOCAL_BASE_URL={shlex.quote(base_url)} && '
        f"mkdir -p {REMOTE_OUTPUT} && "
        f"python -m runners.in_distribution.cold_start.run "
        f"--methods {METHOD} "
        f"--output-dir outputs/experimental/cold_start "
        f"--resume "
        f"--tasks {shlex.quote(task)} "
        f"> {shlex.quote(log_path)} 2>&1; "
        f"echo EXIT:$? >> {shlex.quote(log_path)}"
    )
    # wrap in bash -lc inside tmux
    remote = (
        f"tmux new-session -d -s {shlex.quote(sess)} "
        f"{shlex.quote('bash -lc ' + shlex.quote(inner))}"
    )
    kubectl_exec(pod, remote, check=True, timeout=60)
    log(f"{pod}: started {task} (tmux {sess}, base_url={base_url})")


def download_result(pod: str, task: str, result_dirname: str) -> Path:
    remote_dir = f"{REMOTE_OUTPUT}/{result_dirname}"
    remote_zip = f"/tmp/{result_dirname}.zip"
    local_zip = queue_dir() / f"{result_dirname}.zip"
    kubectl_exec(
        pod,
        f"cd {REMOTE_OUTPUT} && rm -f {shlex.quote(remote_zip)} && "
        f"zip -qr {shlex.quote(remote_zip)} {shlex.quote(result_dirname)}",
        check=True,
        timeout=600,
    )
    if local_zip.exists():
        local_zip.unlink()
    run_cmd(
        [*kubectl_base(), "cp", f"{pod}:{remote_zip}", str(local_zip)],
        check=True,
        capture=False,
    )
    dest_root = local_output_root()
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(dest_root)
    local_dir = dest_root / result_dirname
    if not (local_dir / "metrics.json").exists() or not (local_dir / "metrics_ranking.json").exists():
        raise RuntimeError(f"Downloaded incomplete result for {task}: {local_dir}")
    log(f"{pod}: downloaded {result_dirname} -> {local_dir}")
    return local_dir


@dataclass
class TaskState:
    status: str = "pending"  # pending|running|done|failed
    pod: str | None = None
    attempts: int = 0
    error: str | None = None
    result_dir: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


@dataclass
class QueueState:
    tasks: dict[str, TaskState] = field(default_factory=dict)
    retries: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tasks": {k: v.__dict__ for k, v in self.tasks.items()},
            "retries": self.retries,
        }

    @classmethod
    def load(cls, path: Path) -> QueueState:
        if not path.exists():
            st = cls()
            for t in TASKS:
                st.tasks[t] = TaskState(status="done" if local_task_done(t) else "pending")
            return st
        raw = json.loads(path.read_text())
        st = cls(retries=raw.get("retries", []))
        for t in TASKS:
            info = raw.get("tasks", {}).get(t, {})
            ts = TaskState(**{k: info[k] for k in TaskState.__dataclass_fields__ if k in info})
            if local_task_done(t):
                ts.status = "done"
            st.tasks[t] = ts
        return st

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


class Orchestrator:
    def __init__(self, *, zip_path: Path, max_attempts: int = 5, poll_seconds: float = 30.0):
        self.zip_path = zip_path
        self.max_attempts = max_attempts
        self.poll_seconds = poll_seconds
        self.state = QueueState.load(state_path())
        self._lock = threading.Lock()

    def save(self) -> None:
        with self._lock:
            self.state.save(state_path())

    def setup_all(self) -> None:
        build_zip(self.zip_path)
        with ThreadPoolExecutor(max_workers=len(PODS)) as ex:
            futs = {ex.submit(upload_and_setup_pod, pod, self.zip_path): pod for pod in PODS}
            errors = []
            for fut in as_completed(futs):
                pod = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    errors.append(f"{pod}: {e}")
                    log(f"ERROR setup {pod}: {e}")
            if errors:
                raise RuntimeError("Setup failed:\n" + "\n".join(errors))

    def pending_tasks(self) -> list[str]:
        return [t for t, s in self.state.tasks.items() if s.status == "pending"]

    def running_by_pod(self) -> dict[str, str]:
        return {
            s.pod: t
            for t, s in self.state.tasks.items()
            if s.status == "running" and s.pod
        }

    def free_pods(self) -> list[str]:
        busy = set(self.running_by_pod())
        return [p for p in PODS if p not in busy and pod_ready(p)]

    def assign_pending(self) -> None:
        while True:
            pending = self.pending_tasks()
            free = self.free_pods()
            if not pending or not free:
                return
            pod = free[0]
            task = pending[0]
            try:
                ensure_pod_setup(pod, self.zip_path)
                base_url = discover_llm_base_url(pod)
                # If this pod already finished this task (reload/resume), pull now
                existing = remote_result_dir(pod, task)
                if existing:
                    log(f"{pod}: found existing complete result for {task}")
                    download_result(pod, task, existing)
                    st = self.state.tasks[task]
                    st.status = "done"
                    st.pod = pod
                    st.result_dir = existing
                    st.finished_at = datetime.now(UTC).isoformat()
                    self.save()
                    continue
                start_task(pod, task, base_url)
                st = self.state.tasks[task]
                st.status = "running"
                st.pod = pod
                st.attempts += 1
                st.started_at = datetime.now(UTC).isoformat()
                st.error = None
                self.save()
            except Exception as e:
                log(f"ERROR assigning {task} -> {pod}: {e}")
                st = self.state.tasks[task]
                st.attempts += 1
                st.error = str(e)
                if st.attempts >= self.max_attempts:
                    st.status = "failed"
                else:
                    st.status = "pending"
                    st.pod = None
                    self.state.retries.append(
                        {"task": task, "pod": pod, "error": str(e), "at": datetime.now(UTC).isoformat()}
                    )
                self.save()
                # avoid tight spin on a broken pod
                return

    def poll_running(self) -> None:
        running = [(t, s) for t, s in self.state.tasks.items() if s.status == "running"]
        for task, st in running:
            pod = st.pod
            if not pod:
                st.status = "pending"
                self.save()
                continue
            if not pod_ready(pod):
                log(f"{pod}: not ready while running {task}; trying salvage then requeue")
                existing = None
                try:
                    existing = remote_result_dir(pod, task)
                except Exception:
                    existing = None
                if existing:
                    try:
                        download_result(pod, task, existing)
                        st.status = "done"
                        st.result_dir = existing
                        st.finished_at = datetime.now(UTC).isoformat()
                        self.save()
                        continue
                    except Exception as e:
                        log(f"{pod}: salvage download failed: {e}")
                # requeue
                log(f"{pod}: requeue {task} after pod not ready (attempt {st.attempts})")
                self.state.retries.append(
                    {
                        "task": task,
                        "pod": pod,
                        "error": "pod_not_ready",
                        "at": datetime.now(UTC).isoformat(),
                    }
                )
                st.status = "pending" if st.attempts < self.max_attempts else "failed"
                st.pod = None
                st.error = "pod_not_ready"
                self.save()
                continue

            try:
                existing = remote_result_dir(pod, task)
            except Exception as e:
                log(f"{pod}: poll error for {task}: {e}")
                continue

            if existing:
                try:
                    download_result(pod, task, existing)
                    st.status = "done"
                    st.result_dir = existing
                    st.finished_at = datetime.now(UTC).isoformat()
                    self.save()
                    log(f"DONE {task} on {pod}")
                except Exception as e:
                    log(f"ERROR download {task} from {pod}: {e}")
                    st.error = str(e)
                    if st.attempts >= self.max_attempts:
                        st.status = "failed"
                    else:
                        st.status = "pending"
                        st.pod = None
                    self.save()
                continue

            # still running?
            sess = session_name(task)
            try:
                alive = tmux_has_session(pod, sess)
            except Exception as e:
                log(f"{pod}: tmux check failed for {task}: {e}")
                continue
            if not alive:
                # session ended without metrics
                log(f"{pod}: tmux ended without metrics for {task}; requeue/fail")
                self.state.retries.append(
                    {
                        "task": task,
                        "pod": pod,
                        "error": "tmux_exited_without_metrics",
                        "at": datetime.now(UTC).isoformat(),
                    }
                )
                if st.attempts >= self.max_attempts:
                    st.status = "failed"
                    st.error = "tmux_exited_without_metrics"
                else:
                    st.status = "pending"
                    st.pod = None
                    st.error = "tmux_exited_without_metrics"
                self.save()

    def summary(self) -> dict[str, int]:
        counts = {"pending": 0, "running": 0, "done": 0, "failed": 0}
        for s in self.state.tasks.values():
            counts[s.status] = counts.get(s.status, 0) + 1
        return counts

    def run_queue(self) -> int:
        log(f"Starting queue: {self.summary()}")
        while True:
            # refresh done from local disk
            for t, s in self.state.tasks.items():
                if s.status != "done" and local_task_done(t):
                    s.status = "done"
            self.save()

            self.poll_running()
            self.assign_pending()
            counts = self.summary()
            log(f"status={counts} retries={len(self.state.retries)}")
            if counts["pending"] == 0 and counts["running"] == 0:
                break
            time.sleep(self.poll_seconds)

        failed = [t for t, s in self.state.tasks.items() if s.status == "failed"]
        done = [t for t, s in self.state.tasks.items() if s.status == "done"]
        log(f"Queue finished: done={len(done)} failed={len(failed)}")
        if failed:
            for t in failed:
                log(f"FAILED: {t} ({self.state.tasks[t].error})")
            return 1
        return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--zip-path",
        default=str(queue_dir() / "beyond-click-sim-code.zip"),
        help="Local path for the upload zip",
    )
    p.add_argument("--setup-only", action="store_true")
    p.add_argument("--queue-only", action="store_true", help="Skip setup; assume pods ready")
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--max-attempts", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip_path)
    orch = Orchestrator(
        zip_path=zip_path,
        max_attempts=args.max_attempts,
        poll_seconds=args.poll_seconds,
    )
    if not args.queue_only:
        orch.setup_all()
    if args.setup_only:
        return 0
    return orch.run_queue()


if __name__ == "__main__":
    sys.exit(main())
