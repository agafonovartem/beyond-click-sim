#!/usr/bin/env python3
"""Queue cold-start LLM tasks across agent-exp Kubernetes pods.

Upload/setup once per pod, discover litellm port, run up to N tasks per pod in
tmux until all tasks are remotely complete, then archive/download results in a
separate phase after a quiescence barrier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
PODS = [f"agent-exp-policy-ranking-{i}" for i in range(8)]
METHOD = "llm_preference_listwise_ranking_vllm_qwen36_27b_sumboth_full"
REMOTE_REPO = "/home/jovyan/beyond-click-sim"
REMOTE_ZIP = "/home/jovyan/beyond-click-sim-code.zip"
LOCAL_OUTPUT_REL = Path("outputs") / "experimental" / METHOD
REMOTE_OUTPUT = f"{REMOTE_REPO}/{LOCAL_OUTPUT_REL.as_posix()}"
MAX_TASKS_PER_POD = 4
# Precomputed Agent4Rec taste profiles required on-pod for traits+taste methods.
# The cache is keyed by the split seed only (SPLIT_SEED=0 for the whole
# cold-start grid), so one file covers every task below.
TASTE_CACHE_FILES = [
    "ml-1m_v1_seed0_gpt-4o-mini_agent4rec_modify_v1.jsonl",
]


def required_metric_files(method: str) -> tuple[str, ...]:
    if method.startswith(
        (
            "llm_preference_listwise_ranking_",
            "llm_listwise_ranking_",
            "agent4rec_preference_listwise_ranking_",
            "agent4rec_listwise_ranking_",
        )
    ):
        return ("metrics_ranking.json",)
    return ("metrics.json", "metrics_ranking.json")


def method_requires_item_stats_tasks(method: str) -> bool:
    # Agent4Rec always needs History ratings columns; LLM methods need them
    # only when the method name encodes an explicit item-stats ablation.
    return method.startswith("agent4rec_") or "_item_stats_" in method


def method_requires_taste_cache(method: str) -> bool:
    return "_traits_taste_" in method


def method_target_kind(method: str) -> str:
    if method.startswith(
        (
            "llm_listwise_ranking_",
            "agent4rec_listwise_ranking_",
        )
    ):
        return "interact"
    return "like"


def task_names_for_method(method: str = METHOD) -> list[str]:
    """Task grid, uniform negatives (the pre-redesign compute scope).

    Agent4Rec methods use item_stats tasks; LLM methods use plain tasks.
    Names come from the runner registry — single source of truth for the
    naming scheme. Add "popularity_matched" to STRATEGIES to double the grid
    with the popularity-matched condition (90 tasks).
    """
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo))
    from runners.in_distribution.cold_start.task_builders import (
        cold_start_task_name,
    )

    strategies = ["uniform"]  # add "popularity_matched" for the full grid
    use_item_rating_stats = method_requires_item_stats_tasks(method)
    target_kind = method_target_kind(method)
    return [
        cold_start_task_name(
            "ml-1m",
            target_kind,
            k,
            m,
            strategy,
            sampling_seed,
            use_item_rating_stats=use_item_rating_stats,
        )
        for k in (1, 3, 5)
        for m in (1, 2, 3, 9, 19)
        for strategy in strategies
        for sampling_seed in (0, 1, 2)
    ]


TASKS = task_names_for_method(METHOD)
REQUIRED_METRIC_FILES = required_metric_files(METHOD)

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
    return repo_root() / LOCAL_OUTPUT_REL


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
    # tmux session names must be unique per full task. Truncating the task
    # suffix collides across k=1/3/5 (and similar) cold-start grids.
    method_digest = hashlib.sha256(METHOD.encode("utf-8")).hexdigest()[:8]
    task_digest = hashlib.sha256(task.encode("utf-8")).hexdigest()[:16]
    return f"cs_{method_digest}_{task_digest}"


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
    taste_paths = taste_cache_paths() if method_requires_taste_cache(METHOD) else []
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
    """Copy local .env only for taste-profile methods."""
    if not method_requires_taste_cache(METHOD):
        return
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
    cache_condition = cache_checks if method_requires_taste_cache(METHOD) else "True"
    setup = f"""
set -euo pipefail
rm -rf {REMOTE_REPO}
mkdir -p {REMOTE_REPO}
cd {REMOTE_REPO}
unzip -q -o {REMOTE_ZIP}
/opt/conda/bin/python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q {deps}
mkdir -p {LOCAL_OUTPUT_REL.as_posix()}
export PYTHONPATH="{REMOTE_REPO}/src:{REMOTE_REPO}"
.venv/bin/python -c "from runners.in_distribution.cold_start.methods import METHOD_RUNNERS; assert '{METHOD}' in METHOD_RUNNERS; from pathlib import Path; p=Path('data/canonical/ml-1m/v1/interactions.parquet'); assert p.exists(), p; assert {cache_condition}"
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
    cache_condition = cache_checks if method_requires_taste_cache(METHOD) else "true"
    check = f"""
if [ -x {REMOTE_REPO}/.venv/bin/python ] && [ -f {REMOTE_REPO}/data/canonical/ml-1m/v1/interactions.parquet ] && {cache_condition}; then
  export PYTHONPATH="{REMOTE_REPO}/src:{REMOTE_REPO}"
  if {REMOTE_REPO}/.venv/bin/python -c "from runners.in_distribution.cold_start.methods import METHOD_RUNNERS; assert '{METHOD}' in METHOD_RUNNERS" 2>/dev/null; then
    mkdir -p {REMOTE_OUTPUT}
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


def has_required_metric_files(path: Path) -> bool:
    return all((path / filename).exists() for filename in REQUIRED_METRIC_FILES)


def local_task_done(task: str) -> bool:
    root = local_output_root()
    pattern = f"*_{task}_{METHOD}"
    for d in root.glob(pattern):
        if d.is_dir() and has_required_metric_files(d):
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
    required_checks = " && ".join(
        f'[ -f "$d/{filename}" ]' for filename in REQUIRED_METRIC_FILES
    )
    # Do not shlex.quote inside the glob — quotes become literal characters.
    cmd = f"""
set +e
cd {REMOTE_OUTPUT} 2>/dev/null || exit 0
for d in *_{task}_{method}; do
  [ -d "$d" ] || continue
  if {required_checks}; then
    printf '%s\\n' "$d"
    exit 0
  fi
done
exit 0
"""
    out = kubectl_exec(pod, cmd, check=False, timeout=60)
    return out if out else None


def list_complete_result_dirs(pod: str) -> list[str]:
    """List complete result directory names for this method on one pod."""
    method = _assert_safe_name(METHOD)
    required_checks = " && ".join(
        f'[ -f "$d/{filename}" ]' for filename in REQUIRED_METRIC_FILES
    )
    cmd = f"""
set +e
cd {REMOTE_OUTPUT} 2>/dev/null || exit 0
for d in *_{method}; do
  [ -d "$d" ] || continue
  if {required_checks}; then
    printf '%s\\n' "$d"
  fi
done
exit 0
"""
    out = kubectl_exec(pod, cmd, check=False, timeout=120)
    if not out:
        return []
    return [line for line in out.splitlines() if line.strip()]


def task_from_result_dirname(result_dirname: str) -> str | None:
    """Extract task name from `{timestamp}_{task}_{METHOD}` dirname."""
    method = _assert_safe_name(METHOD)
    suffix = f"_{method}"
    if not result_dirname.endswith(suffix):
        return None
    body = result_dirname[: -len(suffix)]
    # timestamp is 16 chars YYYYMMDDTHHMMSSZ_
    if len(body) < 17 or body[15] != "Z" or body[16] != "_":
        # Fallback: find a known task as substring.
        for task in TASKS:
            marker = f"_{task}_"
            if marker in f"_{body}_":
                return task
        return None
    task = body[17:]
    return task if task in set(TASKS) else None


def remote_result_fingerprint(pod: str, result_dirname: str) -> str:
    """Stable fingerprint of metric files under a completed result dir."""
    result_dirname = _assert_safe_name(result_dirname)
    checks = " ".join(shlex.quote(name) for name in REQUIRED_METRIC_FILES)
    cmd = f"""
set +e
cd {REMOTE_OUTPUT}/{shlex.quote(result_dirname)} 2>/dev/null || {{ echo MISSING; exit 0; }}
for f in {checks}; do
  if [ ! -f "$f" ]; then echo MISSING; exit 0; fi
  stat -c '%n %s %Y' "$f" 2>/dev/null || stat -f '%N %z %m' "$f"
done
"""
    return kubectl_exec(pod, cmd, check=False, timeout=60)


def method_tmux_sessions(pod: str) -> list[str]:
    digest = hashlib.sha256(METHOD.encode("utf-8")).hexdigest()[:8]
    prefix = f"cs_{digest}_"
    out = kubectl_exec(
        pod,
        "tmux list-sessions -F '#{session_name}' 2>/dev/null || true",
        check=False,
        timeout=30,
    )
    if not out:
        return []
    return [line for line in out.splitlines() if line.startswith(prefix)]


def method_runner_pids(pod: str) -> list[str]:
    # Match cold_start.run processes for this method only.
    escaped = re.escape(METHOD)
    out = kubectl_exec(
        pod,
        f"ps -eo pid=,args= 2>/dev/null | grep -E '[p]ython.*runners\\.in_distribution\\.cold_start\\.run.*{escaped}' || true",
        check=False,
        timeout=30,
    )
    pids: list[str] = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if parts:
            pids.append(parts[0])
    return pids


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
    log_path = f"{REMOTE_OUTPUT}/{task}_{METHOD}.log"
    inner = (
        f"cd {REMOTE_REPO} && "
        f"source .venv/bin/activate && "
        f'export PYTHONPATH="{REMOTE_REPO}/src:{REMOTE_REPO}" && '
        f'export BEYOND_CLICK_SIM_VLLM_LOCAL_BASE_URL={shlex.quote(base_url)} && '
        f"mkdir -p {REMOTE_OUTPUT} && "
        f"python -m runners.in_distribution.cold_start.run "
        f"--methods {METHOD} "
        f"--output-dir {LOCAL_OUTPUT_REL.as_posix()} "
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


def download_selected_results(pod: str, result_dirs: list[str]) -> list[Path]:
    """Archive selected completed dirs on one pod and extract locally."""
    if not result_dirs:
        return []
    for d in result_dirs:
        _assert_safe_name(d)
    archive_name = f"{pod}_selected_results.zip"
    remote_zip = f"/tmp/{archive_name}"
    local_zip = queue_dir() / archive_name
    quoted_dirs = " ".join(shlex.quote(d) for d in result_dirs)
    kubectl_exec(
        pod,
        f"cd {REMOTE_OUTPUT} && rm -f {shlex.quote(remote_zip)} && "
        f"zip -qr {shlex.quote(remote_zip)} {quoted_dirs}",
        check=True,
        timeout=1200,
    )
    last_err: Exception | None = None
    for attempt in range(1, 4):
        if local_zip.exists():
            local_zip.unlink()
        try:
            run_cmd(
                [*kubectl_base(), "cp", f"{pod}:{remote_zip}", str(local_zip)],
                check=True,
                capture=False,
            )
            if local_zip.is_file() and local_zip.stat().st_size > 0:
                break
            raise RuntimeError(f"{pod}: empty archive after kubectl cp")
        except Exception as e:
            last_err = e
            log(f"{pod}: kubectl cp attempt {attempt}/3 failed: {e}")
            time.sleep(2 * attempt)
    else:
        raise RuntimeError(f"{pod}: kubectl cp failed after retries: {last_err}")
    dest_root = local_output_root()
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(dest_root)
    extracted: list[Path] = []
    for result_dirname in result_dirs:
        local_dir = dest_root / result_dirname
        if not has_required_metric_files(local_dir):
            raise RuntimeError(f"Downloaded incomplete result: {local_dir}")
        extracted.append(local_dir)
    log(f"{pod}: downloaded {len(extracted)} result dirs -> {dest_root}")
    return extracted


@dataclass
class TaskState:
    status: str = "pending"  # pending|running|remote_done|failed|downloaded
    pod: str | None = None
    attempts: int = 0
    error: str | None = None
    result_dir: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


@dataclass
class QueueState:
    method: str = METHOD
    tasks: dict[str, TaskState] = field(default_factory=dict)
    retries: list[dict[str, str]] = field(default_factory=list)
    discarded_duplicates: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "tasks": {k: v.__dict__ for k, v in self.tasks.items()},
            "retries": self.retries,
            "discarded_duplicates": self.discarded_duplicates,
        }

    @classmethod
    def load(cls, path: Path) -> QueueState:
        if not path.exists():
            st = cls(method=METHOD)
            for t in TASKS:
                if local_task_done(t):
                    st.tasks[t] = TaskState(status="downloaded")
                else:
                    st.tasks[t] = TaskState(status="pending")
            return st
        raw = json.loads(path.read_text())
        if raw.get("method") != METHOD:
            st = cls(method=METHOD)
            for t in TASKS:
                if local_task_done(t):
                    st.tasks[t] = TaskState(status="downloaded")
                else:
                    st.tasks[t] = TaskState(status="pending")
            return st
        st = cls(
            method=METHOD,
            retries=raw.get("retries", []),
            discarded_duplicates=raw.get("discarded_duplicates", []),
        )
        for t in TASKS:
            info = raw.get("tasks", {}).get(t, {})
            ts = TaskState(**{k: info[k] for k in TaskState.__dataclass_fields__ if k in info})
            if local_task_done(t):
                ts.status = "downloaded"
            elif ts.status in {"done", "downloaded"}:
                # Local artifact missing: treat as pending for resume.
                ts.status = "pending"
                ts.result_dir = None
                ts.finished_at = None
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
        self._setup_allowed = True

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

    def running_tasks_by_pod(self) -> dict[str, list[str]]:
        by_pod: dict[str, list[str]] = {p: [] for p in PODS}
        for task, st in self.state.tasks.items():
            if st.status == "running" and st.pod:
                by_pod.setdefault(st.pod, []).append(task)
        return by_pod

    def available_slots(self, pod: str) -> int:
        running = len(self.running_tasks_by_pod().get(pod, []))
        return max(0, MAX_TASKS_PER_POD - running)

    def free_pods(self) -> list[str]:
        """Pods with free slots, least-loaded first, then lower index."""
        ranked: list[tuple[int, int, str]] = []
        for idx, pod in enumerate(PODS):
            if not pod_ready(pod):
                continue
            slots = self.available_slots(pod)
            if slots <= 0:
                continue
            running = MAX_TASKS_PER_POD - slots
            ranked.append((running, idx, pod))
        ranked.sort()
        return [pod for _, _, pod in ranked]

    def assign_pending(self) -> None:
        while True:
            pending = self.pending_tasks()
            free = self.free_pods()
            if not pending or not free:
                return
            pod = free[0]
            task = pending[0]
            try:
                # After run phase starts, never wipe/reinstall a pod.
                if self._setup_allowed:
                    ensure_pod_setup(pod, self.zip_path)
                else:
                    # Require already-ready repo; otherwise skip this pod.
                    check = kubectl_exec(
                        pod,
                        f"[ -x {REMOTE_REPO}/.venv/bin/python ] && echo READY || echo NEED_SETUP",
                        check=False,
                        timeout=30,
                    )
                    if "READY" not in check:
                        log(f"{pod}: not staged and setup disabled; skipping for now")
                        return
                base_url = discover_llm_base_url(pod)
                existing = remote_result_dir(pod, task)
                if existing:
                    log(f"{pod}: found existing complete result for {task}")
                    st = self.state.tasks[task]
                    st.status = "remote_done"
                    st.pod = pod
                    st.result_dir = existing
                    st.finished_at = datetime.now(UTC).isoformat()
                    st.error = None
                    self.save()
                    continue
                start_task(pod, task, base_url)
                self._setup_allowed = False
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
                        {
                            "task": task,
                            "pod": pod,
                            "error": str(e),
                            "at": datetime.now(UTC).isoformat(),
                        }
                    )
                self.save()
                # avoid tight spin on a broken pod
                return

    def _requeue(self, task: str, pod: str | None, error: str) -> None:
        st = self.state.tasks[task]
        self.state.retries.append(
            {
                "task": task,
                "pod": pod or "",
                "error": error,
                "at": datetime.now(UTC).isoformat(),
            }
        )
        if st.attempts >= self.max_attempts:
            st.status = "failed"
            st.error = error
        else:
            st.status = "pending"
            st.pod = None
            st.result_dir = None
            st.error = error
        self.save()

    def poll_running(self) -> None:
        running = [(t, s) for t, s in self.state.tasks.items() if s.status == "running"]
        for task, st in running:
            pod = st.pod
            if not pod:
                st.status = "pending"
                self.save()
                continue
            if not pod_ready(pod):
                log(f"{pod}: not ready while running {task}; checking remote result then requeue")
                existing = None
                try:
                    existing = remote_result_dir(pod, task)
                except Exception:
                    existing = None
                if existing:
                    st.status = "remote_done"
                    st.result_dir = existing
                    st.finished_at = datetime.now(UTC).isoformat()
                    st.error = None
                    self.save()
                    log(f"REMOTE_DONE {task} on {pod} (salvaged after pod not ready)")
                    continue
                log(f"{pod}: requeue {task} after pod not ready (attempt {st.attempts})")
                self._requeue(task, pod, "pod_not_ready")
                continue

            try:
                existing = remote_result_dir(pod, task)
            except Exception as e:
                log(f"{pod}: poll error for {task}: {e}")
                continue

            if existing:
                st.status = "remote_done"
                st.result_dir = existing
                st.finished_at = datetime.now(UTC).isoformat()
                st.error = None
                self.save()
                log(f"REMOTE_DONE {task} on {pod}")
                continue

            # still running?
            sess = session_name(task)
            try:
                alive = tmux_has_session(pod, sess)
            except Exception as e:
                log(f"{pod}: tmux check failed for {task}: {e}")
                continue
            if not alive:
                log(f"{pod}: tmux ended without metrics for {task}; requeue/fail")
                self._requeue(task, pod, "tmux_exited_without_metrics")

    def reconcile_remote_results(self) -> int:
        """Scan all ready pods for complete results of pending tasks.

        Needed after retries or orchestrator restarts so a finished result on
        pod A is not blindly restarted on pod B.
        """
        claimed = 0
        pending = set(self.pending_tasks())
        if not pending:
            return 0
        # One listing per pod (not one kubectl call per task×pod).
        by_task: dict[str, list[tuple[str, str]]] = {}
        for pod in PODS:
            if not pod_ready(pod):
                continue
            try:
                dirs = list_complete_result_dirs(pod)
            except Exception as e:
                log(f"{pod}: reconcile listing failed: {e}")
                continue
            for result_dirname in dirs:
                task = task_from_result_dirname(result_dirname)
                if task is None or task not in pending:
                    continue
                by_task.setdefault(task, []).append((pod, result_dirname))

        for task, found in sorted(by_task.items()):
            found.sort(key=lambda x: (PODS.index(x[0]) if x[0] in PODS else 999, x[1]))
            keep_pod, keep_dir = found[0]
            for disc_pod, disc_dir in found[1:]:
                self.state.discarded_duplicates.append(
                    {
                        "task": task,
                        "kept_pod": keep_pod,
                        "kept_dir": keep_dir,
                        "discarded_pod": disc_pod,
                        "discarded_dir": disc_dir,
                    }
                )
            st = self.state.tasks[task]
            st.status = "remote_done"
            st.pod = keep_pod
            st.result_dir = keep_dir
            st.finished_at = datetime.now(UTC).isoformat()
            st.error = None
            claimed += 1
            log(f"REMOTE_DONE {task} on {keep_pod} (reconciled existing {keep_dir})")
        if claimed:
            self.save()
        return claimed

    def summary(self) -> dict[str, int]:
        counts = {
            "pending": 0,
            "running": 0,
            "remote_done": 0,
            "failed": 0,
            "downloaded": 0,
        }
        for s in self.state.tasks.values():
            counts[s.status] = counts.get(s.status, 0) + 1
        return counts

    def collect_fingerprint_map(self) -> dict[str, str]:
        """Map task -> remote metric fingerprint for remote_done tasks."""
        fps: dict[str, str] = {}
        for task, st in self.state.tasks.items():
            if st.status != "remote_done" or not st.pod or not st.result_dir:
                continue
            try:
                fps[task] = remote_result_fingerprint(st.pod, st.result_dir)
            except Exception as e:
                fps[task] = f"ERROR:{e}"
        return fps

    def quiescence_checks(self) -> tuple[bool, list[str]]:
        """Return (ok, reasons). Does not include the two-poll fingerprint check."""
        reasons: list[str] = []
        counts = self.summary()
        if counts["pending"] != 0:
            reasons.append(f"pending={counts['pending']}")
        if counts["running"] != 0:
            reasons.append(f"running={counts['running']}")
        if counts["failed"] != 0:
            reasons.append(f"failed={counts['failed']}")
        remote_done = [
            t for t, s in self.state.tasks.items() if s.status in {"remote_done", "downloaded"}
        ]
        if len(remote_done) != len(TASKS):
            reasons.append(f"remote_complete={len(remote_done)}/{len(TASKS)}")

        complete_on_pod: dict[str, set[str]] = {}
        for pod in PODS:
            owned = [
                st
                for st in self.state.tasks.values()
                if st.status in {"remote_done", "downloaded"} and st.pod == pod
            ]
            if not owned:
                continue
            if not pod_ready(pod):
                reasons.append(f"{pod}: not ready but owns inventory")
                continue
            try:
                complete_on_pod[pod] = set(list_complete_result_dirs(pod))
            except Exception as e:
                reasons.append(f"{pod}: listing failed: {e}")
                continue

        for task, st in self.state.tasks.items():
            if st.status not in {"remote_done", "downloaded"}:
                continue
            if not st.pod or not st.result_dir:
                reasons.append(f"{task}: missing pod/result_dir inventory")
                continue
            present = complete_on_pod.get(st.pod)
            if present is None:
                continue
            if st.result_dir not in present:
                reasons.append(
                    f"{task}: expected {st.result_dir} on {st.pod}, not in complete listing"
                )

        for pod in PODS:
            if not pod_ready(pod):
                # Unready pods that hold no inventory are fine; inventory check above covers owned ones.
                continue
            try:
                sessions = method_tmux_sessions(pod)
                runners = method_runner_pids(pod)
            except Exception as e:
                reasons.append(f"{pod}: writer check failed: {e}")
                continue
            if sessions:
                reasons.append(f"{pod}: leftover tmux {sessions}")
            if runners:
                reasons.append(f"{pod}: leftover runners {runners}")
        return (not reasons, reasons)

    def wait_for_quiescence(self) -> None:
        log("Waiting for quiescence barrier (two consecutive stable polls)")
        prev_fps: dict[str, str] | None = None
        while True:
            ok, reasons = self.quiescence_checks()
            if not ok:
                log(f"quiescence not ready: {reasons[:8]}")
                prev_fps = None
                time.sleep(self.poll_seconds)
                continue
            fps = self.collect_fingerprint_map()
            if any(v.startswith("ERROR:") or v == "MISSING" for v in fps.values()):
                log("quiescence fingerprints incomplete; retrying")
                prev_fps = None
                time.sleep(self.poll_seconds)
                continue
            if prev_fps is not None and prev_fps == fps:
                log("Quiescence barrier reached")
                return
            log("quiescence first stable poll; confirming once more")
            prev_fps = fps
            time.sleep(self.poll_seconds)

    def build_download_inventory(self) -> dict[str, list[str]]:
        """pod -> selected result_dirnames. Records discarded duplicates if any."""
        by_pod: dict[str, list[str]] = {p: [] for p in PODS}
        # One listing per pod to find accidental duplicates (not task×pod kubectl).
        remote_by_task: dict[str, list[tuple[str, str]]] = {}
        for pod in PODS:
            if not pod_ready(pod):
                continue
            try:
                dirs = list_complete_result_dirs(pod)
            except Exception as e:
                log(f"{pod}: download inventory listing failed: {e}")
                continue
            for result_dirname in dirs:
                task = task_from_result_dirname(result_dirname)
                if task is None:
                    continue
                remote_by_task.setdefault(task, []).append((pod, result_dirname))

        for task in TASKS:
            st = self.state.tasks[task]
            if st.status not in {"remote_done", "downloaded"}:
                raise RuntimeError(f"Cannot download: {task} status={st.status}")
            if not st.pod or not st.result_dir:
                raise RuntimeError(f"Cannot download: {task} missing inventory")
            for other_pod, other_dir in remote_by_task.get(task, []):
                if other_pod == st.pod and other_dir == st.result_dir:
                    continue
                if other_dir != st.result_dir or other_pod != st.pod:
                    self.state.discarded_duplicates.append(
                        {
                            "task": task,
                            "kept_pod": st.pod,
                            "kept_dir": st.result_dir,
                            "discarded_pod": other_pod,
                            "discarded_dir": other_dir,
                        }
                    )
            by_pod.setdefault(st.pod, []).append(st.result_dir)
        self.save()
        return {pod: dirs for pod, dirs in by_pod.items() if dirs}

    def download_all(self) -> None:
        inventory = self.build_download_inventory()
        log(f"Download phase: pods with results={list(inventory)}")
        for pod, dirs in inventory.items():
            # Deduplicate while preserving order
            unique_dirs = list(dict.fromkeys(dirs))
            pod_tasks = [
                t
                for t, st in self.state.tasks.items()
                if st.pod == pod and st.result_dir in unique_dirs
            ]
            if pod_tasks and all(local_task_done(t) for t in pod_tasks):
                for t in pod_tasks:
                    self.state.tasks[t].status = "downloaded"
                self.save()
                log(f"{pod}: already local for {len(pod_tasks)} tasks; skipping re-copy")
                continue
            download_selected_results(pod, unique_dirs)
            for task in pod_tasks:
                st = self.state.tasks[task]
                if not local_task_done(task):
                    raise RuntimeError(f"Local metrics missing after download: {task}")
                st.status = "downloaded"
            self.save()
        missing = [t for t in TASKS if not local_task_done(t)]
        if missing:
            raise RuntimeError(f"Download incomplete; missing local metrics for {len(missing)} tasks")
        for t in TASKS:
            self.state.tasks[t].status = "downloaded"
        self.save()
        log(f"Download complete: {len(TASKS)}/{len(TASKS)} local metrics present")

    def run_phase(self) -> None:
        log(f"Starting run phase: {self.summary()}")
        while True:
            for t, s in self.state.tasks.items():
                if s.status not in {"downloaded"} and local_task_done(t):
                    s.status = "downloaded"
            self.save()

            self.poll_running()
            self.reconcile_remote_results()
            self.assign_pending()
            counts = self.summary()
            running_slots = sum(len(v) for v in self.running_tasks_by_pod().values())
            log(
                f"status={counts} retries={len(self.state.retries)} "
                f"running_slots={running_slots}/{MAX_TASKS_PER_POD * len(PODS)}"
            )
            if counts["pending"] == 0 and counts["running"] == 0:
                break
            time.sleep(self.poll_seconds)

        failed = [t for t, s in self.state.tasks.items() if s.status == "failed"]
        if failed:
            for t in failed:
                log(f"FAILED: {t} ({self.state.tasks[t].error})")
            raise RuntimeError(f"Run phase finished with {len(failed)} failures")
        log(f"Run phase finished: remote_done/downloaded inventory ready ({self.summary()})")

    def run_queue(self) -> int:
        try:
            self.run_phase()
            self.wait_for_quiescence()
            self.download_all()
        except Exception as e:
            log(f"ERROR queue: {e}")
            failed = [t for t, s in self.state.tasks.items() if s.status == "failed"]
            log(f"Queue stopped: summary={self.summary()} failed={failed}")
            return 1
        counts = self.summary()
        log(f"Queue finished: {counts}")
        if counts.get("downloaded", 0) != len(TASKS):
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
    p.add_argument(
        "--download-only",
        action="store_true",
        help="Skip run phase; require remote_done inventory then download",
    )
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--max-attempts", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    local_output_root().mkdir(parents=True, exist_ok=True)
    queue_dir()
    zip_path = Path(args.zip_path)
    orch = Orchestrator(
        zip_path=zip_path,
        max_attempts=args.max_attempts,
        poll_seconds=args.poll_seconds,
    )
    if args.download_only:
        orch._setup_allowed = False
        try:
            orch.wait_for_quiescence()
            orch.download_all()
            return 0
        except Exception as e:
            log(f"ERROR download-only: {e}")
            return 1
    if not args.queue_only:
        orch.setup_all()
    if args.setup_only:
        return 0
    # After setup, allow first assign to ensure_pod_setup for READY check, but
    # upload_and_setup_pod is only needed if a pod is missing the repo.
    return orch.run_queue()


if __name__ == "__main__":
    sys.exit(main())
