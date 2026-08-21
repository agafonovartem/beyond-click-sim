#!/usr/bin/env python3
"""Pod-local detached worker for a cold-start ablation grid manifest.

Reads an immutable per-pod JSON manifest, runs up to N jobs concurrently via
per-job tmux sessions, persists atomic state, and retries failed jobs with a
fresh run token. Survives localhost disconnects.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.cold_start_pod_grid_lib import (  # noqa: E402
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_POLL_SECONDS,
    MAX_TASKS_PER_POD,
    REMOTE_REPO,
    GridJob,
    JobState,
    find_result_dir_for_token,
    has_required_artifacts,
    make_run_token,
    method_output_rel,
    now_iso,
    read_json,
    session_name_for_job,
    write_json_atomic,
)

# Raised well above the client default of 120s. Requests are not slow here, they are
# queued: one TP engine at --max-num-seqs=16 against a listwise job's 240 worker
# threads means most requests wait far longer than they execute, and a client timeout
# is recorded as an LLM error rather than as backlog.
VLLM_TIMEOUT_SECONDS = 1800


def log(msg: str) -> None:
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


def discover_llm_base_url() -> str:
    script = r"""
set -e
port=$(ps aux 2>/dev/null | grep '[r]un_litellm.sh' | awk '{for(i=1;i<=NF;i++) if($i~/run_litellm\.sh/){print $(i+1); exit}}')
if [ -n "$port" ]; then
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:${port}/v1/models" || echo 000)
  if [ "$code" = "200" ]; then echo "litellm:$port"; exit 0; fi
fi
if [ -f ~/litellm.log ]; then
  port=$(grep -Eo 'Uvicorn running on http://[^:]+:[0-9]+' ~/litellm.log | tail -1 | grep -Eo '[0-9]+$')
  if [ -n "$port" ]; then
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:${port}/v1/models" || echo 000)
    if [ "$code" = "200" ]; then echo "litellm:$port"; exit 0; fi
  fi
fi
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
    out = subprocess.check_output(["bash", "-lc", script], text=True).strip()
    kind, _, port = out.partition(":")
    if kind in {"litellm", "vllm"} and port.isdigit() and int(port) > 0:
        return f"http://127.0.0.1:{port}/v1"
    raise RuntimeError(f"no healthy LLM endpoint found ({out!r})")


def tmux_has_session(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def tmux_kill_session(name: str) -> None:
    subprocess.run(
        ["tmux", "kill-session", "-t", name],
        check=False,
        capture_output=True,
        text=True,
    )


def start_job_tmux(
    *,
    session: str,
    method: str,
    task: str,
    run_token: str,
    output_rel: Path,
    log_path: Path,
    base_url: str,
    repo: Path,
    vllm_timeout_seconds: int = VLLM_TIMEOUT_SECONDS,
) -> None:
    tmux_kill_session(session)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    inner = (
        f"cd {shlex.quote(str(repo))} && "
        f"source .venv/bin/activate && "
        f'export PYTHONPATH="{repo}/src:{repo}" && '
        f"export BEYOND_CLICK_SIM_VLLM_LOCAL_BASE_URL={shlex.quote(base_url)} && "
        # A tensor-parallel pod serves one engine at --max-num-seqs=16 while a listwise
        # job drives up to VLLM_MAX_WORKERS=240 threads, so requests queue deeply and
        # the 120s client default times them out — counted as LLM errors, not backlog.
        f"export BEYOND_CLICK_SIM_VLLM_LOCAL_TIMEOUT_SECONDS={shlex.quote(str(vllm_timeout_seconds))} && "
        f"mkdir -p {shlex.quote(str(output_rel))} && "
        f"python -m runners.in_distribution.cold_start.run "
        f"--methods {shlex.quote(method)} "
        f"--tasks {shlex.quote(task)} "
        f"--output-dir {shlex.quote(str(output_rel))} "
        f"--run-token {shlex.quote(run_token)} "
        f"--resume "
        f"> {shlex.quote(str(log_path))} 2>&1; "
        f"echo EXIT:$? >> {shlex.quote(str(log_path))}"
    )
    cmd = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        session,
        "bash",
        "-lc",
        inner,
    ]
    subprocess.run(cmd, check=True)


def read_exit_code(log_path: Path) -> int | None:
    if not log_path.is_file():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if line.startswith("EXIT:"):
            try:
                return int(line.split(":", 1)[1])
            except ValueError:
                return None
    return None


class PodWorker:
    def __init__(
        self,
        *,
        manifest_path: Path,
        state_path: Path,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        max_slots: int = MAX_TASKS_PER_POD,
        repo: Path = REPO_ROOT,
    ) -> None:
        self.manifest_path = manifest_path
        self.state_path = state_path
        self.poll_seconds = poll_seconds
        self.max_attempts = max_attempts
        self.max_slots = max_slots
        self.repo = repo
        self.manifest = read_json(manifest_path)
        self.grid_id = str(self.manifest["grid_id"])
        self.pod = str(self.manifest["pod"])
        self.jobs = [GridJob.from_dict(raw) for raw in self.manifest["jobs"]]
        self.jobs_by_id = {job.job_id: job for job in self.jobs}
        self.state = self._load_or_init_state()

    def _load_or_init_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            raw = read_json(self.state_path)
            if raw.get("grid_id") == self.grid_id and raw.get("pod") == self.pod:
                jobs = {
                    job_id: JobState.from_dict(info)
                    for job_id, info in raw.get("jobs", {}).items()
                    if job_id in self.jobs_by_id
                }
                for job in self.jobs:
                    jobs.setdefault(job.job_id, JobState())
                return {
                    "grid_id": self.grid_id,
                    "pod": self.pod,
                    "updated_at": now_iso(),
                    "phase": raw.get("phase", "running"),
                    "jobs": jobs,
                }
        jobs = {job.job_id: JobState() for job in self.jobs}
        return {
            "grid_id": self.grid_id,
            "pod": self.pod,
            "updated_at": now_iso(),
            "phase": "running",
            "jobs": jobs,
        }

    def save(self) -> None:
        self.state["updated_at"] = now_iso()
        payload = {
            "grid_id": self.state["grid_id"],
            "pod": self.state["pod"],
            "updated_at": self.state["updated_at"],
            "phase": self.state["phase"],
            "jobs": {
                job_id: st.to_dict() for job_id, st in self.state["jobs"].items()
            },
            "summary": self.summary(),
        }
        write_json_atomic(self.state_path, payload)

    def summary(self) -> dict[str, int]:
        counts = {
            "pending": 0,
            "running": 0,
            "retry_wait": 0,
            "completed": 0,
            "failed": 0,
        }
        for st in self.state["jobs"].values():
            counts[st.status] = counts.get(st.status, 0) + 1
        return counts

    def method_root(self, method: str) -> Path:
        return self.repo / method_output_rel(method)

    def control_dir(self) -> Path:
        return self.repo / "outputs" / "ablation" / "_grids" / self.grid_id / "pods" / self.pod

    def mark_completed(self, job: GridJob, st: JobState, result_dir: Path) -> None:
        st.status = "completed"
        st.result_dir = result_dir.name
        st.finished_at = now_iso()
        st.error = None
        if st.exit_code is None:
            st.exit_code = 0
        log(f"COMPLETED {job.job_id} {job.method} {job.task} -> {result_dir.name}")

    def reconcile_job(self, job: GridJob, st: JobState) -> None:
        method_root = self.method_root(job.method)
        # Prefer exact run_token match from current attempt.
        if st.run_token:
            found = find_result_dir_for_token(
                method_root,
                task=job.task,
                method=job.method,
                run_token=st.run_token,
            )
            if found and has_required_artifacts(found, job.required_artifacts):
                self.mark_completed(job, st, found)
                return

        # Broader resume: any complete dir for this pair under the method root.
        pattern = f"*_{job.task}_{job.method}"
        complete = [
            p
            for p in method_root.glob(pattern)
            if has_required_artifacts(p, job.required_artifacts)
        ]
        if complete:
            complete.sort(key=lambda p: p.name)
            chosen = complete[-1]
            self.mark_completed(job, st, chosen)
            return

        if st.status == "running" and st.tmux_session:
            if tmux_has_session(st.tmux_session):
                return
            # Session ended without completion.
            exit_code = read_exit_code(Path(st.log_path)) if st.log_path else None
            st.exit_code = exit_code
            st.attempts = max(st.attempts, 1)
            if st.attempts >= self.max_attempts:
                st.status = "failed"
                st.error = f"tmux_exited_without_metrics exit={exit_code}"
                st.finished_at = now_iso()
                log(f"FAILED {job.job_id}: {st.error}")
            else:
                st.status = "pending"
                st.error = f"tmux_exited_without_metrics exit={exit_code}; will retry"
                st.tmux_session = None
                st.run_token = None
                log(f"RETRY {job.job_id}: {st.error}")

    def running_count(self) -> int:
        return sum(1 for st in self.state["jobs"].values() if st.status == "running")

    def start_pending(self, base_url: str) -> None:
        while self.running_count() < self.max_slots:
            pending = [
                job
                for job in self.jobs
                if self.state["jobs"][job.job_id].status in {"pending", "retry_wait"}
            ]
            if not pending:
                return
            job = pending[0]
            st = self.state["jobs"][job.job_id]
            # Reconcile first in case artifacts appeared.
            self.reconcile_job(job, st)
            if st.status == "completed":
                continue
            attempt = st.attempts + 1
            run_token = make_run_token(self.grid_id, job.job_id, attempt)
            session = session_name_for_job(self.grid_id, job.job_id)
            log_path = self.control_dir() / "logs" / f"{job.job_id}_a{attempt}.log"
            output_rel = method_output_rel(job.method)
            try:
                start_job_tmux(
                    session=session,
                    method=job.method,
                    task=job.task,
                    run_token=run_token,
                    output_rel=output_rel,
                    log_path=log_path,
                    base_url=base_url,
                    repo=self.repo,
                )
            except Exception as exc:  # noqa: BLE001
                st.attempts = attempt
                st.error = f"start_failed: {exc}"
                if st.attempts >= self.max_attempts:
                    st.status = "failed"
                    st.finished_at = now_iso()
                    log(f"FAILED start {job.job_id}: {exc}")
                else:
                    st.status = "pending"
                    log(f"START ERROR {job.job_id}: {exc}")
                self.save()
                return
            st.status = "running"
            st.attempts = attempt
            st.run_token = run_token
            st.tmux_session = session
            st.started_at = now_iso()
            st.finished_at = None
            st.error = None
            st.exit_code = None
            st.log_path = str(log_path)
            st.result_dir = None
            log(
                f"STARTED {job.job_id} attempt={attempt} session={session} "
                f"method={job.method} task={job.task}"
            )
            self.save()

    def poll_once(self) -> None:
        for job in self.jobs:
            st = self.state["jobs"][job.job_id]
            if st.status in {"completed", "failed"}:
                continue
            self.reconcile_job(job, st)

        try:
            base_url = discover_llm_base_url()
        except Exception as exc:  # noqa: BLE001
            log(f"LLM endpoint unavailable: {exc}")
            self.save()
            return

        self.start_pending(base_url)
        counts = self.summary()
        if (
            counts["pending"] == 0
            and counts["running"] == 0
            and counts["retry_wait"] == 0
        ):
            self.state["phase"] = "terminal"
        else:
            self.state["phase"] = "running"
        self.save()
        log(f"status={counts} phase={self.state['phase']}")

    def run_forever(self) -> int:
        log(
            f"Worker start grid={self.grid_id} pod={self.pod} "
            f"jobs={len(self.jobs)} max_slots={self.max_slots}"
        )
        self.control_dir().mkdir(parents=True, exist_ok=True)
        self.save()
        while True:
            self.poll_once()
            counts = self.summary()
            if (
                counts["pending"] == 0
                and counts["running"] == 0
                and counts["retry_wait"] == 0
            ):
                log(f"Worker terminal: {counts}")
                return 0 if counts["failed"] == 0 else 1
            time.sleep(self.poll_seconds)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="Path to per-pod manifest JSON")
    p.add_argument("--state", required=True, help="Path to writable pod state JSON")
    p.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    p.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    p.add_argument("--max-slots", type=int, default=MAX_TASKS_PER_POD)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # Ensure we run from the repo root on the pod.
    os.chdir(REPO_ROOT)
    worker = PodWorker(
        manifest_path=Path(args.manifest),
        state_path=Path(args.state),
        poll_seconds=args.poll_seconds,
        max_attempts=args.max_attempts,
        max_slots=args.max_slots,
        repo=REPO_ROOT,
    )
    return worker.run_forever()


if __name__ == "__main__":
    sys.exit(main())
