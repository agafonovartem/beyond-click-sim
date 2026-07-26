"""Shared helpers for the detached cold-start ablation pod grid."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


NAMESPACE = "a-ashabokov"
PODS = [f"agent-exp-policy-ranking-{i}" for i in range(8)]
REMOTE_REPO = "/home/jovyan/beyond-click-sim"
REMOTE_ZIP = "/home/jovyan/beyond-click-sim-code.zip"
ABLATION_REL = Path("outputs") / "ablation"
MAX_TASKS_PER_POD = 4
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_POLL_SECONDS = 30.0

ABLATION_METHODS = [
    "llm_preference_listwise_ranking_vllm_qwen36_27b_item_stats_sumhistory_full",
    "llm_preference_listwise_ranking_vllm_qwen36_27b_item_stats_sumcandidate_full",
    "llm_preference_listwise_ranking_vllm_qwen36_27b_item_stats_sumboth_full",
    "agent4rec_preference_listwise_ranking_vllm_qwen36_27b_traits_full",
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

TASTE_CACHE_FILES = [
    "ml-1m_v1_seed0_gpt-4o-mini_agent4rec_modify_v1.jsonl",
]

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TIMESTAMP_RE = re.compile(r"^\d{8}T\d{6}Z$")


@dataclass(frozen=True)
class MethodRequirements:
    method: str
    requires_item_stats_tasks: bool
    required_columns: tuple[str, ...]
    required_artifacts: tuple[str, ...]
    requires_taste_cache: bool = False


@dataclass(frozen=True)
class GridJob:
    job_id: str
    method: str
    task: str
    required_artifacts: tuple[str, ...]
    required_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "method": self.method,
            "task": self.task,
            "required_artifacts": list(self.required_artifacts),
            "required_columns": list(self.required_columns),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GridJob":
        return cls(
            job_id=str(raw["job_id"]),
            method=str(raw["method"]),
            task=str(raw["task"]),
            required_artifacts=tuple(raw["required_artifacts"]),
            required_columns=tuple(raw.get("required_columns", ())),
        )


@dataclass
class JobState:
    status: str = "pending"  # pending|running|retry_wait|completed|failed
    attempts: int = 0
    run_token: str | None = None
    result_dir: str | None = None  # leaf under outputs/ablation/<method>/
    tmux_session: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    exit_code: int | None = None
    log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "JobState":
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})


def repo_root() -> Path:
    return REPO_ROOT


def ablation_root() -> Path:
    return repo_root() / ABLATION_REL


def grids_root() -> Path:
    return ablation_root() / "_grids"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def assert_safe_name(name: str, *, label: str = "name") -> str:
    if not name or not _SAFE_NAME_RE.fullmatch(name):
        raise ValueError(f"unsafe {label}: {name!r}")
    return name


def method_requires_item_stats_tasks(method: str) -> bool:
    """True when the method requires `_item_stats` task columns."""
    return method.startswith("agent4rec_") or "_item_stats_" in method


def method_requires_taste_cache(method: str) -> bool:
    return "_traits_taste_" in method


def required_artifacts_for_method(method: str) -> tuple[str, ...]:
    if method.startswith(
        (
            "llm_preference_listwise_ranking_",
            "agent4rec_preference_listwise_ranking_",
        )
    ):
        return ("manifest.json", "metrics_ranking.json")
    if method.startswith(("llm_yes_no_", "agent4rec_yes_no_")):
        return ("manifest.json", "metrics.json", "metrics_ranking.json")
    return ("manifest.json", "metrics.json")


def required_columns_for_method(method: str) -> tuple[str, ...]:
    if method.startswith("agent4rec_"):
        return ("item_rating_mean",)
    if "_item_stats_" in method:
        return ("item_rating_mean", "item_rating_count")
    return ()


def method_requirements(method: str) -> MethodRequirements:
    return MethodRequirements(
        method=method,
        requires_item_stats_tasks=method_requires_item_stats_tasks(method),
        required_columns=required_columns_for_method(method),
        required_artifacts=required_artifacts_for_method(method),
        requires_taste_cache=method_requires_taste_cache(method),
    )


def canonical_item_stats_task_names() -> list[str]:
    from runners.in_distribution.cold_start.task_builders import cold_start_task_name

    return [
        cold_start_task_name(
            "ml-1m",
            k,
            m,
            "uniform",
            sampling_seed,
            use_item_rating_stats=True,
        )
        for k in (1, 3, 5)
        for m in (1, 2, 3, 9, 19)
        for sampling_seed in (0, 1, 2)
    ]


def task_names_for_method(method: str) -> list[str]:
    """Canonical 45-task grid with the correct plain vs item_stats family."""
    from runners.in_distribution.cold_start.task_builders import cold_start_task_name

    use_item_rating_stats = method_requires_item_stats_tasks(method)
    return [
        cold_start_task_name(
            "ml-1m",
            k,
            m,
            "uniform",
            sampling_seed,
            use_item_rating_stats=use_item_rating_stats,
        )
        for k in (1, 3, 5)
        for m in (1, 2, 3, 9, 19)
        for sampling_seed in (0, 1, 2)
    ]


def validate_methods(methods: Iterable[str]) -> list[str]:
    from runners.in_distribution.cold_start.methods import METHOD_RUNNERS

    methods = list(methods)
    unknown = [m for m in methods if m not in METHOD_RUNNERS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    return methods


def validate_tasks(tasks: Iterable[str]) -> list[str]:
    from runners.in_distribution.cold_start.task_builders import TASK_BUILDERS

    tasks = list(tasks)
    unknown = [t for t in tasks if t not in TASK_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    return tasks


def validate_method_task_compatibility(method: str, task: str) -> None:
    reqs = method_requirements(method)
    is_item_stats_task = "_item_stats_" in task or task.startswith("ml-1m_item_stats_")
    if reqs.requires_item_stats_tasks and not is_item_stats_task:
        raise ValueError(
            f"Method {method} requires item_stats tasks; got plain task {task}"
        )
    if (not reqs.requires_item_stats_tasks) and is_item_stats_task:
        # Allowed but unusual; for ablation LLM item_stats methods this won't fire.
        pass


def make_grid_id(methods: list[str], *, seed: str = "ablation") -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(
        ("|".join(methods) + "|" + seed).encode("utf-8")
    ).hexdigest()[:8]
    return f"{seed}_{stamp}_{digest}"


def make_job_id(grid_id: str, method: str, task: str) -> str:
    digest = hashlib.sha256(
        f"{grid_id}|{method}|{task}".encode("utf-8")
    ).hexdigest()[:12]
    return f"j{digest}"


def make_run_token(grid_id: str, job_id: str, attempt: int) -> str:
    assert_safe_name(grid_id, label="grid_id")
    assert_safe_name(job_id, label="job_id")
    if attempt < 1:
        raise ValueError(f"attempt must be >= 1, got {attempt}")
    # Keep short for path length; uniqueness is grid+job+attempt.
    grid_short = hashlib.sha256(grid_id.encode("utf-8")).hexdigest()[:8]
    return f"g{grid_short}_{job_id}_a{attempt}"


def build_jobs(methods: list[str], tasks: list[str] | None = None) -> list[GridJob]:
    methods = validate_methods(methods)
    jobs: list[GridJob] = []
    pairs: set[tuple[str, str]] = set()
    for method in methods:
        method_tasks = tasks if tasks is not None else task_names_for_method(method)
        method_tasks = validate_tasks(method_tasks)
        reqs = method_requirements(method)
        for task in method_tasks:
            validate_method_task_compatibility(method, task)
            pair = (method, task)
            if pair in pairs:
                raise ValueError(f"Duplicate method/task pair: {pair}")
            pairs.add(pair)
            # provisional job_id; rewritten after grid_id is known
            jobs.append(
                GridJob(
                    job_id="pending",
                    method=method,
                    task=task,
                    required_artifacts=reqs.required_artifacts,
                    required_columns=reqs.required_columns,
                )
            )
    return jobs


def assign_jobs_to_pods(
    jobs: list[GridJob],
    *,
    grid_id: str,
    pods: list[str] = PODS,
) -> dict[str, list[GridJob]]:
    """Assign jobs with balanced pod sizes and per-pod method mix.

    Jobs are ordered by interleaved (task_index, method), then each job is
    placed on the least-loaded pod, breaking ties by the fewest copies of that
    method already on the pod, then by pod index.
    """
    if not pods:
        raise ValueError("pods must be non-empty")
    by_method: dict[str, list[GridJob]] = {}
    for job in jobs:
        by_method.setdefault(job.method, []).append(job)
    methods = list(by_method)
    max_len = max((len(by_method[m]) for m in methods), default=0)
    ordered: list[GridJob] = []
    for task_idx in range(max_len):
        for method in methods:
            method_jobs = by_method[method]
            if task_idx < len(method_jobs):
                ordered.append(method_jobs[task_idx])

    assigned: dict[str, list[GridJob]] = {pod: [] for pod in pods}
    method_counts: dict[str, dict[str, int]] = {pod: {} for pod in pods}
    finalized: list[GridJob] = []
    for job in ordered:
        job_id = make_job_id(grid_id, job.method, job.task)
        final = GridJob(
            job_id=job_id,
            method=job.method,
            task=job.task,
            required_artifacts=job.required_artifacts,
            required_columns=job.required_columns,
        )
        finalized.append(final)

        def pod_key(pod: str) -> tuple[int, int, int]:
            return (
                len(assigned[pod]),
                method_counts[pod].get(final.method, 0),
                pods.index(pod),
            )

        pod = min(pods, key=pod_key)
        assigned[pod].append(final)
        method_counts[pod][final.method] = method_counts[pod].get(final.method, 0) + 1

    pair_set = {(j.method, j.task) for j in finalized}
    if len(pair_set) != len(finalized):
        raise RuntimeError("job pair uniqueness check failed")
    job_ids = [j.job_id for j in finalized]
    if len(set(job_ids)) != len(job_ids):
        raise RuntimeError("job_id uniqueness check failed")
    owned: dict[tuple[str, str], str] = {}
    for pod, pod_jobs in assigned.items():
        for job in pod_jobs:
            key = (job.method, job.task)
            if key in owned:
                raise RuntimeError(f"pair {key} assigned to both {owned[key]} and {pod}")
            owned[key] = pod
    return assigned


def build_ablation_grid(
    methods: list[str] | None = None,
    *,
    pods: list[str] = PODS,
    grid_id: str | None = None,
) -> dict[str, Any]:
    methods = list(methods or ABLATION_METHODS)
    methods = validate_methods(methods)
    grid_id = grid_id or make_grid_id(methods)
    assert_safe_name(grid_id, label="grid_id")

    jobs = build_jobs(methods)
    # Force item_stats Cartesian for this ablation: every method gets the same 45.
    tasks = canonical_item_stats_task_names()
    validate_tasks(tasks)
    for method in methods:
        if not method_requires_item_stats_tasks(method):
            raise ValueError(
                f"Ablation grid requires item_stats-capable methods; got {method}"
            )
    jobs = build_jobs(methods, tasks=tasks)
    assigned = assign_jobs_to_pods(jobs, grid_id=grid_id, pods=pods)
    all_jobs = [job for pod in pods for job in assigned[pod]]
    if len(all_jobs) != len(methods) * len(tasks):
        raise RuntimeError(
            f"expected {len(methods) * len(tasks)} jobs, got {len(all_jobs)}"
        )

    return {
        "grid_id": grid_id,
        "created_at": now_iso(),
        "methods": methods,
        "tasks": tasks,
        "pods": list(pods),
        "max_tasks_per_pod": MAX_TASKS_PER_POD,
        "max_attempts": DEFAULT_MAX_ATTEMPTS,
        "ablation_rel": ABLATION_REL.as_posix(),
        "job_count": len(all_jobs),
        "jobs": [job.to_dict() for job in all_jobs],
        "pods_jobs": {
            pod: [job.to_dict() for job in assigned[pod]] for pod in pods
        },
    }


def write_json_atomic(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def has_required_artifacts(path: Path, artifacts: Iterable[str]) -> bool:
    return path.is_dir() and all((path / name).exists() for name in artifacts)


def find_result_dir_for_token(
    method_root: Path,
    *,
    task: str,
    method: str,
    run_token: str,
) -> Path | None:
    pattern = f"*_{run_token}_{task}_{method}"
    hits = [p for p in method_root.glob(pattern) if p.is_dir()]
    if not hits:
        return None
    hits.sort(key=lambda p: p.name)
    return hits[-1]


def parse_leaf_identity(leaf: str, *, method: str, task: str) -> bool:
    """Return True if leaf ends with _{task}_{method} and has a timestamp prefix."""
    suffix = f"_{task}_{method}"
    if not leaf.endswith(suffix):
        return False
    body = leaf[: -len(suffix)]
    parts = body.split("_", 1)
    if not parts or not _TIMESTAMP_RE.fullmatch(parts[0]):
        return False
    return True


def session_name_for_job(grid_id: str, job_id: str) -> str:
    digest = hashlib.sha256(f"{grid_id}|{job_id}".encode("utf-8")).hexdigest()[:16]
    return f"csj_{digest}"


def worker_session_name(grid_id: str, pod: str) -> str:
    digest = hashlib.sha256(f"{grid_id}|{pod}".encode("utf-8")).hexdigest()[:12]
    return f"csw_{digest}"


def method_output_rel(method: str) -> Path:
    return ABLATION_REL / method


def remote_method_output(method: str) -> str:
    return f"{REMOTE_REPO}/{method_output_rel(method).as_posix()}"


def local_method_output(method: str) -> Path:
    return ablation_root() / method


def copytree_if_needed(src: Path, dest: Path) -> None:
    if dest.exists():
        if dest.is_dir() and any(dest.iterdir()):
            # Refuse overwrite of a different existing tree.
            raise FileExistsError(f"Refusing to overwrite existing directory: {dest}")
        if dest.is_file():
            raise FileExistsError(f"Refusing to overwrite existing file: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, dirs_exist_ok=False)
