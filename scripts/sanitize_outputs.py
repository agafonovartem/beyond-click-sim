#!/usr/bin/env python3
"""Quarantine broken run directories that would corrupt result aggregation.

Aggregation loads runs by globbing a metrics file and keeping the most recent
run per ``(task, method)`` cell. Three things genuinely stop a run directory
from contributing a valid measurement, and only those are acted on by default:

  * the metrics file is truncated / unparseable -> ``read_json`` raises and the
    whole collection step dies.
  * the metrics file lacks ``task`` or ``method`` -> the run lands in a
    ``(None, None)`` cell and silently pollutes a group-by.
  * The run aborted before writing any metrics -> invisible to the loader, but
    it still clutters the tree and makes manual seed accounting wrong.

Everything else that is merely *suspicious* -- a directory name that disagrees
with the recorded task/method, incomplete scoring, missing artifacts, duplicate
cells, an undefined main metric, predictions that were never scored -- is
reported but left alone unless you opt in. Dropping a real measurement is a
research call, not a cleanup call.

There is no single metrics filename. Runners write one or both of:

  * ``metrics.json``          -- pointwise runners (interaction / regression /
    preference prediction) and the yes/no cold-start methods.
  * ``metrics_ranking.json``  -- every ``*_ranking`` method. The listwise
    cold-start methods write ONLY this one, by design; see
    ``required_completion_files`` in
    ``runners/in_distribution/cold_start/run.py``.

Either file carries ``task``/``method``, so either can establish identity. A
third file, ``policy_metrics.json``, is a *sidecar*: it has ``task`` but no
``method`` and never appears on its own, so it is not an identity source.

Safety model:
  * Dry run by default. Nothing moves until you pass ``--apply``.
  * ``--apply`` MOVES directories into ``<root>/_quarantine/`` so the operation
    is reversible. Real deletion requires ``--apply --delete``, and deletion
    refuses any directory holding a subdirectory or a ``.parquet``.
  * A directory that *contains* runs with metrics is a container, not an
    aborted run. It is reported and never removed, whatever flags are passed --
    moving it would take every valid run underneath it along.
  * A directory holding ``predictions.parquet`` but no metrics had its
    expensive part done and only scoring left; it is advisory unless you pass
    ``--drop-unscored``.
  * Directories whose name starts with ``_`` are never touched (this is what
    protects ``_task_artifacts`` and ``_quarantine`` itself).

Note that a name/metrics disagreement is usually benign. The pod grid *inserts*
a run token between the timestamp and the task
(``{timestamp}_{run_token}_{task}_{method}``), so those names are recognised
too. The loaders key off the metrics file rather than the path, so use
``--strict-identity`` only if you specifically rely on directory names.

Examples:
    # See what is broken, change nothing:
    scripts/sanitize_outputs.py outputs/in_distribution --recursive

    # Quarantine the broken dirs (reversible):
    scripts/sanitize_outputs.py outputs/in_distribution --recursive --apply

    # Also collapse duplicate cells down to the newest run:
    scripts/sanitize_outputs.py outputs/in_distribution --recursive --dedupe --apply

    # Put back everything that was quarantined in a root:
    scripts/sanitize_outputs.py outputs/in_distribution/policy_ranking_agreement --restore
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

QUARANTINE_DIRNAME = "_quarantine"
RUN_DIR_RE = re.compile(r"^(?P<timestamp>\d{8}T\d{6}Z)_(?P<rest>.+)$")

# Metrics files that carry both ``task`` and ``method``, in precedence order.
# ``policy_metrics.json`` is deliberately absent: it has ``task`` but no
# ``method``, and never appears without ``metrics.json`` beside it.
IDENTITY_METRICS_FILES = ("metrics.json", "metrics_ranking.json")
PREDICTIONS_FILENAME = "predictions.parquet"

# The pod grid inserts a token between the timestamp and the task; see
# ``make_output_dir`` in runners/in_distribution/cold_start/run.py and
# ``make_run_token`` in scripts/cold_start_pod_grid_lib.py. Mirrors
# ``_RUN_TOKEN_RE`` there -- kept local so this stays a stdlib-only script.
RUN_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# --- task-name families, for the --coverage report ------------------------
# policy_ranking, as produced by the policy_ranking_agreement runners.
POLICY_RANKING_TASK_RE = re.compile(
    r"^(?P<dataset>ml-1m|steam)_policy_ranking"
    r"(?:_(?P<target>preference))?"
    r"(?:_eval_users(?P<eval_users>\d+))?"
    r"_seed(?P<seed>\d+)$"
)
# Current cold-start grid. Mirrors ``cold_start_task_name`` in
# runners/in_distribution/cold_start/task_builders.py, and the pattern asserted
# in tests/test_cold_start_task_registry.py.
COLD_LIKE_TASK_RE = re.compile(
    r"^(?P<dataset>.+?)_cold_(?P<target>like|interact)_k(?P<k>\d+)_c(?P<c>\d+)_eval(?P<eval_users>\d+)"
    r"_m(?P<m>\d+)_(?P<strategy>negu|negpop)_split(?P<split>\d+)_sample(?P<sample>\d+)$"
)
# Older cold-start generation; same shape the inspect_cold_start notebook parses.
COLD_START_TASK_RE = re.compile(
    r"^(?P<dataset>.+?)_cold_start_k(?P<k>\d+)_cap(?P<cap>\d+)"
    r"(?:_eval_users(?P<eval_users>\d+)_cg(?P<groups>\d+))?"
    r"_m(?P<m>\d+)_seed(?P<seed>\d+)$"
)
POLICY_RANKING_SEEDS = tuple(range(10))
# Mirrors SAMPLING_SEEDS in runners/in_distribution/cold_start/task_builders.py.
COLD_START_SAMPLING_SEEDS = (0, 1, 2)

# Categories acted on by default: the run cannot contribute a valid measurement
# and holds nothing worth keeping.
FATAL = ("no_metrics", "unparseable_metrics", "missing_identity")
# Categories reported by default, quarantined only when the matching flag is set.
OPT_IN = (
    "identity_mismatch",
    "missing_artifacts",
    "incomplete_scoring",
    "undefined_metric",
    "unscored_predictions",
    "duplicate",
)
# Structural facts, never defects. No flag can promote these to removal.
INFORMATIONAL = ("container", "unparseable_sibling")

REASON_HELP = {
    "container": "holds runs with metrics -- a container, never removed",
    "unparseable_sibling": "one metrics file is corrupt but another is readable",
    "no_metrics": "aborted before writing metrics; nothing but logs left behind",
    "unparseable_metrics": "no readable metrics file (breaks the loader)",
    "missing_identity": "metrics file lacks task/method",
    "identity_mismatch": "metrics disagree with the directory name (often just a naming quirk)",
    "missing_artifacts": "manifest.json or predictions.parquet missing",
    "incomplete_scoring": "scored fraction below the required threshold",
    "undefined_metric": "main metric is null/absent",
    "unscored_predictions": "predictions written but never scored (re-scorable)",
    "duplicate": "superseded by a newer run of the same (task, method)",
}


@dataclass(frozen=True)
class TaskAxes:
    """A task name decomposed into a grouping cell plus a replicate axis."""

    dataset: str
    cell: tuple[tuple[str, str], ...]
    replicate: int
    replicate_name: str
    expected: tuple[int, ...]


@dataclass
class Run:
    path: Path
    timestamp: str = ""
    task: str | None = None
    method: str | None = None
    protocol: str | None = None
    metrics_file: str | None = None
    scored_ratio: float | None = None
    scored_split: str | None = None
    container_runs: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def cell(self) -> tuple[str | None, str | None]:
        return (self.task, self.method)

    @property
    def axes(self) -> TaskAxes | None:
        return parse_task_axes(self.task)


def parse_task_axes(task: str | None) -> TaskAxes | None:
    """Decompose a task name into (dataset, grouping cell, replicate)."""
    if not task:
        return None

    m = COLD_LIKE_TASK_RE.match(task)
    if m:
        return TaskAxes(
            dataset=m.group("dataset"),
            cell=(
                ("target", m.group("target")),
                ("k", m.group("k")),
                ("m", m.group("m")),
                ("neg", m.group("strategy")),
            ),
            replicate=int(m.group("sample")),
            replicate_name="sample",
            expected=COLD_START_SAMPLING_SEEDS,
        )

    m = COLD_START_TASK_RE.match(task)
    if m:
        return TaskAxes(
            dataset=m.group("dataset"),
            cell=(("k", m.group("k")), ("m", m.group("m"))),
            replicate=int(m.group("seed")),
            replicate_name="seed",
            expected=COLD_START_SAMPLING_SEEDS,
        )

    m = POLICY_RANKING_TASK_RE.match(task)
    if m:
        return TaskAxes(
            dataset=m.group("dataset"),
            cell=(),
            replicate=int(m.group("seed")),
            replicate_name="seed",
            expected=POLICY_RANKING_SEEDS,
        )

    return None


def find_roots(paths: list[Path], recursive: bool) -> list[Path]:
    """Return directories that directly contain run directories."""
    roots: list[Path] = []
    for p in paths:
        if not p.is_dir():
            print(f"warning: not a directory, skipping: {p}", file=sys.stderr)
            continue
        if not recursive:
            roots.append(p)
            continue
        # A root is any directory holding at least one <timestamp>_... subdir.
        for candidate in [p, *(d for d in p.rglob("*") if d.is_dir())]:
            if candidate.name.startswith("_"):
                continue
            if any(
                child.is_dir() and RUN_DIR_RE.match(child.name)
                for child in candidate.iterdir()
            ):
                roots.append(candidate)
    # Deduplicate, preserve order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for r in roots:
        rp = r.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(r)
    return unique


def age_seconds(path: Path) -> float:
    """Seconds since this run directory was last written to.

    A run that is still executing has no metrics yet, so it looks exactly like
    an aborted one. The only thing distinguishing them is that the live run is
    still being written, hence this check against the directory and its files.
    """
    newest = path.stat().st_mtime
    try:
        for child in path.iterdir():
            newest = max(newest, child.stat().st_mtime)
    except OSError:
        pass
    return time.time() - newest


def descendant_run_count(path: Path) -> int:
    """How many directories *below* this one hold a metrics file.

    A non-zero count means this directory is a container of runs rather than a
    run that aborted, and moving it would carry every one of them along.
    """
    count = 0
    try:
        children = list(path.rglob("*"))
    except OSError:
        return 0
    for child in children:
        if not child.is_dir():
            continue
        if any(part.startswith("_") for part in child.relative_to(path).parts):
            continue
        if any((child / name).is_file() for name in IDENTITY_METRICS_FILES):
            count += 1
    return count


def dir_size(path: Path) -> int:
    total = 0
    try:
        children = list(path.rglob("*"))
    except OSError:
        return 0
    for child in children:
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def is_trivially_deletable(path: Path) -> bool:
    """True only for directories holding nothing but loose, cheap files.

    ``--delete`` is irreversible, so it is restricted to what can be recreated
    for free: no subdirectories (which could hide whole runs) and no parquet
    (which is the expensive output of a run).
    """
    try:
        children = list(path.rglob("*"))
    except OSError:
        return False
    for child in children:
        if child.is_dir():
            return False
        if child.suffix == ".parquet":
            return False
    return True


def resolve_dotted(payload: dict, dotted: str) -> object:
    value: object = payload
    for part in dotted.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def scored_coverage(metrics: dict) -> tuple[float | None, str | None]:
    """Worst scored fraction across splits, and the split that produced it.

    Pointwise runners record ``requested_rows``/``scored_rows`` at the top
    level. Ranking runners record them per split under ``coverage``, alongside
    a precomputed ``scored_fraction``. Taking the minimum means an under-scored
    ``val`` is not hidden behind a clean ``test``.
    """
    requested = metrics.get("requested_rows")
    scored = metrics.get("scored_rows")
    if (
        isinstance(requested, (int, float))
        and requested
        and isinstance(scored, (int, float))
    ):
        return scored / requested, None

    coverage = metrics.get("coverage")
    if not isinstance(coverage, dict):
        return (None, None)

    worst_fraction: float | None = None
    worst_split: str | None = None
    for split, block in coverage.items():
        if not isinstance(block, dict):
            continue
        fraction = block.get("scored_fraction")
        if not isinstance(fraction, (int, float)):
            split_requested = block.get("requested_rows")
            split_scored = block.get("scored_rows")
            if (
                isinstance(split_requested, (int, float))
                and split_requested
                and isinstance(split_scored, (int, float))
            ):
                fraction = split_scored / split_requested
            else:
                continue
        fraction_value = float(fraction)
        if worst_fraction is None or fraction_value < worst_fraction:
            worst_fraction = fraction_value
            worst_split = str(split)
    if worst_fraction is None or worst_split is None:
        return (None, None)
    return (worst_fraction, worst_split)


def name_matches_identity(name: str, timestamp: str, task: str, method: str) -> bool:
    """Accept both ``{ts}_{task}_{method}`` and ``{ts}_{token}_{task}_{method}``."""
    suffix = f"_{task}_{method}"
    if name == f"{timestamp}{suffix}":
        return True
    prefix = f"{timestamp}_"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return False
    token = name[len(prefix) : -len(suffix)]
    return bool(token) and bool(RUN_TOKEN_RE.fullmatch(token))


def inspect_run(path: Path, *, min_scored_ratio: float | None) -> Run:
    run = Run(path=path)
    m = RUN_DIR_RE.match(path.name)
    if m:
        run.timestamp = m.group("timestamp")

    # Containment is checked first and unconditionally: whatever else is true
    # of this directory, it must never be moved out from over live runs.
    run.container_runs = descendant_run_count(path)
    if run.container_runs:
        run.reasons.append("container")

    parsed: list[tuple[str, dict]] = []
    unparseable: list[str] = []
    for name in IDENTITY_METRICS_FILES:
        metrics_path = path / name
        if not metrics_path.is_file():
            continue
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unparseable.append(name)
            continue
        if isinstance(payload, dict):
            parsed.append((name, payload))
        else:
            unparseable.append(name)

    identity = next(
        ((name, p) for name, p in parsed if p.get("task") and p.get("method")),
        None,
    )

    if identity is None:
        if run.container_runs:
            # A container has no metrics of its own; that is not a defect.
            return run
        if unparseable:
            run.reasons.append("unparseable_metrics")
        elif parsed:
            run.reasons.append("missing_identity")
        elif (path / PREDICTIONS_FILENAME).is_file():
            # The expensive half already ran. Scoring can be redone from this.
            run.reasons.append("unscored_predictions")
        else:
            run.reasons.append("no_metrics")
        return run

    run.metrics_file, metrics = identity
    if unparseable:
        # A readable sibling means the measurement survives; flag, do not doom.
        run.reasons.append("unparseable_sibling")

    run.task = metrics.get("task")
    run.method = metrics.get("method")
    # `protocol` is only written by the policy_ranking protocols; other runners
    # (interaction_prediction, regression_prediction, ...) legitimately omit it,
    # so identity is task+method only. Requiring protocol here would condemn
    # perfectly good results from those runners.
    run.protocol = metrics.get("protocol")

    # The loader trusts the metrics file over the path, so a disagreement means
    # the run would be attributed to a cell it does not belong to.
    if run.timestamp and not name_matches_identity(
        path.name, run.timestamp, str(run.task), str(run.method)
    ):
        run.reasons.append("identity_mismatch")

    if not (path / "manifest.json").exists() or not (path / PREDICTIONS_FILENAME).exists():
        run.reasons.append("missing_artifacts")

    run.scored_ratio, run.scored_split = scored_coverage(metrics)
    if (
        min_scored_ratio is not None
        and run.scored_ratio is not None
        and run.scored_ratio < min_scored_ratio
    ):
        run.reasons.append("incomplete_scoring")

    main_metric = str(metrics.get("main_metric") or "")
    value = resolve_dotted(metrics, main_metric) if main_metric else None
    if not isinstance(value, (int, float)):
        run.reasons.append("undefined_metric")

    return run


def mark_duplicates(runs: list[Run]) -> None:
    """Flag every run in a (task, method) cell except the newest."""
    by_cell: dict[tuple[str | None, str | None], list[Run]] = defaultdict(list)
    for run in runs:
        # Only runs that are otherwise loadable can shadow each other.
        if any(r in FATAL for r in run.reasons):
            continue
        by_cell[run.cell].append(run)
    for cell, group in by_cell.items():
        if cell == (None, None) or len(group) < 2:
            continue
        group.sort(key=lambda r: r.timestamp)
        for run in group[:-1]:
            run.reasons.append("duplicate")


def quarantine(run: Run, root: Path, *, delete: bool) -> str:
    """Move (or delete) one run directory. Returns what actually happened."""
    if not run.path.exists():
        # A nested root may already have been processed by an earlier pass.
        return "vanished"
    if delete:
        if not is_trivially_deletable(run.path):
            return "refused"
        shutil.rmtree(run.path)
        return "deleted"
    dest_dir = root / QUARANTINE_DIRNAME
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / run.path.name
    suffix = 1
    while dest.exists():
        dest = dest_dir / f"{run.path.name}__{suffix}"
        suffix += 1
    shutil.move(str(run.path), str(dest))
    return "quarantined"


def restore(root: Path, *, apply: bool) -> int:
    src_dir = root / QUARANTINE_DIRNAME
    if not src_dir.is_dir():
        print(f"  nothing quarantined in {root}")
        return 0
    entries = [d for d in src_dir.iterdir() if d.is_dir()]
    for d in entries:
        target = root / d.name
        if target.exists():
            print(f"  SKIP (already present): {d.name}")
            continue
        print(f"  restore: {d.name}")
        if apply:
            shutil.move(str(d), str(target))
    if apply and not any(src_dir.iterdir()):
        src_dir.rmdir()
    return len(entries)


def coverage_report(runs: list[Run], doomed: set[Path]) -> None:
    survivors = [r for r in runs if r.path not in doomed]
    by_group: dict[tuple[str, str], dict[tuple, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    meta: dict[tuple[str, str, tuple], TaskAxes] = {}
    for run in survivors:
        axes = run.axes
        if axes is None or run.method is None:
            continue
        by_group[(axes.dataset, run.method)][axes.cell].add(axes.replicate)
        meta[(axes.dataset, run.method, axes.cell)] = axes
    if not by_group:
        return

    print("\n  surviving replicate coverage:")
    for (dataset, method), cells in sorted(by_group.items()):
        print(f"    {dataset} / {method}")
        labels = {
            cell: (" ".join(f"{k}={v}" for k, v in cell) if cell else "(all)")
            for cell in cells
        }
        width = max(len(v) for v in labels.values())
        for cell, replicates in sorted(cells.items()):
            axes = meta[(dataset, method, cell)]
            present = sorted(replicates)
            missing = [s for s in axes.expected if s not in replicates]
            gap = f"  MISSING {missing}" if missing else ""
            print(
                f"      {labels[cell]:<{width}}  {axes.replicate_name} "
                f"n={len(present):<2} {present}{gap}"
            )


@dataclass
class RootResult:
    doomed: int = 0
    flagged: int = 0
    containers: int = 0
    protected: int = 0
    refused: int = 0
    vanished: int = 0
    freed_bytes: int = 0


def process_root(root: Path, args: argparse.Namespace) -> RootResult:
    print(f"\n=== {root} ===")
    entries = sorted(
        d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")
    )
    if not entries:
        print("  (no run directories)")
        return RootResult()

    # An in-flight run is indistinguishable from an aborted one except by age,
    # so anything still being written is left strictly alone.
    min_age = args.min_age_minutes * 60
    fresh = {d for d in entries if min_age > 0 and age_seconds(d) < min_age}
    entries = [d for d in entries if d not in fresh]

    inspected = [inspect_run(d, min_scored_ratio=args.min_scored_ratio) for d in entries]
    containers = [r for r in inspected if "container" in r.reasons]
    runs = [r for r in inspected if "container" not in r.reasons]
    if args.dedupe:
        mark_duplicates(runs)

    active = set(FATAL)
    if args.strict_identity:
        active.add("identity_mismatch")
    if args.require_artifacts:
        active.add("missing_artifacts")
    if args.min_scored_ratio is not None:
        active.add("incomplete_scoring")
    if args.drop_undefined_metric:
        active.add("undefined_metric")
    if args.drop_unscored:
        active.add("unscored_predictions")
    if args.dedupe:
        active.add("duplicate")
    # Belt and braces: informational reasons can never select a directory.
    active -= set(INFORMATIONAL)

    doomed: list[Run] = []
    flagged: list[Run] = []
    for run in runs:
        if not run.reasons:
            continue
        if any(r in active for r in run.reasons):
            doomed.append(run)
        else:
            flagged.append(run)

    counts: dict[str, int] = defaultdict(int)
    for run in inspected:
        for reason in run.reasons:
            counts[reason] += 1

    fatal_free = sum(1 for r in runs if not any(x in FATAL for x in r.reasons))
    advisory_only = sum(
        1 for r in runs if r.reasons and not any(x in FATAL for x in r.reasons)
    )
    result = RootResult(
        containers=len(containers),
        protected=sum(r.container_runs for r in containers),
    )
    print(f"  run directories : {len(runs) + len(fresh)}")
    print(f"  aggregation-safe: {fatal_free}  ({advisory_only} carrying advisories)")
    if containers:
        print(
            f"  containers      : {len(containers)}  [never removed] holding "
            f"{result.protected} run(s) with metrics"
        )
        for run in sorted(containers, key=lambda r: r.name):
            print(f"    [container] {run.container_runs:>3} run(s)  {run.name}")
    if fresh:
        print(
            f"  in progress     : {len(fresh)}  [skipped] modified within "
            f"{args.min_age_minutes} min -- still being written"
        )
        for d in sorted(fresh):
            print(f"    [in-progress] {d.name}")
    for reason in (*INFORMATIONAL, *FATAL, *OPT_IN):
        if reason == "container":
            continue
        if counts.get(reason):
            mark = "remove" if reason in active else "report"
            print(f"  {reason:<20} {counts[reason]:>4}  [{mark}] {REASON_HELP[reason]}")

    verb = "DELETING" if args.delete else "quarantining"
    if doomed:
        total = sum(dir_size(r.path) for r in doomed)
        result.freed_bytes = total
        print(f"\n  {verb} {len(doomed)}  ({human_bytes(total)}):")
        for run in sorted(doomed, key=lambda r: r.name):
            why = ",".join(r for r in run.reasons if r in active)
            print(f"    [{why}] {run.name}")
    if flagged:
        print("\n  flagged only (left in place, use the matching flag to remove):")
        for run in sorted(flagged, key=lambda r: r.name):
            why = ",".join(run.reasons)
            extra = ""
            if run.scored_ratio is not None and run.scored_ratio < 1.0:
                split = f"({run.scored_split})" if run.scored_split else ""
                extra = f" scored={run.scored_ratio:.4f}{split}"
            print(f"    [{why}]{extra} {run.name}")

    if args.apply:
        for run in doomed:
            outcome = quarantine(run, root, delete=args.delete)
            if outcome == "refused":
                result.refused += 1
                print(
                    f"    REFUSED --delete (holds a subdir or parquet; "
                    f"quarantine it instead): {run.name}"
                )
            elif outcome == "vanished":
                result.vanished += 1
                print(f"    SKIP (already gone): {run.name}")

    result.doomed = len(doomed)
    result.flagged = len(flagged)
    if args.coverage:
        coverage_report(runs, {r.path for r in doomed})

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quarantine broken run directories that would corrupt aggregation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Result root(s), e.g. outputs/in_distribution/policy_ranking_agreement",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search the given paths for every directory that holds run directories.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move things. Without this the script only reports.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help=(
            "With --apply, delete instead of moving to _quarantine/ (irreversible). "
            "Refuses any directory holding a subdirectory or a .parquet."
        ),
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Also remove older duplicate runs of the same (task, method), keeping the newest.",
    )
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="Also remove runs missing manifest.json or predictions.parquet.",
    )
    parser.add_argument(
        "--strict-identity",
        action="store_true",
        help=(
            "Also remove runs whose directory name is neither "
            "'<timestamp>_<task>_<method>' nor "
            "'<timestamp>_<run_token>_<task>_<method>'. Off by default: the "
            "loaders key off the metrics file rather than the path."
        ),
    )
    parser.add_argument(
        "--min-scored-ratio",
        type=float,
        default=None,
        metavar="R",
        help=(
            "Also remove runs whose scored fraction is below R (e.g. 0.99). Read "
            "from top-level scored_rows/requested_rows, or from the worst split "
            "in the coverage block."
        ),
    )
    parser.add_argument(
        "--drop-undefined-metric",
        action="store_true",
        help="Also remove runs whose main metric is null (e.g. tau undefined for K<3).",
    )
    parser.add_argument(
        "--drop-unscored",
        action="store_true",
        help=(
            "Also remove runs that wrote predictions.parquet but no metrics. Off "
            "by default: the expensive half already ran and they can be re-scored."
        ),
    )
    parser.add_argument(
        "--min-age-minutes",
        type=float,
        default=30.0,
        metavar="M",
        help=(
            "Never touch a run directory modified within the last M minutes, so a "
            "run that is still executing is not mistaken for an aborted one. "
            "Pass 0 to disable (only safe when nothing is running)."
        ),
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Print surviving replicate coverage per (dataset, method) after the cleanup.",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Move everything in _quarantine/ back and exit.",
    )
    args = parser.parse_args()
    if args.delete and not args.apply:
        parser.error("--delete requires --apply")
    if args.min_scored_ratio is not None and not 0.0 <= args.min_scored_ratio <= 1.0:
        parser.error("--min-scored-ratio must be between 0 and 1")
    return args


def main() -> int:
    args = parse_args()
    roots = find_roots(args.paths, args.recursive)
    if not roots:
        print("No result roots found.", file=sys.stderr)
        return 1

    if args.restore:
        for root in roots:
            print(f"\n=== {root} ===")
            restore(root, apply=args.apply)
        if not args.apply:
            print("\nDRY RUN -- nothing moved. Re-run with --apply.")
        return 0

    total = RootResult()
    for root in roots:
        result = process_root(root, args)
        total.doomed += result.doomed
        total.flagged += result.flagged
        total.containers += result.containers
        total.protected += result.protected
        total.refused += result.refused
        total.vanished += result.vanished
        total.freed_bytes += result.freed_bytes

    print(f"\n{'=' * 60}")
    if total.containers:
        print(
            f"{total.containers} container(s) left untouched, protecting "
            f"{total.protected} run(s) with metrics."
        )
    if args.apply:
        where = "deleted" if args.delete else f"moved to {QUARANTINE_DIRNAME}/"
        moved = total.doomed - total.refused - total.vanished
        print(
            f"{moved} directory(ies) {where} ({human_bytes(total.freed_bytes)}); "
            f"{total.flagged} flagged and left alone."
        )
        if total.refused:
            print(f"{total.refused} refused by --delete; quarantine them instead.")
        if not args.delete and moved:
            print("Undo with: --restore --apply")
    else:
        print("DRY RUN -- nothing changed.")
        print(
            f"{total.doomed} directory(ies) would be removed "
            f"({human_bytes(total.freed_bytes)}); {total.flagged} flagged only."
        )
        print("Re-run with --apply to quarantine them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
