from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import sanitize_outputs as sanitize  # noqa: E402


TASK = "ml-1m_item_stats_cold_like_k1_c5_eval1000_m1_negu_split0_sample0"
METHOD = "agent4rec_preference_listwise_ranking_vllm_qwen36_35b_a3b_traits_taste_gpt4o_mini_sumboth_full"
TS = "20260725T221213Z"


def ranking_metrics(
    *,
    task: str = TASK,
    method: str = METHOD,
    val_fraction: float = 1.0,
    test_fraction: float = 1.0,
) -> dict:
    """A metrics_ranking.json shaped like the listwise cold-start runner writes."""
    return {
        "task": task,
        "method": method,
        "protocol": "direct_listwise_preference_ranking",
        "main_metric": "test_failure_as_zero_group.macro_by_user_group_mean.ndcg@1",
        "test_failure_as_zero_group": {"macro_by_user_group_mean": {"ndcg@1": 0.6758}},
        "coverage": {
            "val": {"requested_rows": 6030, "scored_rows": 6030, "scored_fraction": val_fraction},
            "test": {"requested_rows": 10000, "scored_rows": 10000, "scored_fraction": test_fraction},
        },
    }


def pointwise_metrics(*, task: str = TASK, method: str = METHOD) -> dict:
    return {
        "task": task,
        "method": method,
        "main_metric": "test.macro_by_group.f1",
        "test": {"macro_by_group": {"f1": 0.5}},
        "requested_rows": 100,
        "scored_rows": 90,
    }


def make_run(
    parent: Path,
    *,
    name: str | None = None,
    metrics: dict | None = None,
    metrics_name: str = "metrics_ranking.json",
    artifacts: bool = True,
) -> Path:
    run_dir = parent / (name or f"{TS}_{TASK}_{METHOD}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if metrics is not None:
        (run_dir / metrics_name).write_text(json.dumps(metrics), encoding="utf-8")
    if artifacts:
        (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (run_dir / "predictions.parquet").write_bytes(b"PAR1")
    return run_dir


def inspect(path: Path, *, min_scored_ratio: float | None = None) -> sanitize.Run:
    return sanitize.inspect_run(path, min_scored_ratio=min_scored_ratio)


def make_args(**overrides: object) -> argparse.Namespace:
    defaults = dict(
        apply=False,
        delete=False,
        dedupe=False,
        require_artifacts=False,
        strict_identity=False,
        min_scored_ratio=None,
        drop_undefined_metric=False,
        drop_unscored=False,
        min_age_minutes=0.0,
        coverage=False,
        restore=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# --------------------------------------------------------------------------
# Safety: nothing holding a real measurement may ever be selected for removal.
# --------------------------------------------------------------------------


def test_container_dir_is_never_doomed_under_any_flag(tmp_path: Path) -> None:
    """A run-shaped directory that holds runs must survive every flag combo."""
    batch = tmp_path / "20260716T165403Z_qwen3_8b_ml1m_history_listwise"
    make_run(batch, metrics=ranking_metrics())
    make_run(
        batch,
        name=f"20260716T171539Z_{TASK}_{METHOD}",
        metrics=ranking_metrics(task="other_task"),
    )

    run = inspect(batch)
    assert "container" in run.reasons
    assert run.container_runs == 2
    assert not any(r in sanitize.FATAL for r in run.reasons)

    args = make_args(
        dedupe=True,
        require_artifacts=True,
        strict_identity=True,
        drop_undefined_metric=True,
        drop_unscored=True,
        min_scored_ratio=1.0,
    )
    result = sanitize.process_root(tmp_path, args)
    assert result.doomed == 0
    assert result.containers == 1
    assert result.protected == 2
    assert batch.exists()


def test_container_survives_apply(tmp_path: Path) -> None:
    batch = tmp_path / "20260716T165403Z_batch"
    child = make_run(batch, metrics=ranking_metrics())

    sanitize.process_root(tmp_path, make_args(apply=True))

    assert child.exists()
    assert (child / "metrics_ranking.json").exists()
    assert not (tmp_path / sanitize.QUARANTINE_DIRNAME).exists()


def test_predictions_without_metrics_are_advisory(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, metrics=None)
    (run_dir / "llm_errors.jsonl").write_text("", encoding="utf-8")

    run = inspect(run_dir)
    assert run.reasons == ["unscored_predictions"]
    assert not any(r in sanitize.FATAL for r in run.reasons)

    assert sanitize.process_root(tmp_path, make_args()).doomed == 0
    assert sanitize.process_root(tmp_path, make_args(drop_unscored=True)).doomed == 1


def test_healthy_sibling_survives_a_corrupt_metrics_file(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, metrics=ranking_metrics())
    (run_dir / "metrics.json").write_text('{"task": "ml-1m", trunca', encoding="utf-8")

    run = inspect(run_dir)
    assert run.reasons == ["unparseable_sibling"]
    assert run.task == TASK
    assert run.method == METHOD
    assert run.metrics_file == "metrics_ranking.json"
    assert sanitize.process_root(tmp_path, make_args()).doomed == 0


def test_delete_refuses_subdirs_and_parquet(tmp_path: Path) -> None:
    with_parquet = tmp_path / "20260101T000000Z_a"
    with_parquet.mkdir()
    (with_parquet / "predictions.parquet").write_bytes(b"PAR1")
    assert not sanitize.is_trivially_deletable(with_parquet)

    with_subdir = tmp_path / "20260101T000000Z_b"
    (with_subdir / "nested").mkdir(parents=True)
    assert not sanitize.is_trivially_deletable(with_subdir)

    log_only = tmp_path / "20260101T000000Z_c"
    log_only.mkdir()
    (log_only / "llm_errors.jsonl").write_text("", encoding="utf-8")
    assert sanitize.is_trivially_deletable(log_only)

    empty = tmp_path / "20260101T000000Z_d"
    empty.mkdir()
    assert sanitize.is_trivially_deletable(empty)


def test_delete_leaves_refused_dirs_in_place(tmp_path: Path) -> None:
    run_dir = tmp_path / f"{TS}_{TASK}_{METHOD}"
    (run_dir / "nested").mkdir(parents=True)

    result = sanitize.process_root(tmp_path, make_args(apply=True, delete=True))

    assert result.refused == 1
    assert run_dir.exists()


def test_empty_and_log_only_dirs_stay_fatal(tmp_path: Path) -> None:
    empty = tmp_path / f"{TS}_{TASK}_{METHOD}"
    empty.mkdir()
    assert inspect(empty).reasons == ["no_metrics"]

    log_only = tmp_path / f"20260725T230853Z_{TASK}_{METHOD}"
    log_only.mkdir()
    (log_only / "llm_errors.jsonl").write_text("", encoding="utf-8")
    assert inspect(log_only).reasons == ["no_metrics"]

    assert sanitize.process_root(tmp_path, make_args()).doomed == 2


def test_quarantine_round_trip(tmp_path: Path) -> None:
    run_dir = tmp_path / f"{TS}_{TASK}_{METHOD}"
    run_dir.mkdir()
    (run_dir / "llm_errors.jsonl").write_text("boom", encoding="utf-8")

    sanitize.process_root(tmp_path, make_args(apply=True))
    assert not run_dir.exists()
    assert (tmp_path / sanitize.QUARANTINE_DIRNAME / run_dir.name).is_dir()

    sanitize.restore(tmp_path, apply=True)
    assert (run_dir / "llm_errors.jsonl").read_text(encoding="utf-8") == "boom"
    assert not (tmp_path / sanitize.QUARANTINE_DIRNAME).exists()


# --------------------------------------------------------------------------
# cold_start adoption
# --------------------------------------------------------------------------


def test_ranking_only_run_is_aggregation_safe(tmp_path: Path) -> None:
    """The core regression: metrics_ranking.json alone is a complete run."""
    run_dir = make_run(tmp_path, metrics=ranking_metrics())

    run = inspect(run_dir)
    assert run.reasons == []
    assert run.metrics_file == "metrics_ranking.json"
    assert run.task == TASK
    assert run.method == METHOD
    assert run.protocol == "direct_listwise_preference_ranking"


def test_metrics_without_identity(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, metrics={"main_metric": "test.f1", "test": {"f1": 1.0}})
    assert inspect(run_dir).reasons == ["missing_identity"]


def test_unparseable_lone_metrics_file(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, metrics=None)
    (run_dir / "metrics_ranking.json").write_text("{not json", encoding="utf-8")
    assert inspect(run_dir).reasons == ["unparseable_metrics"]


def test_policy_metrics_is_not_an_identity_source(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, metrics=pointwise_metrics(), metrics_name="metrics.json")
    (run_dir / "policy_metrics.json").write_text(
        json.dumps({"task": TASK, "policy_recommendation_metrics": {}}), encoding="utf-8"
    )

    run = inspect(run_dir)
    assert "missing_identity" not in run.reasons
    assert run.metrics_file == "metrics.json"


def test_pod_grid_run_token_is_not_an_identity_mismatch(tmp_path: Path) -> None:
    name = f"{TS}_gf1adccbd_jb88d4fda2247_a1_{TASK}_{METHOD}"
    run_dir = make_run(tmp_path, name=name, metrics=ranking_metrics())
    assert "identity_mismatch" not in inspect(run_dir).reasons


def test_genuine_name_disagreement_is_flagged(tmp_path: Path) -> None:
    run_dir = make_run(
        tmp_path, name=f"{TS}_some_other_task_and_method", metrics=ranking_metrics()
    )
    assert "identity_mismatch" in inspect(run_dir).reasons


def test_name_matches_identity_rejects_unsafe_token() -> None:
    assert sanitize.name_matches_identity(f"{TS}_{TASK}_{METHOD}", TS, TASK, METHOD)
    assert sanitize.name_matches_identity(f"{TS}_g1_j2_a1_{TASK}_{METHOD}", TS, TASK, METHOD)
    assert not sanitize.name_matches_identity(
        f"{TS}_has space_{TASK}_{METHOD}", TS, TASK, METHOD
    )


def test_scored_ratio_takes_worst_split(tmp_path: Path) -> None:
    run_dir = make_run(
        tmp_path, metrics=ranking_metrics(val_fraction=1.0, test_fraction=0.8)
    )
    run = inspect(run_dir, min_scored_ratio=0.99)
    assert run.scored_ratio == pytest.approx(0.8)
    assert run.scored_split == "test"
    assert "incomplete_scoring" in run.reasons


def test_scored_ratio_flags_a_degraded_val_split(tmp_path: Path) -> None:
    run_dir = make_run(
        tmp_path, metrics=ranking_metrics(val_fraction=0.75, test_fraction=1.0)
    )
    run = inspect(run_dir, min_scored_ratio=0.99)
    assert run.scored_ratio == pytest.approx(0.75)
    assert run.scored_split == "val"


def test_top_level_row_counts_still_work(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, metrics=pointwise_metrics(), metrics_name="metrics.json")
    run = inspect(run_dir, min_scored_ratio=0.99)
    assert run.scored_ratio == pytest.approx(0.9)
    assert run.scored_split is None
    assert "incomplete_scoring" in run.reasons


def test_missing_coverage_never_triggers_incomplete_scoring(tmp_path: Path) -> None:
    metrics = ranking_metrics()
    del metrics["coverage"]
    run_dir = make_run(tmp_path, metrics=metrics)
    run = inspect(run_dir, min_scored_ratio=0.99)
    assert run.scored_ratio is None
    assert "incomplete_scoring" not in run.reasons


def test_parse_task_axes_families() -> None:
    cold_like = sanitize.parse_task_axes(TASK)
    assert cold_like is not None
    assert cold_like.dataset == "ml-1m_item_stats"
    assert cold_like.cell == (
        ("target", "like"),
        ("k", "1"),
        ("m", "1"),
        ("neg", "negu"),
    )
    assert (cold_like.replicate, cold_like.replicate_name) == (0, "sample")

    old = sanitize.parse_task_axes("ml-1m_cold_start_k5_cap20_eval_users1000_cg5_m9_seed2")
    assert old is not None
    assert (old.replicate, old.replicate_name) == (2, "seed")
    assert old.cell == (("k", "5"), ("m", "9"))

    policy = sanitize.parse_task_axes("ml-1m_policy_ranking_eval_users1000_seed6")
    assert policy is not None
    assert (policy.replicate, policy.replicate_name) == (6, "seed")
    assert policy.expected == tuple(range(10))

    assert sanitize.parse_task_axes("not-a-known-task") is None
    assert sanitize.parse_task_axes(None) is None


def test_coverage_report_lists_missing_sample(tmp_path: Path, capsys) -> None:
    for sample, ts in ((0, "20260725T221213Z"), (2, "20260725T231603Z")):
        task = TASK.replace("sample0", f"sample{sample}")
        make_run(tmp_path, name=f"{ts}_{task}_{METHOD}", metrics=ranking_metrics(task=task))

    sanitize.process_root(tmp_path, make_args(coverage=True))
    out = capsys.readouterr().out

    assert "surviving replicate coverage:" in out
    assert "target=like k=1 m=1 neg=negu" in out
    assert "sample n=2  [0, 2]" in out
    assert "MISSING [1]" in out


def test_dedupe_keeps_the_newest_ranking_run(tmp_path: Path) -> None:
    old = make_run(tmp_path, name=f"20260725T230853Z_{TASK}_{METHOD}", metrics=ranking_metrics())
    new = make_run(tmp_path, name=f"20260725T231603Z_{TASK}_{METHOD}", metrics=ranking_metrics())

    sanitize.process_root(tmp_path, make_args(dedupe=True, apply=True))

    assert not old.exists()
    assert new.exists()
    assert (tmp_path / sanitize.QUARANTINE_DIRNAME / old.name).is_dir()
