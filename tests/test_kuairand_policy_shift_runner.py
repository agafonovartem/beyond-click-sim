from __future__ import annotations

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.tasks import KuaiRandPolicyShiftTask, TaskSchema

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.ood.kuairand_policy_shift.run import (
    planned_cells,
    prepare_llm_preflight,
    run_method_suite,
)


PAID_METHOD = "llm_openai_vk_gpt54_mini_caption_categories"
PAID_PROBABILITY_METHOD = (
    "llm_openai_vk_gpt54_mini_probability_caption_categories"
)


class FakeChatCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=response),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=6,
                total_tokens=106,
            ),
        )


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        completions = FakeChatCompletions(responses)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


def test_planned_cells_make_full_policy_square_and_unique_controls() -> None:
    cells = planned_cells(["observed", "none", "permuted"])

    assert len(cells) == 10
    assert sum(cell.history_control == "observed" for cell in cells) == 4
    assert sum(cell.history_control == "permuted" for cell in cells) == 4
    assert sum(cell.history_control == "none" for cell in cells) == 2
    assert sum(cell.matched for cell in cells) == 2


def test_paid_preflight_guard_runs_before_any_fake_client_call(tmp_path) -> None:
    task = _toy_task()
    client = FakeClient(["C1\nC2\nC3"] * 20)

    with pytest.raises(RuntimeError, match="before any API call"):
        run_method_suite(
            {42: task},
            method_name=PAID_METHOD,
            output_dir=tmp_path / "blocked",
            user_ids=["u1", "u2"],
            controls=["observed", "none", "permuted"],
            hard_budget_usd=0.000_000_1,
            execute_llm=True,
            client=client,
            token_counter=lambda _: 100,
        )

    assert client.completions.calls == []


def test_caption_llm_runner_is_test_only_and_writes_all_cells(tmp_path) -> None:
    task = _toy_task()
    client = FakeClient(["C1\nC2\nC3"] * 20)
    output_dir = tmp_path / "llm"

    result = run_method_suite(
        {42: task},
        method_name=PAID_METHOD,
        output_dir=output_dir,
        user_ids=["u1", "u2"],
        controls=["observed", "none", "permuted"],
        hard_budget_usd=1.0,
        execute_llm=True,
        client=client,
        token_counter=lambda _: 1,
    )

    assert result["status"] == "complete"
    assert len(client.completions.calls) == 20  # 10 cells x 2 test users
    predictions = pd.read_parquet(output_dir / "predictions.parquet")
    assert len(predictions) == 60
    assert set(predictions["split"]) == {"test"}
    assert set(predictions["history_control"]) == {
        "observed",
        "permuted",
        "none",
    }
    assert set(predictions["history_policy"]) == {
        "standard",
        "random",
        "none",
    }
    assert set(predictions["metadata_condition"]) == {
        "caption_categories"
    }
    assert predictions.groupby(
        ["history_policy", "target_policy", "history_control"],
        dropna=False,
    ).ngroups == 10
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "comparisons.json").exists()
    assert (output_dir / "manifest.json").exists()

    first_request = client.completions.calls[0]
    assert first_request["max_completion_tokens"] == 24
    assert first_request["reasoning_effort"] == "none"
    assert "temperature" not in first_request
    assert "max_tokens" not in first_request
    messages = first_request["messages"]
    prompt = "\n".join(message["content"] for message in messages)
    assert "description (native Chinese)" in prompt
    assert "categories" in prompt
    assert "clicked after exposure (1=yes, 0=no): 1" in prompt
    candidate_prompt = prompt.split("Candidate items:", maxsplit=1)[1]
    assert "clicked after exposure" not in candidate_prompt
    assert "u1" not in prompt
    assert "std-u1-clicked" not in prompt
    assert "standard" not in prompt.lower()
    assert "random" not in prompt.lower()


def test_llm_runner_resumes_every_completed_group_without_new_calls(
    tmp_path,
) -> None:
    task = _toy_task()
    output_dir = tmp_path / "resumable"
    first_client = FakeClient(["C1\nC2\nC3"] * 20)

    run_method_suite(
        {42: task},
        method_name=PAID_METHOD,
        output_dir=output_dir,
        user_ids=["u1", "u2"],
        controls=["observed", "none", "permuted"],
        execute_llm=True,
        client=first_client,
        token_counter=lambda _: 1,
        max_workers=4,
    )
    assert len(first_client.completions.calls) == 20
    assert len(list((output_dir / "group_checkpoints").rglob("*.json"))) == 20

    # An empty fake client would crash on any API request. All groups must load
    # from their content-addressed checkpoints instead.
    resumed_client = FakeClient([])
    resumed = run_method_suite(
        {42: task},
        method_name=PAID_METHOD,
        output_dir=output_dir,
        user_ids=["u1", "u2"],
        controls=["observed", "none", "permuted"],
        execute_llm=True,
        client=resumed_client,
        token_counter=lambda _: 1,
        max_workers=4,
    )

    assert resumed_client.completions.calls == []
    assert resumed["actual_cost_usd"] == 0.0
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert resumed["new_api_cost_usd"] == 0.0
    assert (
        resumed["provider_cost_including_cached_usd"]
        == manifest["llm"]["provider_cost_including_cached_usd"]
    )
    assert resumed["provider_cost_including_cached_usd"] > 0.0
    cache = manifest["llm"]["group_checkpoint_cache"]
    assert cache["content_addressed"] is True
    assert cache["hits"] == 20
    assert cache["misses"] == 0
    assert cache["writes"] == 0
    assert cache["failed_misses"] == 0
    assert cache["hits_without_usage"] == 0
    assert manifest["llm"]["new_api_usage"]["requests"] == 0
    assert manifest["llm"]["cached_provider_usage"] == {
        "completion_tokens": 120,
        "hits_without_usage": 0,
        "prompt_tokens": 2000,
        "requests": 20,
        "total_tokens": 2120,
    }


def test_preflight_counts_only_test_groups_and_never_calls_client() -> None:
    task = _toy_task()
    from runners.ood.kuairand_policy_shift.methods import METHOD_REGISTRY

    estimate, prompts = prepare_llm_preflight(
        {42: task},
        spec=METHOD_REGISTRY[PAID_METHOD],
        user_ids=["u1", "u2"],
        controls=["observed", "none"],
        token_counter=lambda _: 1,
    )

    assert len(prompts) == 12  # (four observed + two no-history) x two users
    assert estimate.n_prompts == 12
    assert estimate.maximum_output_tokens == 12 * 24


def test_probability_runner_uses_all_rows_and_control_user_subset(
    tmp_path,
) -> None:
    task = _toy_task()
    client = FakeClient(
        ['{"C1":0.8,"C2":0.2,"C3":0.1}'] * 16
    )
    output_dir = tmp_path / "probability"

    result = run_method_suite(
        {42: task},
        method_name=PAID_PROBABILITY_METHOD,
        output_dir=output_dir,
        user_ids=["u1", "u2"],
        control_user_ids=["u1"],
        controls=["observed", "none", "permuted"],
        hard_budget_usd=1.0,
        execute_llm=True,
        client=client,
        token_counter=lambda _: 1,
    )

    assert result["status"] == "complete"
    assert len(client.completions.calls) == 16
    predictions = pd.read_parquet(output_dir / "predictions.parquet")
    assert len(predictions) == 48
    permuted = predictions[
        predictions["history_control"].eq("permuted")
    ]
    assert set(permuted["user_id"]) == {"u1"}
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert all(cell["probability_scores"] for cell in metrics["cells"])
    assert all("log_loss" in cell for cell in metrics["cells"])
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["cohort"]["n_users"] == 2
    assert manifest["cohort"]["permuted_control_users"] == 1
    assert manifest["metrics"]["primary"] == ["log_loss", "brier"]


def test_classical_runner_reuses_same_fixed_targets(tmp_path) -> None:
    task = _toy_task()
    output_dir = tmp_path / "mean"

    result = run_method_suite(
        {42: task},
        method_name="global_mean",
        output_dir=output_dir,
        user_ids=["u1", "u2"],
        controls=["observed"],
    )

    assert result["status"] == "complete"
    predictions = pd.read_parquet(output_dir / "predictions.parquet")
    assert len(predictions) == 24
    assert predictions.groupby(
        ["task_seed", "target_policy", "candidate_group"]
    )["item_id"].nunique().eq(3).all()
    assert len(result["cells"]) == 4
    assert all("ndcg@3" in cell for cell in result["cells"])
    assert all("hit_rate@1" in cell for cell in result["cells"])


def test_text_affinity_keeps_constant_label_source_users_out_of_profiles_only(
    tmp_path,
) -> None:
    task = _toy_task()
    for history in (task.standard_history, task.random_history):
        mask = history["user_id"].eq("u2")
        history.loc[mask, "target"] = 1
        history.loc[mask, "item_description"] = "source-only vocabulary"

    output_dir = tmp_path / "text-affinity"
    result = run_method_suite(
        {42: task},
        method_name="text_affinity_caption_categories",
        output_dir=output_dir,
        user_ids=["u1"],
        controls=["observed"],
    )

    assert result["status"] == "complete"
    manifest = json.loads((output_dir / "manifest.json").read_text())
    diagnostics = manifest["fit_diagnostics"]
    assert len(diagnostics) == 4
    assert all(entry["n_source_users"] == 2 for entry in diagnostics)
    assert all(entry["n_profile_users"] == 1 for entry in diagnostics)
    assert all(
        entry["n_skipped_constant_label_users"] == 1
        for entry in diagnostics
    )
    assert all(entry["n_target_users"] == 1 for entry in diagnostics)
    assert all(
        entry["target_profile_user_fraction"] == 1.0
        for entry in diagnostics
    )


def _toy_task() -> KuaiRandPolicyShiftTask:
    schema = TaskSchema(
        target_column="target",
        feature_columns=("item_description", "item_category_path"),
        id_columns=("user_id", "item_id"),
        candidate_group_column="candidate_group",
        history_context_columns=("history_click",),
    )
    standard = _history("standard")
    random = _history("random")
    standard_permuted = _permute_history(standard)
    random_permuted = _permute_history(random)
    val_standard = _targets("val", "standard")
    val_random = _targets("val", "random")
    test_standard = _targets("test", "standard")
    test_random = _targets("test", "random")
    return KuaiRandPolicyShiftTask(
        name="toy-kuairand-policy-shift",
        standard_history=standard,
        random_history=random,
        standard_history_permuted=standard_permuted,
        random_history_permuted=random_permuted,
        val_standard_targets=val_standard,
        val_random_targets=val_random,
        test_standard_targets=test_standard,
        test_random_targets=test_random,
        schema=schema,
        manifest={
            "dataset": "kuairand-pure",
            "seed": 42,
            "candidate_construction": {
                "kind": "observed_c3_pseudo_list",
            },
        },
    )


def _history(policy: str) -> pd.DataFrame:
    rows = []
    for user_id in ("u1", "u2"):
        for position, click in enumerate((1, 0), start=1):
            item_id = f"{policy}-{user_id}-{'clicked' if click else 'skipped'}"
            rows.append(
                _row(
                    user_id=user_id,
                    item_id=item_id,
                    description=(
                        "用户喜欢的猫咪视频"
                        if click
                        else "用户跳过的房产视频"
                    ),
                    category="宠物 > 猫" if click else "房产家居",
                    history_click=click,
                    target=click,
                    candidate_group=pd.NA,
                    history_policy=policy,
                    target_policy=pd.NA,
                )
            )
    return pd.DataFrame(rows)


def _permute_history(history: pd.DataFrame) -> pd.DataFrame:
    permuted = history.copy()
    permuted["history_click"] = (
        permuted.groupby("user_id")["history_click"]
        .transform(lambda values: values.iloc[::-1].to_numpy())
        .astype(int)
    )
    permuted["target"] = permuted["history_click"]
    return permuted


def _targets(split: str, policy: str) -> pd.DataFrame:
    rows = []
    for user_id in ("u1", "u2"):
        labels = (1, 0, 0)
        for position, target in enumerate(labels, start=1):
            rows.append(
                _row(
                    user_id=user_id,
                    item_id=f"{split}-{policy}-{user_id}-c{position}",
                    description=f"候选短视频 {position}：猫咪和音乐",
                    category="宠物 > 猫" if position == 1 else "音乐",
                    history_click=pd.NA,
                    target=target,
                    candidate_group=f"{split}_{policy}_{user_id}_c3",
                    history_policy=pd.NA,
                    target_policy=policy,
                )
            )
    return pd.DataFrame(rows)


def _row(
    *,
    user_id: str,
    item_id: str,
    description: str,
    category: str,
    history_click: object,
    target: int,
    candidate_group: object,
    history_policy: object,
    target_policy: object,
) -> dict[str, object]:
    return {
        "user_id": user_id,
        "item_id": item_id,
        "item_description": description,
        "item_category_path": category,
        "history_click": history_click,
        "target": target,
        "candidate_group": candidate_group,
        "history_policy": history_policy,
        "target_policy": target_policy,
        "event_date": "2022-05-04",
        "event_timestamp_ms": 1,
        "interaction_id": f"interaction-{item_id}",
    }
