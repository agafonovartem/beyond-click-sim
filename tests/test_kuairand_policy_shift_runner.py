from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.tasks import (
    KuaiRandStandardToRandomTask,
    TaskSchema,
)
from runners.ood.kuairand_policy_shift.checkpoint import (
    GroupCheckpointStore,
    build_legacy_group_checkpoint_key,
)
from runners.ood.kuairand_policy_shift.methods import (
    METHOD_REGISTRY,
    make_scorer,
)
from runners.ood.kuairand_policy_shift.run import (
    _cell_dict,
    _llm_request_config,
    _parse_args,
    _split_model_xy,
    planned_cells,
    prepare_llm_preflight,
    run_method_suite,
    write_combined_analysis,
)


GPT_METHOD = "llm_openai_vk_gpt55_listwise_english"


class FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
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
                completion_tokens=4,
                total_tokens=104,
            ),
        )


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


def test_planned_cells_are_two_for_pop_and_four_for_gpt() -> None:
    pop_cells = planned_cells(METHOD_REGISTRY["popularity"])
    gpt_cells = planned_cells(METHOD_REGISTRY[GPT_METHOD])

    assert [
        (cell.target_policy, cell.history_control) for cell in pop_cells
    ] == [
        ("standard", "observed"),
        ("random", "observed"),
    ]
    assert {
        (cell.target_policy, cell.history_control) for cell in gpt_cells
    } == {
        ("standard", "observed"),
        ("random", "observed"),
        ("standard", "none"),
        ("random", "none"),
    }


def test_preflight_materializes_every_prompt_without_client_calls() -> None:
    task = _toy_task()

    estimate, prompts = prepare_llm_preflight(
        {0: task},
        spec=METHOD_REGISTRY[GPT_METHOD],
        user_ids=["u1", "u2"],
        token_counter=lambda _: 1,
    )

    assert len(prompts) == 8  # 2 users x 2 policies x 2 controls.
    assert estimate.n_prompts == 8
    assert estimate.maximum_output_tokens == 8 * 64
    assert estimate.prompt_plan_sha256
    assert any("- No response history available." in str(prompt) for prompt in prompts)


def test_popularity_runs_on_full_source_and_has_no_common_warm_fallback(
    tmp_path: Path,
) -> None:
    task = _toy_task()
    result = run_method_suite(
        {0: task},
        method_name="popularity",
        output_dir=tmp_path / "popularity",
        user_ids=["u1", "u2"],
    )

    assert result["status"] == "complete"
    assert result["n_predictions"] == 8
    predictions = pd.read_parquet(
        tmp_path / "popularity" / "predictions.parquet"
    )
    assert set(predictions["target_policy"]) == {"standard", "random"}
    assert set(predictions["history_control"]) == {"observed"}
    assert not predictions["method_fallback"].any()
    # Item a has two positive source responses; b/c/d have one or zero.
    scores = predictions.groupby("item_id")["score"].first().to_dict()
    assert scores["a"] == 2.0
    assert scores["b"] == 1.0


def test_first_suite_rejects_full_pool_tasks_and_cli_axis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _toy_task()
    task.target_pool = "full_pool"

    with pytest.raises(ValueError, match="supports only.*common_warm"):
        run_method_suite(
            {0: task},
            method_name="popularity",
            output_dir=tmp_path / "popularity",
            user_ids=["u1", "u2"],
        )

    monkeypatch.setattr(
        sys,
        "argv",
        ["kuairand-policy-shift", "--target-pool", "full_pool"],
    )
    with pytest.raises(SystemExit):
        _parse_args()


def test_gpt_runner_uses_english_target_response_prompt_and_checkpoints(
    tmp_path: Path,
) -> None:
    task = _toy_task()
    output_dir = tmp_path / GPT_METHOD
    first_client = FakeClient(["C1\nC2"] * 8)

    result = run_method_suite(
        {0: task},
        method_name=GPT_METHOD,
        output_dir=output_dir,
        user_ids=["u1", "u2"],
        execute_llm=True,
        client=first_client,
        token_counter=lambda _: 1,
        max_workers=2,
    )

    assert result["status"] == "complete"
    assert len(first_client.completions.calls) == 8
    assert len(list((output_dir / "group_checkpoints").rglob("*.json"))) == 8
    first_request = first_client.completions.calls[0]
    assert first_request["model"] == "gpt-5.5"
    assert first_request["max_completion_tokens"] == 64
    assert first_request["reasoning_effort"] == "none"
    assert "temperature" not in first_request
    assert "max_tokens" not in first_request
    prompt = "\n".join(
        message["content"] for message in first_request["messages"]
    )
    assert "Response target:" in prompt
    assert "English caption:" in prompt
    assert "English cover text:" in prompt
    assert "English category:" in prompt
    assert "Positive-preference target:" not in prompt
    assert "item_id" not in prompt
    assert "target_policy" not in prompt

    resumed_client = FakeClient([])
    resumed = run_method_suite(
        {0: task},
        method_name=GPT_METHOD,
        output_dir=output_dir,
        user_ids=["u1", "u2"],
        execute_llm=True,
        client=resumed_client,
        token_counter=lambda _: 1,
        max_workers=2,
    )

    assert resumed["status"] == "complete"
    assert resumed_client.completions.calls == []
    assert resumed["new_api_cost_usd"] == 0.0


def test_legacy_checkpoint_is_migrated_without_an_api_call(
    tmp_path: Path,
) -> None:
    task = _toy_task()
    spec = METHOD_REGISTRY[GPT_METHOD]
    output_dir = tmp_path / GPT_METHOD
    store = GroupCheckpointStore(output_dir / "group_checkpoints")
    for cell_spec in planned_cells(spec):
        cell = task.cell(
            split="test",
            target_policy=cell_spec.target_policy,
            history_control=cell_spec.history_control,
        )
        X_history, y_history = _split_model_xy(
            cell.history,
            target_column=cell.schema.target_column,
        )
        X_target, _ = _split_model_xy(
            cell.targets,
            target_column=cell.schema.target_column,
        )
        scorer = make_scorer(
            spec,
            seed=0,
            client=FakeClient([]),
        ).fit(X_history, y_history)
        request_config = _llm_request_config(scorer)
        cell_identity = {
            "task_seed": 0,
            "target_pool": "common_warm",
            **_cell_dict(cell_spec),
        }
        for group_id, group in X_target.groupby(
            "candidate_group",
            sort=False,
        ):
            labels = [
                f"C{position}" for position in range(1, len(group) + 1)
            ]
            messages = scorer._build_messages(
                user_id=group["user_id"].iloc[0],
                candidates=group,
                labels=labels,
            )
            item_ids = group["item_id"].tolist()
            legacy_key = build_legacy_group_checkpoint_key(
                task_identity="a" * 64,
                cell=cell_identity,
                user_id=group["user_id"].iloc[0],
                candidate_group=group_id,
                candidate_item_ids=item_ids,
                prompt_messages=messages,
                model=str(spec.model),
                request_config=request_config,
            )
            store.save(
                legacy_key,
                item_ids=item_ids,
                scores=[2.0, 1.0],
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 4,
                    "total_tokens": 104,
                },
            )

    client = FakeClient([])
    result = run_method_suite(
        {0: task},
        method_name=GPT_METHOD,
        output_dir=output_dir,
        user_ids=["u1", "u2"],
        execute_llm=True,
        client=client,
        token_counter=lambda _: 1,
    )
    manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert result["status"] == "complete"
    assert client.completions.calls == []
    assert len(list((output_dir / "group_checkpoints").rglob("*.json"))) == 16
    assert manifest["llm"]["group_checkpoint_cache"]["hits"] == 8
    assert manifest["llm"]["group_checkpoint_cache"]["legacy_hits"] == 8
    assert manifest["llm"]["group_checkpoint_cache"]["legacy_migrations"] == 8
    assert manifest["llm"]["group_checkpoint_cache"]["writes"] == 0
    assert manifest["llm"]["new_api_usage"]["requests"] == 0


@pytest.mark.parametrize(
    ("responses", "expected_status", "expected_errors", "successful_groups"),
    [
        (["not a ranking", *(["C1\nC2"] * 7)], "complete_with_errors", 1, 7),
        ([], "failed", 8, 0),
    ],
)
def test_llm_failures_produce_non_complete_run_status(
    tmp_path: Path,
    responses: list[str],
    expected_status: str,
    expected_errors: int,
    successful_groups: int,
) -> None:
    output_dir = tmp_path / expected_status / GPT_METHOD
    result = run_method_suite(
        {0: _toy_task()},
        method_name=GPT_METHOD,
        output_dir=output_dir,
        user_ids=["u1", "u2"],
        execute_llm=True,
        client=FakeClient(responses),
        token_counter=lambda _: 1,
    )
    manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert result["status"] == expected_status
    assert result["n_errors"] == expected_errors
    assert manifest["status"] == expected_status
    assert manifest["llm"]["errors"] == expected_errors
    assert manifest["llm"]["group_outcomes"] == {
        "planned": 8,
        "successful": successful_groups,
        "failed": expected_errors,
    }


def test_combined_analysis_requires_byte_identical_targets(
    tmp_path: Path,
) -> None:
    task = _toy_task()
    run_method_suite(
        {0: task},
        method_name="popularity",
        output_dir=tmp_path / "popularity",
        user_ids=["u1", "u2"],
    )
    run_method_suite(
        {0: task},
        method_name=GPT_METHOD,
        output_dir=tmp_path / GPT_METHOD,
        user_ids=["u1", "u2"],
        execute_llm=True,
        client=FakeClient(["C1\nC2"] * 8),
        token_counter=lambda _: 1,
    )

    combined = write_combined_analysis(tmp_path)

    assert combined["status"] == "complete"
    comparison_names = {
        row["comparison"] for row in combined["comparisons"]
    }
    assert comparison_names == {
        "gpt_history_minus_no_history",
        "gpt_history_minus_popularity",
    }
    assert (tmp_path / "combined_analysis.json").exists()

    path = tmp_path / GPT_METHOD / "predictions.parquet"
    predictions = pd.read_parquet(path)
    predictions.loc[0, "target"] = 1 - int(predictions.loc[0, "target"])
    predictions.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Target rows differ"):
        write_combined_analysis(tmp_path)


def test_combined_analysis_rejects_incomplete_llm_run(
    tmp_path: Path,
) -> None:
    task = _toy_task()
    run_method_suite(
        {0: task},
        method_name="popularity",
        output_dir=tmp_path / "popularity",
        user_ids=["u1", "u2"],
    )
    run_method_suite(
        {0: task},
        method_name=GPT_METHOD,
        output_dir=tmp_path / GPT_METHOD,
        user_ids=["u1", "u2"],
        execute_llm=True,
        client=FakeClient(["not a ranking", *(["C1\nC2"] * 7)]),
        token_counter=lambda _: 1,
    )

    with pytest.raises(ValueError, match="requires complete method runs"):
        write_combined_analysis(tmp_path)


def _toy_task() -> KuaiRandStandardToRandomTask:
    source_rows = [
        _history_row("u1", "a", 1),
        _history_row("u1", "b", 1),
        _history_row("u2", "a", 1),
        _history_row("u2", "c", 0),
        _history_row("trainer", "d", 1),
    ]
    source = pd.DataFrame(source_rows)
    visible = source.loc[source["user_id"].isin({"u1", "u2"})].copy()
    standard = _target_frame("standard")
    random = _target_frame("random")
    schema = TaskSchema(
        target_column="target",
        feature_columns=(
            "item_caption_en",
            "item_cover_text_en",
            "item_category_path_en",
        ),
        id_columns=("user_id", "item_id"),
        candidate_group_column="candidate_group",
        history_context_columns=("history_click",),
    )
    return KuaiRandStandardToRandomTask(
        name="toy-kuairand-standard-to-random",
        target_pool="common_warm",
        visible_history=visible,
        training_source=source,
        item_features=pd.DataFrame(
            {
                "item_id": ["a", "b", "c", "d"],
                "item_caption_en": [
                    "Caption a",
                    "Caption b",
                    "Caption c",
                    "Caption d",
                ],
                "item_cover_text_en": [
                    "Cover a",
                    "Cover b",
                    "Cover c",
                    "Cover d",
                ],
                "item_category_path_en": [
                    "Category > Subcategory",
                ]
                * 4,
            }
        ),
        val_standard_targets=standard.copy(),
        val_random_targets=random.copy(),
        test_standard_targets=standard,
        test_random_targets=random,
        schema=schema,
        manifest={
            "protocol": "kuairand-standard-to-random-v2",
            "hashes": {"task_identity_sha256": "a" * 64},
        },
    )


def _history_row(user_id: str, item_id: str, target: int) -> dict[str, object]:
    return {
        "user_id": user_id,
        "item_id": item_id,
        "item_caption_en": f"Caption {item_id}",
        "item_cover_text_en": f"Cover {item_id}",
        "item_category_path_en": "Category > Subcategory",
        "history_click": target,
        "target": target,
        "candidate_group": pd.NA,
    }


def _target_frame(policy: str) -> pd.DataFrame:
    rows = []
    for user_id, items in {
        "u1": (("a", 1), ("b", 0)),
        "u2": (("a", 0), ("c", 1)),
    }.items():
        group = f"test_{policy}_{user_id}"
        for item_id, target in items:
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "item_caption_en": f"Caption {item_id}",
                    "item_cover_text_en": f"Cover {item_id}",
                    "item_category_path_en": "Category > Subcategory",
                    "history_click": pd.NA,
                    "target": target,
                    "candidate_group": group,
                    "target_policy": policy,
                    "target_pool": "common_warm",
                    "source_exposure_count": 20,
                    "source_distinct_user_count": 5,
                    "source_positive_user_count": 1,
                }
            )
    return pd.DataFrame(rows)
