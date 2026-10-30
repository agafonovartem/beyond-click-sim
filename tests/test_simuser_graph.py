import numpy as np
import pytest

from beyond_click_sim.scorers.simuser.graph import TrainingGraph
from beyond_click_sim.scorers.simuser.profiles import add_training_statistics


def graph_fixture():
    catalog = {
        item: {
            "title": item,
            "summary": "plot",
            "genres": [genre],
            "training_mean_rating": None,
            "training_interaction_count": 0,
        }
        for item, genre in [("a", "X"), ("b", "X"), ("c", "Y"), ("d", "Z"), ("e", "Z")]
    }
    events = [
        {"user": u, "item": i, "rating": r, "event": f"event{n}"}
        for n, (u, i, r) in enumerate(
            [("u", "a", 1), ("u", "b", 4), ("v", "a", 5), ("v", "c", 3), ("w", "c", 5)]
        )
    ]
    add_training_statistics(catalog, events)
    graph = TrainingGraph(catalog, events)
    graph.vectors = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0]]
    )
    return graph


def test_hand_computed_pathsim_projection_fusion_and_counts():
    graph = graph_fixture()
    ui_count, genre_count, ui, genre = graph.item_channels("a")
    np.testing.assert_allclose(ui_count, [2, 1, 1, 0, 0])
    np.testing.assert_allclose(genre_count, [1, 1, 0, 0, 0])
    np.testing.assert_allclose(ui, [1, 2 / 3, 0.5, 0, 0])
    np.testing.assert_allclose(genre, [1, 1, 0, 0, 0])
    projection = graph.user_projection("u")
    np.testing.assert_allclose(
        projection, [(1 + 5 / 6) / 2, (5 / 6 + 1) / 2, 0.125, 0, 0]
    )
    records = {r["item_id"]: r for r in graph.ranked("u", "a")}
    assert "a" not in records
    assert records["b"]["score"] == pytest.approx(
        0.75 * (0.8 * (5 / 6) + 0.2 * (11 / 12)) + 0.25
    )
    assert records["c"]["score"] == pytest.approx(
        0.75 * (0.8 * 0.25 + 0.2 * 0.125) + 0.25 * 0.5
    )
    assert records["d"]["score"] == 0
    assert records["e"]["score"] == 0.125


def test_zero_denominators_missing_channel_stable_ties_and_training_only_metadata():
    graph = graph_fixture()
    assert graph.item_channels("d")[2][4] == 0
    assert graph.item_channels("d")[3][4] == 1
    assert (
        graph.catalog["a"]["training_mean_rating"] == 3
    )  # Includes target's permitted rating.
    assert graph.catalog["a"]["training_interaction_count"] == 2
    assert graph.catalog["e"]["training_mean_rating"] is None
    assert graph.catalog["e"]["training_interaction_count"] == 0
    assert ("u", "c") not in graph.edges
    graph.vectors[:] = [1, 0]
    ranked = graph.ranked("u", "a")
    assert [r["item_id"] for r in ranked if r["item_id"] in {"d", "e"}] == ["d", "e"]
    for row in graph.ranked("u", "d"):
        if row["item_id"] == "e":
            assert (
                row["item_similarity"] == 0.5
            )  # No renormalizing the absent collaborative channel.


def test_actual_structural_witnesses_user_prefixes_feedback_and_semantic_only():
    graph = graph_fixture()
    paths = graph.witnesses("u", "a", "b")
    assert paths["item_user_item"] == [["a", "u", "b"]]
    assert paths["item_genre_item"] == [["a", "genre:X", "b"]]
    for key in ("user_history_user_item", "user_history_genre_item"):
        assert all(p[0] == "u" and p[1] in graph.histories["u"] for p in paths[key])
    ranked = {"C1": graph.ranked("u", "a"), "C2": graph.ranked("u", "b")}
    evidence = graph.retrieve("u", ranked, 10)
    assert (
        len(evidence) == 5
    )  # Other candidates are allowed; each query excludes itself only.
    e = next(row for row in evidence if row["item_id"] == "e")
    assert all(
        q["semantic_only"] and not any(q["witnesses"].values()) for q in e["queries"]
    )
    b = next(row for row in evidence if row["item_id"] == "b")
    assert b["target_training_feedback"] == [
        {"event_id": "event1", "rating": 4, "relation": "liked"}
    ]
    c = next(row for row in evidence if row["item_id"] == "c")
    assert c["target_training_feedback"] == []
    assert all(
        q["item_id"] != q["query_item"] for row in evidence for q in row["queries"]
    )
    assert {r["id"] for r in graph.retrieve("u", ranked, 1)} <= {
        r["id"] for r in graph.retrieve("u", ranked, 2)
    }
