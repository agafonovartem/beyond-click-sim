from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from beyond_click_sim.scorers import (
    Agent4RecListwiseRankingScorer,
    Agent4RecPreferenceListwiseRankingScorer,
    Agent4RecPreferenceYesNoScorer,
    Agent4RecProfileGenerator,
    Agent4RecYesNoScorer,
)
from beyond_click_sim.scorers.agent4rec.prompts import (
    AGENT4REC_FORCED_ITEMS_SYSTEM_PROMPT_TEMPLATE,
    AGENT4REC_SOCIAL_TRAITS_SYSTEM_PROMPT_TEMPLATE,
    agent4rec_system_prompt,
    agent4rec_user_prompt,
)
from beyond_click_sim.scorers.agent4rec.yes_no import parse_agent4rec_watch_response
from beyond_click_sim.scorers.agent4rec.yes_no import (
    parse_agent4rec_preference_response,
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
            ]
        )


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        completions = FakeChatCompletions(responses)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


def test_agent4rec_listwise_scorers_reuse_profile_prompt() -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["h1", "h2"],
            "item_genres": ["Animation", "Crime"],
            "rating": [5, 2],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "candidate_group": ["g1", "g1"],
            "item_title": ["Lion King", "Heat"],
        },
        index=["a", "b"],
    )
    profile_generator = Agent4RecProfileGenerator(profile_components=("traits",))
    interaction_client = FakeClient(["1. C2\n2. C1"])
    interaction_scorer = Agent4RecListwiseRankingScorer(
        client=interaction_client,
        model="fake-model",
        profile_generator=profile_generator,
        candidate_description_columns=("item_title",),
    ).fit(X_train, pd.Series([1, 1]))

    assert interaction_scorer.score(X_test).to_dict() == {"a": 0.0, "b": 1.0}
    interaction_prompt = interaction_client.completions.calls[0]["messages"][1][
        "content"
    ]
    assert "##recommended list##" in interaction_prompt
    assert "most likely to least likely" in interaction_prompt
    assert "Use every candidate ID exactly once" in interaction_prompt
    assert "exactly 2 non-empty lines" in interaction_prompt

    preference_client = FakeClient(["1. C1\n2. C2"])
    preference_scorer = Agent4RecPreferenceListwiseRankingScorer(
        client=preference_client,
        model="fake-model",
        profile_generator=Agent4RecProfileGenerator(profile_components=("traits",)),
        candidate_description_columns=("item_title",),
        target_description="The user would rate the movie at least 4 out of 5.",
    ).fit(X_train, pd.Series([1, 1]))

    assert preference_scorer.score(X_test).to_dict() == {"a": 1.0, "b": 0.0}
    preference_prompt = preference_client.completions.calls[0]["messages"][1][
        "content"
    ]
    assert "Positive-preference target:" in preference_prompt
    assert "rate the movie at least 4 out of 5" in preference_prompt


def test_parse_agent4rec_watch_response_maps_reordered_decisions_by_label() -> None:
    parsed = parse_agent4rec_watch_response(
        "ID: C2; MOVIE: B; WATCH: No; REASON: not ok\n"
        "ID: C1; MOVIE: A; WATCH: yes; REASON: ok",
        labels=["C1", "C2"],
    )

    assert parsed == {"C1": 1.0, "C2": 0.0}


def test_parse_agent4rec_watch_response_accepts_colon_in_movie_field() -> None:
    parsed = parse_agent4rec_watch_response(
        "ID: C1; MOVIE: Austin Powers: The Spy Who Shagged Me (1999); WATCH: no; REASON: comedy\n"
        "ID: C2; MOVIE: Bowfinger (1999); WATCH: yes; REASON: ok",
        labels=["C1", "C2"],
    )

    assert parsed == {"C1": 0.0, "C2": 1.0}


def test_parse_agent4rec_watch_response_accepts_qwen_label_prefix() -> None:
    parsed = parse_agent4rec_watch_response(
        "C1: [1]; MOVIE: A; WATCH: yes; REASON: ok\n"
        "C2: [2]; MOVIE: B; WATCH: no; REASON: not ok",
        labels=["C1", "C2"],
    )

    assert parsed == {"C1": 1.0, "C2": 0.0}


def test_parse_agent4rec_watch_response_accepts_unlabeled_title() -> None:
    parsed = parse_agent4rec_watch_response(
        "C1: Austin Powers: The Spy Who Shagged Me (1999); WATCH: no; REASON: comedy\n"
        "C2: Bowfinger (1999); WATCH: yes; REASON: ok",
        labels=["C1", "C2"],
    )

    assert parsed == {"C1": 0.0, "C2": 1.0}


def test_parse_agent4rec_watch_response_accepts_unlabeled_title_with_apostrophe() -> None:
    parsed = parse_agent4rec_watch_response(
        "C1: General's Daughter, The (1999); WATCH: no; REASON: thriller\n"
        "C2: Ghostbusters (1984); WATCH: yes; REASON: classic comedy",
        labels=["C1", "C2"],
    )

    assert parsed == {"C1": 0.0, "C2": 1.0}


def test_parse_agent4rec_watch_response_accepts_literal_backslash_n_and_escaped_apostrophe() -> None:
    parsed = parse_agent4rec_watch_response(
        "C1: General\\'s Daughter, The (1999); WATCH: no; REASON: thriller\\n"
        "C2: Ghostbusters (1984); WATCH: yes; REASON: classic comedy",
        labels=["C1", "C2"],
    )

    assert parsed == {"C1": 0.0, "C2": 1.0}


def test_parse_agent4rec_watch_response_accepts_logged_qwen_unlabeled_title_response() -> None:
    parsed = parse_agent4rec_watch_response(
        "C1: Austin Powers: The Spy Who Shagged Me (1999); WATCH: no; REASON: Comedy genre does not align with my preference for family-friendly, romantic, or classic films.\n"
        "C2: Bowfinger (1999); WATCH: no; REASON: Comedy genre does not align with my preference for family-friendly, romantic, or classic films.\n"
        "C3: Toy Story 2 (1999); WATCH: yes; REASON: Animation and Comedy fit my taste for family-friendly comedies and feel-good content.\n"
        "C4: 10 Things I Hate About You (1999); WATCH: yes; REASON: Romantic Comedy aligns perfectly with my preferred genres.\n"
        "C5: Arachnophobia (1990); WATCH: no; REASON: Thriller and Sci-Fi elements do not align with my preference for family-friendly and feel-good movies.\n"
        "C6: Shakespeare in Love (1998); WATCH: yes; REASON: Romantic Comedy and Drama fit my taste for romantic and feel-good films.\n"
        "C7: Runaway Bride (1999); WATCH: yes; REASON: Romantic Comedy aligns perfectly with my preferred genres.\n"
        "C8: General's Daughter, The (1999); WATCH: no; REASON: Thriller and Drama do not align with my preference for family-friendly and feel-good movies.\n"
        "C9: Never Been Kissed (1999); WATCH: yes; REASON: Romantic Comedy aligns perfectly with my preferred genres.\n"
        "C10: Ghostbusters (1984); WATCH: yes; REASON: Classic Comedy fits my taste for classic films and family-friendly humor.",
        labels=[f"C{i}" for i in range(1, 11)],
    )

    assert parsed == {
        "C1": 0.0,
        "C2": 0.0,
        "C3": 1.0,
        "C4": 1.0,
        "C5": 0.0,
        "C6": 1.0,
        "C7": 1.0,
        "C8": 0.0,
        "C9": 1.0,
        "C10": 1.0,
    }


def test_parse_agent4rec_watch_response_accepts_logged_qwen_unlabeled_title_repr_escapes() -> None:
    parsed = parse_agent4rec_watch_response(
        "C1: Austin Powers: The Spy Who Shagged Me (1999); WATCH: no; REASON: Comedy genre does not align with my preference for family-friendly, romantic, or classic films.\\n"
        "C2: Bowfinger (1999); WATCH: no; REASON: Comedy genre does not align with my preference for family-friendly, romantic, or classic films.\\n"
        "C3: Toy Story 2 (1999); WATCH: yes; REASON: Animation and Comedy fit my taste for family-friendly comedies and feel-good content.\\n"
        "C4: 10 Things I Hate About You (1999); WATCH: yes; REASON: Romantic Comedy aligns perfectly with my preferred genres.\\n"
        "C5: Arachnophobia (1990); WATCH: no; REASON: Thriller and Sci-Fi elements do not align with my preference for family-friendly and feel-good movies.\\n"
        "C6: Shakespeare in Love (1998); WATCH: yes; REASON: Romantic Comedy and Drama fit my taste for romantic and feel-good films.\\n"
        "C7: Runaway Bride (1999); WATCH: yes; REASON: Romantic Comedy aligns perfectly with my preferred genres.\\n"
        "C8: General\\'s Daughter, The (1999); WATCH: no; REASON: Thriller and Drama do not align with my preference for family-friendly and feel-good movies.\\n"
        "C9: Never Been Kissed (1999); WATCH: yes; REASON: Romantic Comedy aligns perfectly with my preferred genres.\\n"
        "C10: Ghostbusters (1984); WATCH: yes; REASON: Classic Comedy fits my taste for classic films and family-friendly humor.",
        labels=[f"C{i}" for i in range(1, 11)],
    )

    assert parsed == {
        "C1": 0.0,
        "C2": 0.0,
        "C3": 1.0,
        "C4": 1.0,
        "C5": 0.0,
        "C6": 1.0,
        "C7": 1.0,
        "C8": 0.0,
        "C9": 1.0,
        "C10": 1.0,
    }


def test_parse_agent4rec_watch_response_accepts_reason_with_semicolon() -> None:
    parsed = parse_agent4rec_watch_response(
        "C1: Amadeus; WATCH: yes; REASON: Classic cinema with strong performances and emotional depth fitting my exclusive taste.\n"
        "C2: Good Will Hunting; WATCH: no; REASON: Lacks the international or classic prestige I demand; too mainstream.\n"
        "C3: Gandhi; WATCH: yes; REASON: Historical drama with profound themes and strong character focus aligns with my preferences.\n"
        "C4: Dead Man Walking; WATCH: yes; REASON: Intense character-driven narrative with emotional depth and strong performances.\n"
        "C5: Driving Miss Daisy; WATCH: no; REASON: Perceived as overly sentimental and lacking the intellectual rigor I require.\n"
        "C6: To Kill a Mockingbird; WATCH: yes; REASON: Classic cinema with thought-provoking themes and exceptional character work.\n"
        "C7: Like Water for Chocolate; WATCH: yes; REASON: International film with unique artistic style and emotional depth.\n"
        "C8: Edward Scissorhands; WATCH: no; REASON: Too whimsical and fantasy-oriented; does not match my serious, realistic taste.\n"
        "C9: Searching for Bobby Fischer; WATCH: no; REASON: Lacks the historical weight and international scope I prioritize.\n"
        "C10: Sense and Sensibility; WATCH: yes; REASON: Classic period drama with strong performances and emotional resonance.",
        labels=[f"C{i}" for i in range(1, 11)],
    )

    assert parsed == {
        "C1": 1.0,
        "C2": 0.0,
        "C3": 1.0,
        "C4": 1.0,
        "C5": 0.0,
        "C6": 1.0,
        "C7": 1.0,
        "C8": 0.0,
        "C9": 0.0,
        "C10": 1.0,
    }


def test_parse_agent4rec_watch_response_accepts_reason_with_semicolon_literal_backslash_n() -> None:
    parsed = parse_agent4rec_watch_response(
        "C1: Amadeus; WATCH: yes; REASON: Classic cinema with strong performances and emotional depth fitting my exclusive taste.\\n"
        "C2: Good Will Hunting; WATCH: no; REASON: Lacks the international or classic prestige I demand; too mainstream.\\n"
        "C3: Gandhi; WATCH: yes; REASON: Historical drama with profound themes and strong character focus aligns with my preferences.\\n"
        "C4: Dead Man Walking; WATCH: yes; REASON: Intense character-driven narrative with emotional depth and strong performances.\\n"
        "C5: Driving Miss Daisy; WATCH: no; REASON: Perceived as overly sentimental and lacking the intellectual rigor I require.\\n"
        "C6: To Kill a Mockingbird; WATCH: yes; REASON: Classic cinema with thought-provoking themes and exceptional character work.\\n"
        "C7: Like Water for Chocolate; WATCH: yes; REASON: International film with unique artistic style and emotional depth.\\n"
        "C8: Edward Scissorhands; WATCH: no; REASON: Too whimsical and fantasy-oriented; does not match my serious, realistic taste.\\n"
        "C9: Searching for Bobby Fischer; WATCH: no; REASON: Lacks the historical weight and international scope I prioritize.\\n"
        "C10: Sense and Sensibility; WATCH: yes; REASON: Classic period drama with strong performances and emotional resonance.",
        labels=[f"C{i}" for i in range(1, 11)],
    )

    assert parsed == {
        "C1": 1.0,
        "C2": 0.0,
        "C3": 1.0,
        "C4": 1.0,
        "C5": 0.0,
        "C6": 1.0,
        "C7": 1.0,
        "C8": 0.0,
        "C9": 0.0,
        "C10": 1.0,
    }


def test_parse_agent4rec_watch_response_accepts_game_field() -> None:
    parsed = parse_agent4rec_watch_response(
        "ID: C1; GAME: Portal; WATCH: yes; REASON: puzzle game\n"
        "ID: C2; GAME: Dota 2; WATCH: no; REASON: not preferred",
        labels=["C1", "C2"],
    )

    assert parsed == {"C1": 1.0, "C2": 0.0}


def test_parse_agent4rec_preference_response_maps_decisions_by_label() -> None:
    parsed = parse_agent4rec_preference_response(
        "ID: C2; MOVIE: Heat; PREFERENCE: no; REASON: below target\n"
        "ID: C1; MOVIE: Aladdin; PREFERENCE: yes; REASON: meets target",
        labels=["C1", "C2"],
    )

    assert parsed == {"C1": 1.0, "C2": 0.0}


def test_parse_agent4rec_preference_response_accepts_unlabeled_title() -> None:
    parsed = parse_agent4rec_preference_response(
        "C1: Heat; PREFERENCE: no; REASON: below target\n"
        "C2: Aladdin; PREFERENCE: yes; REASON: meets target",
        labels=["C1", "C2"],
    )

    assert parsed == {"C1": 0.0, "C2": 1.0}


def test_parse_agent4rec_watch_response_requires_all_decisions() -> None:
    with pytest.raises(ValueError, match="Missing Agent4Rec watch decisions"):
        parse_agent4rec_watch_response(
            "ID: C1; MOVIE: A; WATCH: yes; REASON: ok",
            labels=["C1", "C2"],
        )


def test_parse_agent4rec_watch_response_rejects_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="Duplicate Agent4Rec candidate labels"):
        parse_agent4rec_watch_response(
            "ID: C1; MOVIE: A; WATCH: yes; REASON: ok\n"
            "ID: C1; MOVIE: B; WATCH: no; REASON: not ok",
            labels=["C1", "C2"],
        )


def test_parse_agent4rec_watch_response_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="Unknown Agent4Rec candidate labels"):
        parse_agent4rec_watch_response(
            "ID: C1; MOVIE: A; WATCH: yes; REASON: ok\n"
            "ID: C3; MOVIE: B; WATCH: no; REASON: not ok",
            labels=["C1", "C2"],
        )


def test_agent4rec_system_prompt_matches_full_social_traits_template() -> None:
    prompt = agent4rec_system_prompt(
        taste="animated comedies; family movies",
        activity="activity description",
        conformity="conformity description",
        diversity="diversity description",
    )

    assert prompt == AGENT4REC_SOCIAL_TRAITS_SYSTEM_PROMPT_TEMPLATE.format(
        taste="animated comedies; family movies",
        activity="activity description",
        conformity="conformity description",
        diversity="diversity description",
    )


def test_agent4rec_system_prompt_omits_missing_profile_parts() -> None:
    prompt = agent4rec_system_prompt(
        taste=None,
        activity="activity description",
        conformity=None,
        diversity="diversity description",
    )

    assert "Your activity trait is described as: activity description" in prompt
    assert "Your conformity trait is described as:" not in prompt
    assert "Your diversity trait is described as: diversity description" in prompt
    assert "Beyond that, your movie tastes are:" not in prompt
    assert "The conformity characteristic measures" not in prompt


def test_agent4rec_system_prompt_uses_forced_items_template_for_taste_only() -> None:
    prompt = agent4rec_system_prompt(
        taste="animated comedies; family movies",
        activity=None,
        conformity=None,
        diversity=None,
    )

    assert prompt == AGENT4REC_FORCED_ITEMS_SYSTEM_PROMPT_TEMPLATE.format(
        taste="animated comedies; family movies"
    )


def test_agent4rec_user_prompt_uses_taste_instruction_only_when_taste_exists() -> None:
    traits_only_prompt = agent4rec_user_prompt(
        candidates="C1. <- Movie ->",
        taste=None,
    )
    taste_prompt = agent4rec_user_prompt(
        candidates="C1. <- Movie ->",
        taste="animated comedies",
    )

    assert "Judge each movie using your available profile" in traits_only_prompt
    assert "You only watch movies which align with your taste" not in traits_only_prompt
    assert "You only watch movies which align with your taste" in taste_prompt


def test_agent4rec_user_prompt_can_use_game_field() -> None:
    prompt = agent4rec_user_prompt(
        candidates="C1. <- Portal ->",
        taste=None,
        entity_field="GAME",
        entity_name="game",
        entity_plural="games",
    )

    assert "Please judge all games in the ##recommended list##" in prompt
    assert "Use this format: ID: [candidate id]; GAME: [game name]; WATCH:" in prompt
    assert "Judge each game using your available profile" in prompt
    assert "Judge each movie using your available profile" not in prompt

    taste_prompt = agent4rec_user_prompt(
        candidates="C1. <- Portal ->",
        taste="puzzle games",
        entity_field="GAME",
        entity_name="game",
        entity_plural="games",
    )
    assert "You only choose games which align with your taste" in taste_prompt
    assert "You only watch movies which align with your taste" not in taste_prompt
    assert "If you don't want to choose a game" in taste_prompt
    assert "If you don't want to watch a game" not in taste_prompt


def test_agent4rec_yes_no_scorer_uses_profile_prompt() -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["i1", "i2", "i3"],
            "item_title": ["Toy Story", "Aladdin", "Heat"],
            "item_genres": ["Animation|Comedy", "Animation", "Crime"],
            "rating": [5, 4, 2],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "candidate_group": ["g1", "g1"],
            "item_title": ["Lion King", "Godfather"],
            "item_rating_mean": [4.153, 4.567],
            "item_genres": ["Animation", "Crime"],
        },
        index=["a", "b"],
    )
    client = FakeClient(
        [
            "ID: C2; MOVIE: Godfather; WATCH: no; REASON: crime movies do not fit the user's taste\n"
            "ID: C1; MOVIE: Lion King; WATCH: yes; REASON: animated movies fit the user's taste"
        ]
    )

    scorer = Agent4RecYesNoScorer(
        client=client,
        model="fake-model",
        candidate_description_columns=("item_title", "item_rating_mean", "item_genres"),
        column_labels={
            "item_title": "movie title",
            "item_rating_mean": "History ratings",
            "item_genres": "genres",
        },
    ).fit(X_train, pd.Series([1, 1, 1], name="target"))
    scores = scorer.score(X_test)

    assert scores.to_dict() == {"a": 1.0, "b": 0.0}
    assert len(client.completions.calls) == 1
    system_prompt = client.completions.calls[0]["messages"][0]["content"]
    user_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "Assume you are a user" in system_prompt
    assert "Your activity trait is described as:" not in system_prompt
    assert "Your conformity trait is described as:" not in system_prompt
    assert "Your diversity trait is described as:" not in system_prompt
    assert "##recommended list##" in user_prompt
    assert "C1. <- Lion King -> <- History ratings:4.15 -> <- genres:Animation ->" in user_prompt
    assert "C2. <- Godfather -> <- History ratings:4.57 -> <- genres:Crime ->" in user_prompt
    assert "Use this format: ID: [candidate id]; MOVIE: [movie name]; WATCH: [yes or no]; REASON: [brief reason]" in user_prompt
    assert "Judge each movie using your available profile" in user_prompt
    assert "You only watch movies which align with your taste" not in user_prompt
    assert "rating from 1 to 5" not in user_prompt
    assert "Toy Story" not in user_prompt


# ---------------------------------------------------------------------------
# Agent4RecYesNoScorer — itemwise prompt style
# ---------------------------------------------------------------------------


def test_agent4rec_yes_no_scorer_rejects_invalid_prompt_style() -> None:
    with pytest.raises(ValueError, match="prompt_style"):
        Agent4RecYesNoScorer(
            client=FakeClient([]),
            model="fake-model",
            candidate_description_columns=("item_title",),
            prompt_style="unknown",
        )


def test_agent4rec_yes_no_scorer_itemwise_uses_bare_yes_no_prompt() -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["i1", "i2", "i3"],
            "item_title": ["Toy Story", "Aladdin", "Heat"],
            "item_genres": ["Animation|Comedy", "Animation", "Crime"],
            "rating": [5, 4, 2],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1"],
            "candidate_group": ["g1"],
            "item_title": ["Lion King"],
            "item_genres": ["Animation"],
        },
        index=["a"],
    )
    client = FakeClient(["yes"])

    scorer = Agent4RecYesNoScorer(
        client=client,
        model="fake-model",
        candidate_description_columns=("item_title", "item_genres"),
        prompt_style="itemwise",
    ).fit(X_train, pd.Series([1, 1, 1], name="target"))
    scores = scorer.score(X_test)

    assert scores.to_dict() == {"a": 1.0}
    assert len(client.completions.calls) == 1
    system_prompt = client.completions.calls[0]["messages"][0]["content"]
    user_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "Your activity trait is described as:" not in system_prompt
    assert "##candidate movie##" in user_prompt
    assert "Would you watch this movie?" in user_prompt
    assert "Answer with exactly one word: yes or no." in user_prompt
    assert "C1" not in user_prompt
    assert "WATCH:" not in user_prompt


def test_agent4rec_yes_no_scorer_itemwise_scores_each_item_separately() -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_id": ["i1"],
            "item_title": ["Toy Story"],
            "item_genres": ["Animation"],
            "rating": [5],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "candidate_group": ["g1", "g2"],
            "item_title": ["Lion King", "Godfather"],
            "item_genres": ["Animation", "Crime"],
        },
        index=["a", "b"],
    )
    client = FakeClient(["yes", "no"])

    scorer = Agent4RecYesNoScorer(
        client=client,
        model="fake-model",
        candidate_description_columns=("item_title", "item_genres"),
        prompt_style="itemwise",
    ).fit(X_train, pd.Series([1], name="target"))
    scores = scorer.score(X_test)

    assert scores.to_dict() == {"a": 1.0, "b": 0.0}
    assert len(client.completions.calls) == 2


def test_agent4rec_yes_no_scorer_itemwise_rejects_group_size_gt_1() -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_id": ["i1"],
            "item_title": ["Toy Story"],
            "item_genres": ["Animation"],
            "rating": [5],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "candidate_group": ["g1", "g1"],
            "item_title": ["Lion King", "Godfather"],
            "item_genres": ["Animation", "Crime"],
        },
        index=["a", "b"],
    )
    scorer = Agent4RecYesNoScorer(
        client=FakeClient(["yes"]),
        model="fake-model",
        candidate_description_columns=("item_title", "item_genres"),
        prompt_style="itemwise",
    ).fit(X_train, pd.Series([1], name="target"))

    with pytest.raises(ValueError, match="exactly one row"):
        scorer.score(X_test)


def test_agent4rec_yes_no_scorer_itemwise_parse_error_includes_raw_response() -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_id": ["i1"],
            "item_title": ["Toy Story"],
            "item_genres": ["Animation"],
            "rating": [5],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1"],
            "candidate_group": ["g1"],
            "item_title": ["Lion King"],
            "item_genres": ["Animation"],
        }
    )
    scorer = Agent4RecYesNoScorer(
        client=FakeClient(["I am not sure, maybe?"]),
        model="fake-model",
        candidate_description_columns=("item_title", "item_genres"),
        prompt_style="itemwise",
    ).fit(X_train, pd.Series([1], name="target"))

    with pytest.raises(ValueError, match="raw_response='I am not sure, maybe\\?'"):
        scorer.score(X_test)


def test_agent4rec_yes_no_scorer_batch_parse_error_includes_raw_response() -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_id": ["i1"],
            "item_title": ["Toy Story"],
            "item_genres": ["Animation"],
            "rating": [5],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1"],
            "candidate_group": ["g1"],
            "item_title": ["Lion King"],
            "item_genres": ["Animation"],
        }
    )
    scorer = Agent4RecYesNoScorer(
        client=FakeClient(["this is not the expected format at all"]),
        model="fake-model",
        candidate_description_columns=("item_title", "item_genres"),
    ).fit(X_train, pd.Series([1], name="target"))

    with pytest.raises(ValueError, match="raw_response="):
        scorer.score(X_test)
    X_train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["i1", "i2", "i3"],
            "item_title": ["Toy Story", "Aladdin", "Heat"],
            "item_genres": ["Animation|Comedy", "Animation", "Crime"],
            "rating": [5, 4, 2],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "candidate_group": ["g1", "g1"],
            "item_title": ["Lion King", "Godfather"],
            "item_genres": ["Animation", "Crime"],
        },
        index=["a", "b"],
    )
    client = FakeClient(
        [
            "ID: C1; MOVIE: Lion King; PREFERENCE: yes; REASON: likely high rating\n"
            "ID: C2; MOVIE: Godfather; PREFERENCE: no; REASON: likely low rating"
        ]
    )

    scorer = Agent4RecPreferenceYesNoScorer(
        client=client,
        model="fake-model",
        target_description=(
            "The user would rate the candidate movie at least 4 out of 5."
        ),
        candidate_description_columns=("item_title", "item_genres"),
        column_labels={
            "item_title": "movie title",
            "item_genres": "genres",
        },
    ).fit(X_train, pd.Series([1, 1, 0], name="target"))

    scores = scorer.score(X_test)

    assert scores.to_dict() == {"a": 1.0, "b": 0.0}
    system_prompt = client.completions.calls[0]["messages"][0]["content"]
    user_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "Your activity trait is described as:" not in system_prompt
    assert "Positive-preference target:" in user_prompt
    assert "rate the candidate movie at least 4 out of 5" in user_prompt
    assert "PREFERENCE: [yes or no]" in user_prompt
    assert "WATCH:" not in user_prompt
    assert "Toy Story" not in user_prompt


def test_agent4rec_yes_no_scorer_requires_build_taste_before_score(tmp_path) -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["i1", "i2", "i3"],
            "item_title": ["Toy Story", "Aladdin", "Heat"],
            "item_genres": ["Animation|Comedy", "Animation", "Crime"],
            "rating": [5, 4, 2],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1"],
            "candidate_group": ["g1"],
            "item_title": ["Lion King"],
            "item_rating_mean": [4.153],
            "item_genres": ["Animation"],
        }
    )
    scoring_client = FakeClient([])
    taste_client = FakeClient([])
    profile_generator = Agent4RecProfileGenerator(
        profile_components=("traits", "taste"),
        taste_client=taste_client,
        taste_client_name="openai",
        taste_model="gpt-4o-mini",
        taste_cache_path=tmp_path / "taste.jsonl",
    )
    scorer = Agent4RecYesNoScorer(
        client=scoring_client,
        model="fake-model",
        profile_generator=profile_generator,
        candidate_description_columns=("item_title", "item_rating_mean", "item_genres"),
    ).fit(X_train, pd.Series([1, 1, 1], name="target"))

    with pytest.raises(RuntimeError, match="Call scorer.build_taste"):
        scorer.score(X_test)

    assert scoring_client.completions.calls == []
    assert taste_client.completions.calls == []


def test_agent4rec_yes_no_scorer_build_taste_then_scores_with_taste(tmp_path) -> None:
    X_train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["i1", "i2", "i3"],
            "item_title": ["Toy Story", "Aladdin", "Heat"],
            "item_genres": ["Animation|Comedy", "Animation", "Crime"],
            "rating": [5, 4, 2],
        }
    )
    X_test = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "candidate_group": ["g1", "g1"],
            "item_title": ["Lion King", "Godfather"],
            "item_rating_mean": [4.153, 4.567],
            "item_genres": ["Animation", "Crime"],
        },
        index=["a", "b"],
    )
    scoring_client = FakeClient(
        [
            "ID: C1; MOVIE: Lion King; WATCH: yes; REASON: animated taste\n"
            "ID: C2; MOVIE: Godfather; WATCH: no; REASON: not animated"
        ]
    )
    taste_client = FakeClient(
        [
            "TASTE: I enjoy animated family movies.\n"
            "REASON: I rated animated movies highly.\n"
            "HIGH RATINGS: animated family movies\n"
            "LOW RATINGS: crime movies"
        ]
    )
    profile_generator = Agent4RecProfileGenerator(
        profile_components=("traits", "taste"),
        taste_client=taste_client,
        taste_client_name="openai",
        taste_model="gpt-4o-mini",
        taste_cache_path=tmp_path / "taste.jsonl",
    )
    scorer = Agent4RecYesNoScorer(
        client=scoring_client,
        model="fake-model",
        profile_generator=profile_generator,
        candidate_description_columns=("item_title", "item_rating_mean", "item_genres"),
        column_labels={
            "item_title": "movie title",
            "item_rating_mean": "History ratings",
            "item_genres": "genres",
        },
    ).fit(X_train, pd.Series([1, 1, 1], name="target"))

    scorer.build_taste(X_test)
    scores = scorer.score(X_test)

    assert scores.to_dict() == {"a": 1.0, "b": 0.0}
    assert len(taste_client.completions.calls) == 1
    taste_messages = taste_client.completions.calls[0]["messages"]
    assert "user gives a rating of 5 for following movies: Toy Story" in taste_messages[1]["content"]
    assert "user gives a rating of 2 for following movies: Heat" in taste_messages[1]["content"]
    system_prompt = scoring_client.completions.calls[0]["messages"][0]["content"]
    user_prompt = scoring_client.completions.calls[0]["messages"][1]["content"]
    assert "Your movie tastes are: enjoy animated family movies." in system_prompt
    assert "You only watch movies which align with your taste" in user_prompt
