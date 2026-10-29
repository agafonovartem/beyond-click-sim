# History-conditioned LLM scorer

Source: `src/beyond_click_sim/scorers/history/` (`yes_no.py`, `regression.py`, `listwise.py`, `prompts.py`, `common.py`, `selection.py`).
Runner wiring (adds dataset config, item-stats/summary modifications, CLI-facing `run_method`): `runners/in_distribution/interaction_prediction/methods/llm_yes_no.py` and sibling `regression_prediction`/`listwise` runner modules.

This document is descriptive only — it explains what the code does, not whether the design is scientifically sound.

The separate [LLMRank adaptation](llmrank_design.md) adds an explicit recency
reminder and sums three rankings of shuffled candidate presentations. Its
MovieLens runner requires temporal task provenance; it does not change the
yes/no or ordinary listwise protocols described here.

## 1. What this scorer is

This is the plain, non-persona LLM response simulator: given a user's **raw interaction history rows** (formatted as text, one line per row) and a set of **explicit candidate rows**, ask an OpenAI-compatible chat model to predict a response — yes/no interaction, yes/no positive-preference, a ranking, or a numeric rating.

It differs from the Agent4Rec-style scorer (`docs/scorers/agent4rec.md`) in one structural way: there is **no persona/system-prompt profile module**. The system prompt is a fixed, generic instruction; all user-specific context is the literal list of the user's past interaction rows, shown verbatim (with whichever columns you configure) in the user prompt. There are no social traits, no generated taste paragraph, no LLM call to build a profile — history formatting is the only user-conditioning mechanism.

It is a **candidate-scorer, not a recommender**: candidates are supplied externally (as `candidate_group_column`-grouped rows); the scorer never generates its own candidate set.

## 2. File map

| File | Responsibility |
|---|---|
| `selection.py` | `select_history_by_user` / `UserHistory` — shared history-window selection (also used by Agent4Rec). |
| `common.py` | Two tiny shared helpers: `_chat_completion_text` (extract text from an OpenAI-style response) and `_format_prompt_value` (numeric/JSON-list value formatting for prompts). |
| `prompts.py` | All prompt templates: interaction yes/no, preference yes/no, interaction listwise, preference listwise, itemwise policy-ranking, regression. |
| `yes_no.py` | `LLMInteractionYesNoScorer` (binary interaction decision) and `LLMPreferenceYesNoScorer` (binary preference-target decision). Response parsers. |
| `regression.py` | `LLMRegressor` — one-row numeric response prediction (e.g. 1-5 rating). |
| `listwise.py` | `LLMInteractionListwiseRankingScorer` and `LLMPreferenceListwiseRankingScorer` — direct ranking of a candidate group. |
| `__init__.py` | Public re-exports. |

All scorer classes implement `Scorer` (`fit(X, y) -> self`, `score(X) -> pd.Series`) from `scorers/base.py`. Unlike the Agent4Rec scorers, there is no separate `build_taste` stage — `fit` alone is sufficient before `score`.

## 3. History selection (shared by every scorer here)

`select_history_by_user` (`selection.py`, identical helper used by Agent4Rec) picks, for each user, the **last `max_history_items` rows in the order they appear in the fit frame** (`select_user_history_positions` in `scorers/constant.py`):

- `max_history_items=None` → keep the full history for that user.
- `max_history_items=0` → keep no history rows (empty history block in the prompt).
- Otherwise → keep the last N rows per user, in original order, then renumber them `H1, H2, ...` starting from the oldest kept row.
- Selection is **positional on the fit frame**, not time-sorted by any explicit timestamp column — "last N" means "last N rows as they appear in `X_train`", so whatever ordering `X_train` already has (typically chronological, but this is a property of the upstream task builder, not of this scorer) determines the window.

This selection happens once, inside `fit()`; `score()` never re-touches it.

## 4. Prompt construction pipeline — what actually gets sent to the LLM

This is the core mechanic shared (with small template differences) by every scorer variant in this package. Concretely, for `LLMInteractionYesNoScorer`:

### Step 1 — `fit(X, y, history_user_ids=None)`

For every requested user, format each selected history row into one line via `_format_item_description`:

```
H{position}. {label_1}: {value_1}; {label_2}: {value_2}; ...
```

- Columns come from `history_description_columns` (configured at construction).
- For each column: if the cell is `NaN` or `""`, it's **dropped from that line entirely** (no `column: None` placeholder).
- Column name → prompt label uses `column_labels.get(column, column)` — i.e. the raw dataframe column name by default, or a friendlier label if provided (e.g. `PREFIXED_ITEM_RATING_MEAN_COLUMN → "average rating"`).
- Values are rendered by `_format_prompt_value` (`common.py`):
  - if `json_list_columns` marks the column and the value is a JSON string encoding a list, it's parsed and rendered as a comma-joined list (`"Action, Adventure"`), or `"none"` if the parsed list is empty;
  - booleans render as `"True"`/`"False"`;
  - integers render as bare ints;
  - other real numbers: if integral in value, render as int, else 2-decimal fixed point (`f"{numeric:.2f}"`);
  - anything else: `str(value)`.
- If **every** configured column is empty/NaN for a row, the line becomes `H{position}. no item description`.

The list of formatted lines is stored per user on `self.history_by_user_`. Nothing is sent to the LLM yet — this is purely local string formatting from already-materialized dataframe rows.

### Step 2 — `score(X)`, per candidate group

For each `candidate_group_column` group (must contain exactly one distinct `user_id`, else `ValueError`):

1. **Batch mode (default, `prompt_style="batch"`)**: build `labels = ["C1", "C2", ..., "Cn"]` for the `n` rows in the group (in group row order — this order is whatever order the input dataframe presents the group in; it is **not re-sorted** by the scorer). Each candidate row is formatted the same way as history rows but via a **separate** column set, `candidate_description_columns`, and a `Ck. ...` label prefix instead of `Hk.`.
2. Fill `INTERACTION_YES_NO_USER_PROMPT_TEMPLATE`:

```
User interaction history:
{history}

Candidate items:
{candidates}

For each candidate, answer whether the user would interact with it.

Required output format:
Return exactly one line for each candidate label.
The text before each colon must match these labels:
{output_labels}
Fill each line after the colon with either yes or no.
```
   where `{history}` is the joined `H1...` lines (or the literal string `"- No interaction history available."` if the user has zero history rows — e.g. a cold-start user, or `max_history_items=0`), `{candidates}` is the joined `C1...` lines, and `{output_labels}` is one `Ck:` line per candidate.
3. System prompt is the fixed `INTERACTION_YES_NO_SYSTEM_PROMPT` (no per-user variation):

```
You are simulating a user's response in a recommender system.
Given the user's interaction history and candidate items, decide whether the user would interact with each candidate.
Use only the provided history and candidate information.
Return only one line per candidate in the required format.
```
4. Messages sent to `client.chat.completions.create(...)`:
```python
[
    {"role": "system", "content": INTERACTION_YES_NO_SYSTEM_PROMPT},
    {"role": "user",   "content": <filled INTERACTION_YES_NO_USER_PROMPT_TEMPLATE>},
]
```
   with `model`, `temperature`, `max_tokens`, and (if configured) `extra_body` (e.g. vLLM's `chat_template_kwargs={"enable_thinking": False}`) passed alongside.
5. Response text is parsed by `parse_yes_no_response(text, labels)`: splits on lines, requires each non-blank line to contain `:`, the part before `:` must be a known label (unknown label → `ValueError`, duplicate label → `ValueError`), the part after must be exactly `"yes"` or `"no"` (case-insensitive) → `1.0`/`0.0`. Any expected label missing from the response → `ValueError`. There is no fuzzy fallback: any parse failure raises and is handled by the runner's per-group retry loop (§7).

2. **Itemwise mode (`prompt_style="itemwise"`)**: only usable when each group has exactly one row (else `ValueError`). Skips the `Ck` labeling entirely; builds `POLICY_RANKING_ITEMWISE_USER_PROMPT_TEMPLATE` (single "Recommended item:" block, no per-line label), asks for a **bare** `yes`/`no` word, parsed by `parse_single_yes_no_response` (must be exactly `"yes"` or `"no"` after stripping, nothing else tolerated). One LLM call per candidate row instead of one per group — more calls, but no multi-candidate list-following required from the model, and errors are isolated to a single row rather than an entire group.

### Preference variant (`LLMPreferenceYesNoScorer`)

Subclasses `LLMInteractionYesNoScorer`; identical `fit`/history logic, but overrides `_build_messages` to use `PREFERENCE_YES_NO_SYSTEM_PROMPT` / `PREFERENCE_YES_NO_USER_PROMPT_TEMPLATE` instead. The user prompt gains an extra block:

```
User feedback history:
{history}

Positive-preference target:
{target_description}

Candidate items:
{candidates}

For each candidate, answer whether the user's response would meet the positive-preference target.
...
```

`target_description` is a required, caller-supplied non-empty string (constructor raises otherwise) — it defines what "positive preference" means for whatever target is being simulated (e.g. "the user gives this movie a rating of 4 or above"). The scorer itself does not infer or validate this text's relationship to the actual target column; it is taken at face value from the caller. Parsing is identical (`parse_yes_no_response`).

### Listwise variants (`listwise.py`)

Both subclass the yes/no scorers and reuse `fit`/history/formatting; only `_build_messages` and the parser differ:

- **`LLMInteractionListwiseRankingScorer`**: asks the model to output a full permutation of all candidate labels, most→least likely to be interacted with, one label per line, using `INTERACTION_LISTWISE_RANKING_USER_PROMPT_TEMPLATE`. This template is explicit about strictness: it states the exact candidate count, tells the model to "silently verify that every allowed label appears exactly once" and self-correct, forbids item names/rank numbers/explanations, and requires the **first character of the response to be `C`** (an anchor meant to suppress conversational preamble). Parsed by `parse_ranked_labels_response` (`listwise.py`): extracts every `\bC\d+\b` token via regex, verifies no unknown/duplicate/missing labels relative to the expected set, else `ValueError`. The ranked list is converted to numeric scores as `score(Ck) = n_candidates - rank_of(Ck)` (rank 1 → highest score), so it composes with the same downstream ranking-metric pipeline as pointwise scores.
- **`LLMPreferenceListwiseRankingScorer`**: same output contract, but ranks by "most likely to meet the positive-preference target" (`PREFERENCE_LISTWISE_RANKING_USER_PROMPT_TEMPLATE`, requires `target_description`).

Both templates explicitly instruct: *"The history is context only: never include history items in the ranking."* — a guard against the model conflating the history block with the candidates to be ranked.

### Regression variant (`LLMRegressor`)

Not a subclass of the yes/no scorer — a separate `Scorer` implementation, one row scored per LLM call (no `candidate_group_column` concept; `score(X)` iterates `X.iterrows()` directly).

- `fit` stores history the same way, keyed by `history_description_columns` (falls back to `item_description_columns`, then to `candidate_description_columns`, if not explicitly given — see §5).
- Each row's message uses `REGRESSION_USER_PROMPT_TEMPLATE`:

```
User interaction history:
{history}

Candidate item:
{candidate}

Task:
{target_description}

Required output format:
{output_instructions}
```
  with system prompt `REGRESSION_SYSTEM_PROMPT` (generic "simulate a numeric response" instruction). Both `target_description` (what's being predicted) and `output_instructions` (how to format the answer, e.g. "Return a single integer from 1 to 5.") are required non-empty strings supplied by the caller — this scorer makes no assumptions about the numeric scale itself beyond what `valid_values` enforces at parse time.
- `parse_regression_value_response`: strict — the entire response, after stripping whitespace, must fullmatch `[+-]?\d+` (a bare integer, no `RATING:` prefix tolerated here, unlike the Agent4Rec regressor). The integer must be a member of `valid_values` (`Sequence[int]`, validated non-bool-ints at construction) or the call raises.

## 5. Column configuration: history vs. candidate columns, and the `item_description_columns` shortcut

Every scorer takes **two independently-configurable column sets**:
- `history_description_columns` — fields shown for each *history* row.
- `candidate_description_columns` — fields shown for each *candidate* row.

They intentionally default to different resolution paths so callers don't have to specify both when they're the same, but can diverge when they need to:

```python
if candidate_description_columns is None:
    candidate_description_columns = item_description_columns
if history_description_columns is None:
    history_description_columns = item_description_columns
if history_description_columns is None:
    history_description_columns = candidate_description_columns
```

`item_description_columns` is a convenience "use the same columns for both" shortcut. Both resolved column tuples must end up non-empty or `__init__` raises.

**Why this separation exists** (per the docstring on `LLMInteractionYesNoScorer`): history rows can safely include information that would leak the target if shown on a candidate — e.g. history can show the user's own `rating` for a past item, while the candidate description for an unseen item obviously cannot show its (unknown) rating. Keeping the two column sets separate is the mechanism that prevents that class of accidental leakage; it does not happen automatically — it is the caller's responsibility to choose disjoint-enough column sets (see the dataset defaults in §7, which do put `rating` only on the history side for ml-1m).

## 6. Response parsing summary

| Parser | Used by | Contract | Failure modes |
|---|---|---|---|
| `parse_yes_no_response` | batch yes/no, preference yes/no | `Ck: yes` / `Ck: no`, one non-blank line per label | empty response; missing `:`; unknown label; duplicate label; answer not exactly yes/no; any expected label absent |
| `parse_single_yes_no_response` | itemwise yes/no | bare `yes` or `no`, whole response | anything else |
| `parse_ranked_labels_response` | both listwise scorers | every `Ck` token found via regex must form a valid permutation of expected labels | no candidate tokens found; unknown label; duplicate label; missing label |
| `parse_regression_value_response` | `LLMRegressor` | whole response is one bare integer in `valid_values` | not a bare integer; integer outside `valid_values` |

None of these parsers attempt partial recovery or fuzzy matching — any deviation from the exact contract raises `ValueError`, which the runner-level retry/error-capture logic (§7) treats as a scoring failure for that group/row.

## 7. Runner-level wiring and modifications (`runners/.../methods/llm_yes_no.py` et al.)

The scorer classes themselves are dataset-agnostic and have no notion of "item stats" or "item summaries" — those are **runner-level prompt-column modifications** applied before constructing the scorer. `run_method(...)` in `llm_yes_no.py` is the orchestration entry point.

### 7.1 Base per-dataset prompt columns

```python
DATASET_PROMPT_COLUMNS = {
    "ml-1m": {
        "history_description_columns": ("item_title", "item_genres", "rating"),
        "candidate_description_columns": ("item_title", "item_genres"),
    },
    "steam": {
        "history_description_columns": ("item_title", "item_genres_json", "item_tags_json", "playtime_forever"),
        "candidate_description_columns": ("item_title", "item_genres_json", "item_tags_json"),
    },
}
```
Note the asymmetry described in §5: ml-1m's history includes the user's own `rating` for past items; candidates never do. `DATASET_JSON_LIST_COLUMNS` marks Steam's `item_genres_json`/`item_tags_json` for JSON-list unpacking (ml-1m has none, since its `item_genres` is presumably already a plain delimited string).

### 7.2 `use_item_stats` — the "item stats" modification

A boolean flag on `run_method`. When `True`, `maybe_add_item_rating_prompt_columns` (`runners/in_distribution/llm_item_stats.py`) **appends** `PREFIXED_ITEM_RATING_STATS_COLUMNS = (item_rating_mean, item_rating_count)` to **both** the history and candidate column tuples:

```python
return {
    "history_description_columns": (*prompt_columns["history_description_columns"], *PREFIXED_ITEM_RATING_STATS_COLUMNS),
    "candidate_description_columns": (*prompt_columns["candidate_description_columns"], *PREFIXED_ITEM_RATING_STATS_COLUMNS),
}
```

These two columns (`item_rating_mean`, `item_rating_count` — see `src/beyond_click_sim/tasks/item_statistics.py`) are **item-level, train-only aggregate statistics** computed upstream by the task builder (`TrainItemRatingStatistics.enrich_items`): for ml-1m, the item's mean rating and interaction count *within the train interactions only*; for Steam, the item's mean prior playtime and interaction count. They are attached to every row (history and candidate alike) as a fixed per-item feature, i.e. they are the same "population popularity/quality prior" regardless of which specific user is being scored.

When `use_item_stats=True`, `item_rating_column_labels(dataset_name, use_item_stats=True)` also supplies friendlier prompt labels (`ITEM_RATING_COLUMN_LABELS`), e.g. for ml-1m: `rating → "user rating"`, `item_rating_mean → "average rating"`, `item_rating_count → "number of prior reviews"`; for Steam: `playtime_forever → "user playtime minutes"`, `item_rating_mean → "average prior playtime minutes"`, `item_rating_count → "number of prior interactions"`. When `use_item_stats=False`, no labels are injected (raw column names would be used, but the columns aren't added to the prompt at all in that case).

Concretely, with item stats on, a candidate line for ml-1m becomes e.g.:
```
C1. Toy Story (1995); genres: Animation|Children's|Comedy; average rating: 4.15; number of prior reviews: 452
```
instead of just:
```
C1. Toy Story (1995); genres: Animation|Children's|Comedy
```
This is a purely additive prompt-column change — it does not alter the templates, the system prompt, or the parsing contract; it only changes what the model is told about each item.

There is one dataset-config check: `_require_item_rating_prompt_config` raises `ValueError` if `use_item_stats=True` is requested for a dataset not present in `ITEM_RATING_COLUMN_LABELS`.

### 7.3 `summary_visibility` — item-summary (LLM-generated movie-plot text) modification

A second, independently-togglable additive modification: `summary_visibility: "none" | "history" | "candidate" | "both"`, resolved via `resolve_item_summary_visibility` (`runners/in_distribution/item_summaries.py`) into `{history: bool, candidate: bool, any: bool}`. `maybe_add_item_summary_prompt_columns` appends the shared `ITEM_SUMMARY_COLUMN` (`TASK_MOVIE_SUMMARY_COLUMN`, ml-1m only — raises for other datasets) to whichever of `history_description_columns` / `candidate_description_columns` were requested, and the labeled column gets the fixed label `"summary"` (`ITEM_SUMMARY_COLUMN_LABEL`). `task_item_summary_metadata` verifies the task actually carries validated, non-missing summary provenance (canonical enrichment metadata with a `source_sha256`) before allowing use, and raises if summaries are requested but not present/complete on the task.

This composes independently with `use_item_stats` — both can be enabled simultaneously, in which case both sets of extra columns are appended (item-stats columns first, then the summary column, per the order the two `maybe_add_*` calls are chained in `run_method`).

### 7.4 `run_method(...)` full option surface

```python
def run_method(
    task, output_dir, *,
    method_name, client_name, model,
    max_candidate_groups,                       # cap number of candidate groups scored (smoke vs full runs)
    max_history_items = 20,                      # per-user history window size passed to the scorer
    temperature = 0.0,
    max_tokens = 256,
    max_llm_attempts = 5,                        # retries per candidate group on parse/API failure
    max_workers = 1,                              # thread-pool concurrency across candidate groups
    use_item_stats = False,                       # §7.2
    summary_visibility = "none",                  # §7.3
    extra_body = None,                            # passed through to chat.completions.create, e.g. vLLM enable_thinking
    scorer_class = LLMInteractionYesNoScorer,      # swap in LLMPreferenceYesNoScorer, listwise variants, etc.
    scorer_kwargs = None,                          # extra kwargs forwarded to scorer_class (e.g. target_description)
    serving_metadata = None,                       # free-form provenance recorded into manifest.json only
    source_metadata = None,
)
```

Beyond prompt-column assembly, `run_method`:
- Builds `X_train`/`X_test` via `task_xy(task)` + `limit_candidate_groups` (caps to the first `max_candidate_groups` distinct groups, in whatever order they appear — used for cheap "smoke" runs vs. unrestricted "full" runs).
- Fits the scorer on `X_train` restricted to history for the *test* users only (`history_user_ids=X_test["user_id"].drop_duplicates()`), i.e. train rows for users who don't appear in the eval set are simply never formatted into history — an efficiency measure with no scoring effect.
- Delegates actual per-group scoring to `_score_groups`/`_score_one_group` (defined once in `llm_yes_no.py`, imported by the Agent4Rec runner too): each candidate group gets up to `max_llm_attempts` independent tries; any group that never succeeds gets its scores left as `NaN` and its errors logged into `llm_errors.jsonl`, without aborting the rest of the run. `max_workers > 1` parallelizes across groups (not within a group) via `ThreadPoolExecutor`.
- Computes **two parallel result payloads** from the same scored output, always both: a pointwise metrics payload (`test` = parsed-rows-only; `test_failure_as_negative` = failed/unparsed rows counted as a hard "no" rather than dropped) and a ranking metrics payload (`test` = parsed-rows-only ranking; `test_failure_as_zero_group` = any group containing a failed row gets hit-rate/NDCG = 0 for that group, rather than being dropped or scored as random). Both are written regardless of `scorer_class` — pointwise scorers get evaluated under a ranking protocol too, and vice versa for listwise scorers, since all of them ultimately produce one `score` per row.
- Writes `manifest.json` (scorer class/config, resolved prompt columns and labels, `uses_item_stats`, `summary_visibility` + summary provenance metadata, decision-rule/parser contract, retry/worker limits, LLM error count, requested-vs-scored candidate-group counts, full task manifest, current git commit) plus `metrics.json` and a ranking metrics file.

## 8. Summary of user-facing options and their effects

| Option | Where set | Effect |
|---|---|---|
| `history_description_columns` / `candidate_description_columns` / `item_description_columns` | scorer `__init__` | Which item/interaction fields are rendered into history vs. candidate lines; the shortcut collapses both to one set. Keeping them distinct is the leakage guard described in §5. |
| `column_labels` | scorer `__init__` | Prompt-facing label per column (defaults to the raw column name). |
| `json_list_columns` | scorer `__init__` | Which columns hold JSON-encoded lists that should be unpacked to a comma-joined string rather than shown as raw JSON text. |
| `max_history_items` | scorer `__init__` | Size of the per-user history window (`None`=all, `0`=none, `N`=last N rows in fit-frame order). |
| `temperature` / `max_tokens` / `extra_body` | scorer `__init__` | Standard chat-completion sampling controls, passed straight through to `chat.completions.create`. |
| `prompt_style` (yes/no scorers only) | scorer `__init__` | `"batch"`: one call per candidate group, `Ck: yes/no` lines. `"itemwise"`: one call per single candidate row, bare yes/no, requires singleton groups. |
| `target_description` | `LLMPreferenceYesNoScorer`, `LLMPreferenceListwiseRankingScorer`, `LLMRegressor` | Defines what the binary/ranking/numeric decision is actually predicting; must be supplied by the caller, not inferred by the scorer. |
| `output_instructions` | `LLMRegressor` | Free-text instruction for how the numeric answer should be formatted. |
| `valid_values` | `LLMRegressor` | The closed integer set the parsed response must belong to; anything else is a parse failure. |
| `use_item_stats` (runner) | `run_method` | Appends item-level train-only aggregate stats (mean rating/playtime + interaction count) to both history and candidate descriptions, dataset-specific labels. |
| `summary_visibility` (runner) | `run_method` | `"none"/"history"/"candidate"/"both"` — appends an LLM-generated item-summary column to history and/or candidate descriptions (ml-1m only; requires validated task provenance). |
| `max_candidate_groups` (runner) | `run_method` | Caps the number of candidate groups scored (smoke-test vs. full-eval runs); no effect on prompt content, only on how many groups are scored. |
| `max_llm_attempts` / `max_workers` (runner) | `run_method` | Per-group retry budget on parse/API failure, and thread-pool concurrency across groups. Does not change prompt content. |
| `scorer_class` / `scorer_kwargs` (runner) | `run_method` | Which scorer variant (`LLMInteractionYesNoScorer`, `LLMPreferenceYesNoScorer`, listwise variants) is instantiated, and any extra constructor kwargs (e.g. `target_description`, `prompt_style`). |
