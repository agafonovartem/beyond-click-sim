# Task: prepare the KuaiRec dataset for the `policy_ranking_agreement` experiment

**Audience:** an autonomous agent working in the `beyond-click-sim` repository.
**Status of this document:** standalone. Everything needed to start is here; no prior conversation is assumed.

---

## 0. Goal

Produce a canonical KuaiRec dataset that the `policy_ranking_agreement` family of runners can consume, so
that the experiment can be run against an **unbiased** reference ordering instead of an exposure-biased one.

### Definition of done

1. `data/raw/kuairec/` contains the raw archive, unpacked, with a `PROVENANCE.json` recording source URLs,
   SHA256s, licence and known deviations. *(Already done — see §3.4. Verify, do not re-download.)*
2. A new adapter `src/beyond_click_sim/data/adapters/kuairec.py` materialises
   `data/canonical/kuairec/v1/{users,items,interactions,manifest}.parquet`.
3. The canonical tables satisfy the column contract in §5, verified by explicit assertions, not by inspection.
4. A preflight notebook (`notebooks/eda_kuairec_canonical.ipynb` or an extension of the existing
   `notebooks/eda_kuairec.ipynb`) demonstrates that every question in §6 has been answered with a measured
   number, and that every pitfall in §7 has been checked.
5. **No existing dataset changes behaviour.** `pytest tests/` stays at its current baseline, and the ml-1m and
   Steam prompt-rendering hashes are byte-identical before and after. Backward compatibility is a hard
   requirement, not a preference — published ml-1m and Steam runs must stay reproducible.

### Explicitly out of scope

- Running any LLM scorer, or any change that spends tokens.
- Tuning the policy roster (that is a separate runner, `runners/in_distribution/policy_tuning_rerank/`).
- Modifying the shared scorer classes' behaviour for ml-1m / Steam / Coat.

### Working rules

- **Additive only.** Every per-dataset setting in this codebase lives in a dict keyed by dataset name. Add a
  `"kuairec"` key; never edit an existing key, and never change a shared default to suit KuaiRec.
- **Measure, don't assume.** Every claim about the data in your final report must come with the number and the
  cell that produced it.
- **Report blockers, don't silently work around them.** If a shared component genuinely cannot accommodate
  KuaiRec, say so and propose the minimal change; do not fork it without flagging.

---

## 1. What the `policy_ranking_agreement` experiment is

### 1.1 The question

> Does a user simulator rank recommender policies in the same order that real data does?

The motivation behind it is the paper's larger claim: that **LLM-simulated labels are a better evaluation
signal for recommenders than offline held-out data**. Policy-ranking agreement is the experiment meant to
support that claim.

### 1.2 The pipeline

Entry point: `runners/in_distribution/policy_ranking_agreement/run.py`.

| step | what happens | code |
| --- | --- | --- |
| 1 | Load canonical tables | `task_builders.py:load_canonical_dataset` |
| 2 | Filter users with < 10 interactions | `MinUserInteractionsFilter`, `src/beyond_click_sim/tasks/filters.py:35` |
| 3 | Split per user 80/20 | `RandomFractionSplitter`, seeded |
| 4 | Sample 1,000 evaluation users | `PostSplitUserSampler` |
| 5 | Fit ~6 policies on train | `src/beyond_click_sim/tasks/policies.py` |
| 6 | Each policy emits a top-k slate per user | `policy.recommend()`, `POLICY_K = 10` |
| 7 | Attach the **real** label to every slate row | `src/beyond_click_sim/tasks/policy_ranking.py:178-184` |
| 8 | A **scorer** attaches a **simulated** label to every slate row | `runners/in_distribution/policy_ranking_agreement/methods/` |
| 9 | Per policy: mean simulated score → simulated utility; mean real target → real utility | `runners/in_distribution/policy_ranking_agreement/methods/common.py` |
| 10 | Rank policies twice, compare with Kendall τ / Spearman ρ | `src/beyond_click_sim/evaluation/policy_ranking.py:policy_ranking_agreement_metrics` |

**Main metric:** `test.kendall_tau` (`runners/in_distribution/policy_ranking_agreement/metrics.py`). scipy's tau-b, so ties are handled.

The policy roster is 6 policies — `RandomPolicy`, `PopularityPolicy`, `ItemKNNPolicy`, `ALSPolicy`,
`BPRPolicy`, `LightGCNTorchPolicy` — with hyperparameters from `configs/policies/*.json`
(see `docs/policy_configs.md`).

### 1.3 The three variants that exist today

| runner | datasets | target | candidates | k | reference |
| --- | --- | --- | --- | --- | --- |
| `policy_ranking_agreement` | ml-1m, steam | `target_interact` | free retrieval from full catalogue | 10 | biased 20% split |
| `policy_ranking_agreement_preference` | ml-1m, steam | `target_like_ge4` / `target_played_120` | free retrieval | 10 | biased 20% split |
| `policy_ranking_agreement_rerank` | coat | `target_like_ge4` | re-rank the user's own held-out pool | 5 | **unbiased (MCAR)** |

**KuaiRec belongs in the third family.** The rerank fork already exists and already solves the structural
problems; read it as the template, not as something to be re-derived.
Key files: `runners/in_distribution/policy_ranking_agreement_rerank/`,
`src/beyond_click_sim/tasks/policy_reranking.py`, `src/beyond_click_sim/tasks/splitters.py:145`.

---

## 2. The MNAR problem — why ml-1m cannot support the claim

### 2.1 What MNAR means here

Ratings in ml-1m are **Missing Not At Random**. A user rated a movie only if (a) some system surfaced it and
(b) the user chose to watch and rate it. Both are functions of popularity and of the platform's own
recommender. The observed matrix is therefore a *biased sample* of the full preference matrix, and it is
biased in exactly the direction a recommender would be scored on.

### 2.2 What we currently do with ml-1m, and why the claim collapses

In the ml-1m instance, the "real" policy ordering (step 7 above) is computed from a **held-out 20% of the same
logged data**. That makes the argument circular:

- **High τ does not support the claim.** If the simulator agrees with the held-out ranking, all we have shown
  is that the simulator reproduces the biased benchmark. But the benchmark's bias is precisely what the paper
  claims simulated labels improve upon. Agreement with the thing you claim to beat is not evidence that you
  beat it.
- **Low τ is uninterpretable.** Disagreement has two indistinguishable explanations: the simulator is wrong,
  or the simulator is right and the biased benchmark is wrong. The experiment has no way to separate them.

Either outcome leaves the claim unsupported. The reference ordering is not ground truth about user preference;
it is a measurement of the logging policy.

### 2.3 The zero-fill makes it concrete and much worse

`src/beyond_click_sim/tasks/policy_ranking.py:178-184`:

```python
recs[self.target_column] = [
    test_targets.get((row[self.user_column], row[self.item_column]), 0)
    for _, row in recs.iterrows()
]
```

Any recommended `(user, item)` pair that is **not** in the user's held-out rows is labelled **0**. And the
default `target_source_column` is `target_interact` (`src/beyond_click_sim/tasks/policy_ranking.py:68`), which is `1` for every observed
ml-1m row. So `target == 0` is exactly equivalent to "this pair was never observed".

Measured on the real seed-0 run
(`outputs/in_distribution/policy_ranking_agreement/20260722T120856Z_ml-1m_policy_ranking_eval_users1000_seed0_popularity_scorer/`):

- 1,000 users × 6 policies × k=10 = **60,000 slate rows**
- **78.46% of them are fabricated negatives** — not observed dislikes, just absences
- per policy: ALS 63.9%, ItemKNN 68.9%, BPR 77.2%, Popularity 80.7%, LightGCN 80.9%, **Random 99.1%**

The resulting "real utilities":

| policy | real utility |
| --- | --- |
| ALSPolicy | 0.3605 |
| ItemKNNPolicy | 0.3106 |
| BPRPolicy | 0.2284 |
| PopularityPolicy | 0.1928 |
| LightGCNPolicy | 0.1912 |
| RandomPolicy | 0.0088 |

`RandomPolicy` does not score 0.0088 because random recommendations are 40× worse for the user. It scores
0.0088 because a random item is almost never in this user's held-out 33-of-3,883-item window. **The real
utility is measuring exposure overlap, not preference.** The ordering it produces is close to a popularity
ordering by construction.

Whatever a simulator is compared against here, it is not user preference.

### 2.4 What is needed instead

A reference ordering built from labels whose *presence* does not depend on the recommender — either randomly
assigned exposure (MCAR) or full observation. Then:

- the real ordering is a statement about preference, not about the log;
- every slate slot carries a real label, so no zero-fill is required;
- high τ means the simulator recovers an ordering that offline biased evaluation *cannot*, which is the claim.

### 2.5 What was already tried: Coat

Coat (Schnabel et al., ICML 2016) has 290 users who rated 24 self-selected coats (MNAR) **and** 16 assigned
uniformly at random (MCAR). The MCAR half is an unbiased reference. The rerank runner and
`notebooks/eda_coat.ipynb` implement and validate this.

**It works structurally but is too small to conclude anything.** From
`notebooks/policy_ranking_agreement_rerank.ipynb`, over all 17 scored methods:

- Kendall τ ranges from −0.067 to +0.467
- the non-LLM `popularity_scorer` floor is τ = 0.414 — **only 2 of 16 LLM scorers beat it**
- under a user-level bootstrap, **all 17 methods' 95% intervals contain the popularity floor**, and 16 of 17
  contain τ = 0
- only 9 of the 15 policy pairs have a reference ordering that survives resampling

Root causes: 290 users, and a 16-item pool from which k=5 is drawn, which compresses every policy's utility
into the range 0.190–0.260 with the top three separated by 0.0034.

**This is why KuaiRec is needed: the same unbiased design, at a scale where the differences resolve.**

---

## 3. KuaiRec

### 3.1 What it is

A Kuaishou short-video dataset (Gao et al., CIKM '22). Its defining property: for 1,411 users and 3,327
videos, **almost every cell of the user × item grid was observed** — 99.62% density. That is not a random
sample of exposure; it is near-complete observation, which is strictly stronger than Coat's MCAR half.

### 3.2 Structure

| file | rows | grain | role |
| --- | --- | --- | --- |
| `big_matrix.csv` | 12,530,806 | interaction | Ordinary logged traffic, 7,176 × 10,728 at 16.3% density. **MNAR — this is the training signal.** |
| `small_matrix.csv` | 4,676,570 | interaction | The fully-observed block, 1,411 × 3,327 at 99.62%. **The unbiased reference.** |
| `item_categories.csv` | 10,728 | video | `feat` = stringified list of anonymised category ids (1–4 per video) |
| `item_daily_features.csv` | 343,341 | (video, date) | author, upload type, duration, show/play/like counts, `video_tag_name` |
| `user_features.csv` | 7,176 | user | activity/follower counts + 18 opaque `onehot_feat*` |
| `social_network.csv` | 472 | user | friend lists (only 6.6% of users) |
| `kuairec_caption_category.csv` | 10,728 | video | Chinese caption, cover text, topic tags, 3-level category path |
| `user_features_raw.csv` | 7,176 | user | **de-anonymised** user features: gender, age band, city, phone, ISP |
| `video_raw_categories_multi.csv` | 26,826 | (video, category) | multi-label raw taxonomy with confidences |

Both matrices share one schema: `user_id`, `video_id`, `play_duration`, `video_duration`, `time`, `date`,
`timestamp`, `watch_ratio`. **There are no ratings, likes or clicks at interaction level.** The only feedback
signal is `watch_ratio = play_duration / video_duration`, which exceeds 1 when a video loops.

### 3.3 Why it solves the MNAR problem

Verified in `notebooks/eda_kuairec.ipynb`:

| property | measured | consequence |
| --- | --- | --- |
| small-matrix density | 99.62% over 1,411 × 3,327 | the reference is computed, not estimated — zero exposure-sampling noise |
| `(user, video)` cells in both matrices | **0** | train/eval separation is structural |
| big-matrix rows inside the eval block | **0** | the block was surgically removed; no overlap-removal step needed, unlike Coat's 366 cells |
| users with zero positives at `watch_ratio ≥ 2` | **0** | Coat lost 53 of 290 users to a degenerate target |
| Spearman(exposure in big, unbiased quality) | **−0.012** | **the popularity shortcut is gone.** On Coat, popularity scored τ=0.414 and beat 14 of 16 LLM scorers |
| Spearman(quality on biased traffic, unbiased quality) | **+0.932**, 1.50× level inflation | item quality transfers almost perfectly — see the warning below |
| utility span at k=5 (random → oracle) | 0.047 → 1.000, span **0.954** | vs Coat's 0.300; **3.2× more room** for policies to differ |
| evaluation users | 1,411 | vs Coat's 290; ~2.2× narrower bootstrap intervals from sample size alone |

**Warning that must survive into the paper.** Because biased and unbiased item quality correlate at ρ = +0.93,
a simulator cannot distinguish itself on this dataset by judging *items* well — a trivial estimator built from
biased data already does that. Whatever separates policies here must come from **personalisation**: matching
items to the right users. State that as the hypothesis; do not discover it afterwards.

**Second caveat.** The evaluation block is a *selected* sub-population. Its 3,327 videos are ~37× more exposed
in ordinary traffic than the other 7,401, so results describe mainstream content only. Nothing here supports a
claim about tail-item recommendation.

### 3.4 Provenance (already fetched)

`data/raw/kuairec/` holds the unpacked archive. Zenodo record `18164998`, DOI `10.5281/zenodo.18164998`.
`KuaiRec.zip` = 431,964,858 bytes, sha256 `c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c`.

**Licence: CC BY-SA 4.0.** Zenodo's metadata says CC BY 4.0, but `LICENSE` inside the archive is
Attribution-ShareAlike 4.0, matching kuairec.com. The archive text is binding; the ShareAlike term belongs in
the paper's data statement.

`PROVENANCE.json` in that directory records per-file hashes and the known deviations. Read it first.

---

## 4. How the scorers work, and what they consume from ml-1m

A **scorer** attaches a simulated label to every slate row (step 8 in §1.2). Three families exist. To prepare
KuaiRec you must know exactly which columns each one reads, because those columns are what the adapter has to
produce.

### 4.1 `popularity_scorer` — the non-LLM control

`runners/in_distribution/policy_ranking_agreement/methods/popularity.py`,
`src/beyond_click_sim/scorers/popularity.py`.

Sums the task target per item over the train frame. **Needs only `item_id` and the target column.** Run it
first: if it already recovers the unbiased ordering, no LLM result on the dataset means anything.

### 4.2 History-conditioned scorer (`llm_yes_no`)

`src/beyond_click_sim/scorers/history/yes_no.py` (`LLMPreferenceYesNoScorer`),
configured in `runners/in_distribution/policy_ranking_agreement/methods/llm_yes_no.py`.

**How it works.** For each user, the train rows are rendered as a numbered history block
(`H1. item_title: Toy Story (1995); item_genres: Animation|Children's|Comedy; rating: 5`). Candidates from
one slate are rendered as `C1`, `C2`, … in one chat request. The model answers `C1: yes`, `C2: no`, …;
`yes → 1.0`, `no → 0.0`.

**No persona, no user covariates — the history is the only user representation.**

Two prompt styles: `batch` (one call per (user, policy) slate) and `itemwise` (one call per item, deduplicated
to one call per unique `(user, item)`).

**What it reads from ml-1m** (`runners/in_distribution/policy_ranking_agreement/methods/llm_yes_no.py`):

```python
DATASET_PROMPT_COLUMNS = {
    "ml-1m": {
        "history_description_columns": ("item_title", "item_genres", "rating"),
        "candidate_description_columns": ("item_title", "item_genres"),
    },
}
```

plus `MAX_HISTORY_ITEMS = 20` (keeps the **latest** N train rows per user) and a `target_description` string
that states the question in the dataset's own terms.

The templates themselves are domain-neutral — `src/beyond_click_sim/scorers/history/prompts.py` contains zero
occurrences of "movie"/"film"/"game". **This family needs no new prompt assets for a new dataset**, only new
column names and a target description.

### 4.3 Agent4Rec scorer (`agent4rec_yes_no`)

`src/beyond_click_sim/scorers/agent4rec/{yes_no,profiles,prompts}.py`,
configured in `runners/in_distribution/policy_ranking_agreement/methods/agent4rec_yes_no.py`.

Fundamentally different: **the user's history is never shown to the scoring model.** Instead the user is
compressed into a persona, and the persona is what conditions the judgement. Three components:

**(a) Social traits** — three ordinal tiers each, computed deterministically from the train frame
(`src/beyond_click_sim/scorers/agent4rec/profiles.py:443`):

| trait | computed from | needs |
| --- | --- | --- |
| activity | per-user row count, binned at p60/p90 | `user_column` |
| conformity | mean squared deviation of the user's rating from the item's mean rating, binned at p25/p80 | `rating_column` |
| diversity | number of distinct genres covering 80% of the user's genre-token mass, binned at p33/p66 | `genre_column` |

Each tier maps to a prose description (`AGENT4REC_*_DESCRIPTIONS` tables in `profiles.py`).

**(b) Taste** — an LLM-generated free-text profile. The user's rating history is bucketed and sent to a
*separate* model (gpt-4o-mini) which returns TASTE/REASON pairs plus HIGH/LOW-rating conclusions. Cached on
disk per (task, taste model, prompt version).

There are currently **three** taste kinds (`src/beyond_click_sim/scorers/agent4rec/profiles.py:33`):
`_SUPPORTED_TASTE_PROMPT_KINDS = frozenset({"rating", "playtime", "coat_rating"})`, dispatched at
`src/beyond_click_sim/scorers/agent4rec/profiles.py:659`. `rating`/`coat_rating` bucket history into ratings 1–5 and **raise on any value outside
that range** (`src/beyond_click_sim/scorers/agent4rec/profiles.py:700`). `playtime` buckets a continuous engagement measure into zero/low/medium/high
with thresholds in minutes.

**(c) Persona line** (optional, added for Coat) — a sentence built from user covariates, e.g.
*"You are a female shopper aged 20-30, from an urban area, and you are very interested in fashion."*
Built by `_build_persona_by_user` in `runners/in_distribution/policy_ranking_agreement_rerank/methods/agent4rec_yes_no.py`.

**What it reads from ml-1m:**

```python
DATASET_CANDIDATE_COLUMNS = {"ml-1m": ("item_title", "item_genres")}
DATASET_COLUMN_LABELS   = {"ml-1m": {"item_title": "movie title", "item_genres": "genres"}}
DATASET_PROFILE_GENERATOR_KWARGS = {"ml-1m": {}}   # all defaults
DATASET_PROMPT_KWARGS   = {"ml-1m": {}}            # domain_name="movie", etc.
DATASET_SUPPORTS_TASTE  = {"ml-1m": True}
```

`{}` means the generator's own defaults apply (`src/beyond_click_sim/scorers/agent4rec/profiles.py:120-127`): `item_column="item_id"`,
`rating_column="rating"`, `genre_column="item_genres"`, `title_column="item_title"`.

### 4.4 Column map: ml-1m → KuaiRec

This is the crux of the preparation job.

| role | ml-1m column | KuaiRec source | notes |
| --- | --- | --- | --- |
| user key | `user_id` | `user_id` | |
| item key | `item_id` | `video_id` | |
| item display name | `title` | `caption` | **Chinese**; 158 of 3,327 eval items have none |
| item category | `genres` (`Action\|Drama`) | `feat` ids → names | **Chinese**; ids are anonymised, see §6 Q6 |
| history context / rating | `rating` (1–5) | `watch_ratio` (continuous, max 571) | **no natural 1–5 scale** |
| binary target | `target_like_ge4` | `watch_ratio >= 2.0` → `target_watch_ratio_ge2` | conventional threshold |
| provenance for splitting | *(none — random split)* | which matrix the row came from | must be synthesised |
| user covariates | `gender`, `age`, `occupation`, `zip_code` | `user_features_raw.csv` | partly Chinese |

---

## 5. The canonical contract to satisfy

Model the adapter on `src/beyond_click_sim/data/adapters/coat.py`. Reference schema
(`data/canonical/coat/v1/`):

```
users.parquet        user_id, raw_user_id, <covariates…>
items.parquet        item_id, raw_item_id, title, genres, <attributes…>
interactions.parquet interaction_id, user_id, item_id, event_type, <signal…>,
                     source, target_*, …
manifest.json        dataset, adapter, version, raw_sources, tables, id_policy,
                     standard_targets, caveats
```

`PolicyRerankingTaskBuilder.build` (`src/beyond_click_sim/tasks/policy_reranking.py:48-57`) validates on entry
that `interactions` carries `user_id`, `item_id`, `source`, the target column and every history-context column;
that `users` carries `user_id`; and that `items` carries `item_id`.

**The `source` column is the one that does not exist in KuaiRec and must be synthesised.**
`SourcePartitionSplitter` (`src/beyond_click_sim/tasks/splitters.py:145`) partitions on it and — as constructed
by the runners, with no arguments — accepts **only** the literal values `self_selected` (→ train) and `random`
(→ test), raising on anything else (`src/beyond_click_sim/tasks/splitters.py:190-196`). See §6 Q2.

Item and user columns are prefixed with `item_` / `user_` at task-build time, so a canonical `title` column is
seen by scorers as `item_title`.

---

## 6. Questions that must be answered during preparation

Answer each with a measured number and record it in the notebook. Several have no obviously right answer —
where that is the case, state the options, pick one, and justify it.

**Q1 — Binary label.** How is the interaction/preference target synthesised from `watch_ratio`?
The conventional threshold is `>= 2.0`, giving a 4.65% positive rate in the small matrix and 7.47% in the big
one. Report the positive rate, the median positives per user, and the number of users with zero positives, for
thresholds in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0}. Also decide whether a `target_interact` analogue is
meaningful at all — on a fully-observed block "an interaction occurred" is true of every cell and therefore
degenerate.

**Q2 — The `source` column.** `big_matrix` → train, `small_matrix` → test is the obvious mapping. But the
splitter's default values are `self_selected` / `random`, and calling a *fully observed* block "random" is
semantically wrong. Choose between: (a) emit the literal defaults and document the misnomer; (b) emit
honest values (`logged` / `fully_observed`) and pass `train_value` / `test_value` through the runner. Option
(b) touches the runner and must be checked for backward compatibility against Coat.

**Q3 — Item text.** Does KuaiRec have titles, descriptions, genres? Report coverage over the **3,327
evaluation items** specifically, not the full 10,728, for: `caption`, `manual_cover_text`, `topic_tag`, the
three category-name levels, and `video_tag_name`. Note that `UNKNOWN` is used as a string sentinel rather than
null, so a naive null count reads as zero.

**Q4 — Item distinctness (Gate A).** Can the simulator tell two candidate items apart from what it is shown?
Measure how many of the 3,327 eval items have a unique rendered description, and how often two items *within
the same user's slate* are indistinguishable. Coat failed this badly enough to need synthesised titles;
KuaiRec's captions are largely unique (3,124 distinct among 3,169) but its category paths are not
(97.4% of items share a full 3-level path with another).

**Q5 — Language.** Which columns are Chinese, and how many strings need translating? Chinese columns are:
`caption`, `manual_cover_text`, `topic_tag`, `first/second/third_level_category_name`
(`kuairec_caption_category.csv`); `category_name`, `root_name`, `parent_name`
(`video_raw_categories_multi.csv`); `video_tag_name` (`item_daily_features.csv`); and
`fre_country`, `fre_country_region`, `fre_province`, `fre_city`, `fre_city_level`, `fre_community_type`
(`user_features_raw.csv`, where `phone_brand` is mixed Latin/Chinese). `gender`, `age_range`,
`user_active_degree`, `upload_type`, `video_type` are already English.

Decide and justify: translate to English, or prompt in Chinese? Prompting in Chinese confounds
"does the simulator rank policies correctly" with "how well does this backbone handle Chinese", and the four
backbones in the roster differ substantially there. A prior translation of a sibling dataset exists at
`quairank_mnar_experiment/semantic_translation_gpt55/` (KuaiRand-Pure, 7,506 items, gpt-5.5) — it does **not**
cover KuaiRec, but the pipeline shape is reusable.

**Q6 — Category ids.** `item_categories.feat` holds anonymised integer ids (31 distinct) with no names shipped.
Prompting with bare integers is meaningless. `notebooks/eda_kuairec.ipynb` §2.2 shows that
`feat[0] == first_level_category_id` for 67% of videos, which transfers 28 of the 31 names (ids 0, 23, 30
remain unnamed). Decide whether to use that recovered mapping, the caption table's category path directly, or
the separate raw taxonomy — and state what happens to the items it does not cover.

**Q7 — Agent4Rec taste kind.** KuaiRec fits **neither** existing taste kind: `rating`/`coat_rating` raise on
values outside 1–5, and `playtime`'s buckets are minutes with game-flavoured prompts. A fourth kind is needed
(`watch_ratio`), with its own bucket edges — natural cuts are 0 (never started), <1 (abandoned), 1–2 (watched
through), ≥2 (the positive) — plus a system prompt, user template, version id, builder, and a dispatch branch
at `src/beyond_click_sim/scorers/agent4rec/profiles.py:659`. This is the same five-piece job the Coat suite was; model it on
`AGENT4REC_COAT_TASTE_*` in `src/beyond_click_sim/scorers/agent4rec/prompts.py`.

**Q8 — Agent4Rec traits.** For each of activity / conformity / diversity, decide enable-or-disable and justify
from the data:
- *activity*: on Coat this was disabled because its tiers varied only through an artifact. On KuaiRec,
  big-matrix activity spans 100 to 16,015 interactions — real variation. Likely enable.
- *conformity*: needs a `rating_column`. Pointing it at raw `watch_ratio` lets the 571-max tail dominate the
  squared deviation. Clip, or use the binary target, and say which.
- *diversity*: needs a `genre_column`. Verify the token mass is not dominated by one degenerate token the way
  Coat's gender token was.

New description tables are needed for whichever traits are enabled — pass them as constructor kwargs
(`src/beyond_click_sim/scorers/agent4rec/profiles.py:111-113`), which requires **no** edit to `profiles.py` trait logic.

**Q9 — History length.** `MAX_HISTORY_ITEMS` is 20 on ml-1m and 24 on Coat, in both cases because that
approximates a full user history. KuaiRec evaluation users have a **median of 390** big-matrix rows. What is
the right cap, and on what basis? The scorer keeps the *latest* N (`src/beyond_click_sim/scorers/history/yes_no.py:58`), and KuaiRec has
real timestamps, so "latest" is meaningful here in a way it is not on Coat. This is now a genuine
hyperparameter.

**Q10 — Duplicate interactions.** `big_matrix` contains **2,229,837 repeat `(user, video)` rows (17.8%)**,
max 2,224 views of one video by one user. Collapse to one row per pair, or keep as separate events? This
changes the fit frame, the history rendering, and every count-based trait. `small_matrix` has no duplicates.

**Q11 — User covariates for the persona.** Which of the 37 `user_features_raw.csv` columns belong in the
persona line, and how do they render grammatically? Note the age distribution is heavily skewed (43% of users
are `12-17`), which is a population fact a persona-based simulator will amplify.

**Q12 — Scale and cost.** Report the projected run dimensions before anything is launched:
1,411 users × 6 policies × k gives the slate-row count; batch mode is one LLM call per (user, policy); itemwise
is one per unique (user, item). Compare against Coat's 8,700 rows / ~1,710 batch calls / 3,409 itemwise calls.
Note that itemwise dedup was effective on Coat only because six policies overlapped heavily on a 16-item pool —
with 3,327 candidates that overlap largely vanishes.

---

## 7. Pitfalls

Each of these fails **silently** — none raises an exception on its own.

**P1 — `kuairec_caption_category.csv` cannot be read with `pd.read_csv`.** Four captions contain raw newlines
inside unquoted fields. The C engine raises `Buffer overflow caught`; the **python engine does not raise** and
returns 10,732 rows instead of 10,728, with torn halves carrying whatever landed in column 0 as `video_id`.
A repairing loader is in `notebooks/eda_kuairec.ipynb` §2.1. Assert the record count is 10,728 and that
`video_id` is contiguous `0..10727`.

**P2 — `UNKNOWN` is a string sentinel, not null.** Present in the caption, category and user tables. Null
counts read as zero, and `UNKNOWN` becomes a legitimate-looking category in a prompt. Filter explicitly
wherever coverage is measured.

**P3 — `item_daily_features.csv` is at `(video_id, date)` grain**, 343,341 rows over 63 dates. A direct merge
on `video_id` fans each item out ~32×. Worse: joining a *daily* popularity count onto a training row leaks the
future if the date is at or after the interaction. Aggregate before joining, and be explicit about the window.

**P4 — Injecting item statistics is the biased signal.** `item_rating_mean`-style features computed over the
big matrix are exactly the exposure-biased quantity the unbiased reference exists to strip out. Showing them
to the simulator lets it partly reproduce the biased evaluation. There is a guard for this, but it reads
`if use_item_stats and dataset_name == "coat"`
(`runners/in_distribution/policy_ranking_agreement_rerank/methods/llm_yes_no.py:515`) — a hardcoded string, so it will
**silently not fire for kuairec**. Fix it to key off a set.

**P5 — Do not put an exposure proxy in the prompt.** `show_cnt`, `play_cnt`, `video_tag_id` popularity and
similar are records of what the platform surfaced. Same reasoning as P4. Coat's `on_front_page` was excluded
for exactly this reason.

**P6 — `watch_ratio` has an extreme tail.** Max 571 in both matrices; 33,188 big-matrix rows exceed 10. Any
mean over `watch_ratio` is dominated by looping videos. Threshold, or clip first — and this is the same trap
in the conformity trait (Q8).

**P7 — `small_matrix.date` has 181,992 nulls (3.9%)** and is a float column, not int. Any date-based logic
drops or mis-keys those rows. `timestamp` is never null; use it for ordering.

**P8 — `video_raw_categories_multi.csv` does not cover every video.** 10,683 of 10,728 (6 of the missing ones
are in the evaluation set). An inner join silently shrinks the catalogue. Left-join and treat absence as its
own category.

**P9 — Do not break Coat, ml-1m or Steam.** Every per-dataset setting is a dict literal; add a key, never edit
one. Before and after your change, verify: `pytest tests/` at the same baseline, and identical prompt-rendering
hashes for ml-1m and Steam via `scripts/render_coat_prompts.py --dataset ml-1m --hash-only` (and `steam`).
Adding a key to a dict that a test asserts on **by whole-dict equality** will break it — that has happened once
already in this codebase; the fix is to check whether the entry is needed yet, not to edit the test.

**P10 — The persona builder currently hardcodes Coat vocabulary.**
`_build_persona_by_user` (`runners/in_distribution/policy_ranking_agreement_rerank/methods/agent4rec_yes_no.py:153`) unconditionally applies
`_COAT_GENDER_WORDS` and computes an article for `urban`/`suburban`. KuaiRec's gender values are `F`/`M` and
there is no location-article slot. Needs per-dataset value mapping, not a shared body.

**P11 — The history scorer renders history for every train user.** `.fit(X_train, y_train)` passes no
`history_user_ids`, so `src/beyond_click_sim/scorers/history/yes_no.py:146-150` keeps all 7,176 users when only 1,411 are
evaluated — a `select_history_by_user` pass over 12.5M rows and ~172K rendered strings. Agent4Rec already does
this correctly via `profile_user_ids`. One-argument fix, but it matters at KuaiRec's scale.

**P12 — Do not reuse another dataset's prompts.** Pointing KuaiRec at the movie or coat taste suite would
produce plausible-looking output from a "movie taste analyst" shown videos — succeeding, and silently wrong.
This exact bug was shipped once for Coat and caught only on review. `DATASET_SUPPORTS_TASTE` should stay
`False` for kuairec until its own suite exists, so that it fails loudly.

**P13 — The train/eval split is structurally unusual.** The big matrix contains **zero** rows inside the
evaluation block. A CF policy therefore learns the evaluation items' factors from a *disjoint set of users*,
and the evaluation users' factors from a *disjoint set of items*. Nothing in training pairs the two. Do not
"fix" this — it is the property that makes the protocol leakage-free. But expect hyperparameters not to carry
over from ml-1m or Coat, and expect the popularity prior to behave differently.

**P14 — Do not compare KuaiRec results to Coat or ml-1m results.** These are separate experiments with
different targets, different candidate construction and different k. The method *roster* is comparable; the
numbers are not.

---

## 8. Reference index

| what | where |
| --- | --- |
| Base experiment | `runners/in_distribution/policy_ranking_agreement/` |
| Unbiased variant to model on | `runners/in_distribution/policy_ranking_agreement_rerank/` |
| Re-ranking task builder | `src/beyond_click_sim/tasks/policy_reranking.py` |
| Provenance-based splitter | `src/beyond_click_sim/tasks/splitters.py:145` |
| Zero-fill (the MNAR defect) | `src/beyond_click_sim/tasks/policy_ranking.py:178-184` |
| Adapter to model on | `src/beyond_click_sim/data/adapters/coat.py` |
| History scorer | `src/beyond_click_sim/scorers/history/{yes_no,prompts}.py` |
| Agent4Rec scorer | `src/beyond_click_sim/scorers/agent4rec/{yes_no,profiles,prompts}.py` |
| Policy roster config | `configs/policies/preference.json`, `docs/policy_configs.md` |
| Coat preflight (the template) | `notebooks/eda_coat.ipynb` |
| KuaiRec raw EDA (already done) | `notebooks/eda_kuairec.ipynb` |
| Coat results + bootstrap | `notebooks/policy_ranking_agreement_rerank.ipynb` |
| Offline prompt renderer / hash gate | `scripts/render_coat_prompts.py` |
| Known defects | `issues.md` |
| Project working principles | `AGENTS.md` |
