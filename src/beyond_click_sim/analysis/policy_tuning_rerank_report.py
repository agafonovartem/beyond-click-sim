"""Report panels for the Coat re-ranking tuner (`tasks/policy_tuning_rerank.py`).

A sibling of `policy_tuning_report.py`, not an extension of it. That module's
panels are built for the ml-1m protocol — ten split seeds, a `TPESampler` that
resamples configs, and an inner validation split carved out of train — and more
than half of them have nothing to read here:

* one seed, so no cross-seed stability, shared-config selection, or seed noise;
* an exhaustive `GridSampler` with `n_trials == grid_size`, so search coverage is
  100% by construction and there are no duplicate trials to measure determinism
  from;
* the objective is `mean_hit_rate` over a fixed-size slate, not `ndcg@10` over a
  free-retrieval list.

What replaces them is the thing this protocol has and ml-1m does not: the
validation labels are also the evaluation labels, so the panels below are mostly
about how much that costs. See `docs/policy_configs.md`.

Two artifacts are read together — the production roster (tuned on every
evaluation user) and the diagnostic (tuned on half, scored on both halves).
Only the diagnostic carries a `holdout` arm, so only it can measure the curse.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from beyond_click_sim.analysis.policy_tuning_report import (
    apply_style,
    save_open_figures,
)
from beyond_click_sim.tasks.policy_tuning import default_artifacts_dir

__all__ = [
    "RerankTuningArtifacts",
    "apply_style",
    "load_rerank_artifacts",
    "panel_cost",
    "panel_curse_vs_margins",
    "panel_grid_landscape",
    "panel_provenance",
    "panel_roster_comparison",
    "panel_selected_roster",
    "panel_winners_curse",
    "save_open_figures",
]

FIGSIZE = (11.0, 4.2)
BASELINE_COLOR = "#888888"
TUNED_COLOR = "#2b6cb0"
CURSE_COLOR = "#c05621"


@dataclass(frozen=True)
class RerankTuningArtifacts:
    """Flat tables built from one production and one diagnostic artifact."""

    production: dict[str, Any]
    diagnostic: dict[str, Any] | None
    identity: pd.DataFrame
    selection: pd.DataFrame
    trials: pd.DataFrame
    curse: pd.DataFrame
    paths: dict[str, Path]

    @property
    def tuned_policies(self) -> list[str]:
        return sorted(self.selection.loc[self.selection["tuned"], "policy"])

    @property
    def baseline_policies(self) -> list[str]:
        return sorted(self.selection.loc[~self.selection["tuned"], "policy"])

    def grid(self, policy: str) -> pd.DataFrame:
        """Every evaluated point for one policy, unused columns dropped."""

        table = self.trials[self.trials["policy"] == policy].copy()
        return table.dropna(axis=1, how="all")

    def __repr__(self) -> str:
        return (
            f"RerankTuningArtifacts(policies={len(self.selection)}, "
            f"trials={len(self.trials)}, "
            f"diagnostic={'yes' if self.diagnostic else 'no'})"
        )


def _artifact_name(dataset: str, target: str, n_users: int, seed: int) -> str:
    return f"{dataset}_{target}_rerank_users{n_users}_seed{seed}.json"


def load_rerank_artifacts(
    dataset: str = "coat",
    artifacts_dir: str | Path | None = None,
    *,
    target: str = "target_like_ge4",
    seed: int = 0,
    production_users: int | None = None,
    diagnostic_users: int | None = None,
) -> RerankTuningArtifacts:
    """Load the production and diagnostic artifacts into flat tables.

    With both user counts left as None the pair is discovered from the
    directory: the artifact tuned on the most users is the production roster,
    and the one tuned on fewer is the diagnostic. Passing them explicitly pins
    the pair when more than two are present.
    """

    artifacts_dir = (
        Path(artifacts_dir).expanduser().resolve()
        if artifacts_dir is not None
        else default_artifacts_dir()
    )
    found: dict[int, Path] = {}
    for path in sorted(artifacts_dir.glob(f"{dataset}_{target}_rerank_users*_seed{seed}.json")):
        n_users = int(path.stem.split("_users")[1].split("_seed")[0])
        found[n_users] = path
    if not found:
        raise FileNotFoundError(
            f"No re-ranking tuning artifacts for {dataset!r} in {artifacts_dir}. "
            "Tune first with: uv run python -m "
            "runners.in_distribution.policy_tuning_rerank.run"
        )

    prod_users = production_users if production_users is not None else max(found)
    if prod_users not in found:
        raise FileNotFoundError(
            f"No artifact tuned on {prod_users} users; present: {sorted(found)}"
        )
    if diagnostic_users is not None:
        diag_users = diagnostic_users
    else:
        smaller = [n for n in found if n < prod_users]
        diag_users = max(smaller) if smaller else None

    paths = {"production": found[prod_users]}
    production = json.loads(paths["production"].read_text(encoding="utf-8"))
    diagnostic = None
    if diag_users is not None:
        paths["diagnostic"] = found[diag_users]
        diagnostic = json.loads(paths["diagnostic"].read_text(encoding="utf-8"))

    identity = _identity_table(production, diagnostic)
    selection = _selection_table(production)
    trials = _trials_table(production, diagnostic)
    curse = _curse_table(diagnostic)
    return RerankTuningArtifacts(
        production=production,
        diagnostic=diagnostic,
        identity=identity,
        selection=selection,
        trials=trials,
        curse=curse,
        paths=paths,
    )


def _identity_table(
    production: dict[str, Any], diagnostic: dict[str, Any] | None
) -> pd.DataFrame:
    rows = []
    for arm, payload in (("production", production), ("diagnostic", diagnostic)):
        if payload is None:
            continue
        identity = payload["identity"]
        leakage = identity["leakage"]
        rows.append({
            "arm": arm,
            "dataset": identity["dataset"],
            "protocol": identity["protocol"],
            "objective": identity["objective"],
            "policy_k": identity["policy_k"],
            "target": identity["target_source_column"],
            "search_space_revision": identity.get("search_space_revision"),
            "fit_rows": identity["fit_frame"]["n_rows"],
            "fit_users": identity["fit_frame"]["n_users"],
            "dropped_overlap_rows": identity["splitter"].get("dropped_overlap_rows_"),
            "tuning_users": leakage["n_tuning_users"],
            "holdout_users": leakage["n_holdout_users"],
            "eval_users_in_run": leakage["n_eval_users_in_run"],
            "overlaps_eval_users": leakage["overlaps_eval_users"],
            "search_spaces_sha256": identity["search_spaces_sha256"][:12],
            "generated_at": payload.get("generated_at"),
            "git_commit": (payload.get("git_commit") or "")[:8],
        })
    return pd.DataFrame(rows).set_index("arm")


def _selection_table(production: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, entry in sorted(production["policies"].items()):
        rows.append({
            "policy": name,
            "tuned": entry["tuned"],
            "objective": entry["val_objective"],
            "grid_size": entry["grid_size"],
            "n_boundary_params": len(entry.get("boundary_params", [])),
            "boundary_params": "; ".join(entry.get("boundary_params", [])) or "-",
            "selected_params": json.dumps(entry["selected_params"], sort_keys=True),
        })
    return pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)


def _trials_table(
    production: dict[str, Any], diagnostic: dict[str, Any] | None
) -> pd.DataFrame:
    rows = []
    for arm, payload in (("production", production), ("diagnostic", diagnostic)):
        if payload is None:
            continue
        for name, entry in payload["policies"].items():
            selected = entry["selected_params"]
            for trial in entry["trials"]:
                if trial["state"] != "complete":
                    continue
                metrics = trial["metrics"]
                mugm = metrics.get("ranking", {}).get("macro_by_user_group_mean", {})
                row = {
                    "arm": arm,
                    "policy": name,
                    "number": trial["number"],
                    "objective": trial["value"],
                    "is_selected": trial["params"] == selected,
                    "tuning_utility": metrics.get("mean_hit_rate"),
                    "holdout_utility": metrics.get("holdout_mean_hit_rate"),
                    "fallback_users": metrics.get("rank_candidates_fallback_users"),
                    "seconds": metrics.get("fit_recommend_seconds"),
                    "ndcg@5": mugm.get("ndcg@5"),
                    "hit_rate@5": mugm.get("hit_rate@5"),
                }
                if row["holdout_utility"] is not None:
                    row["gap"] = row["tuning_utility"] - row["holdout_utility"]
                for key, value in trial["params"].items():
                    row[f"param__{key}"] = value
                rows.append(row)
    return pd.DataFrame(rows)


def _curse_table(diagnostic: dict[str, Any] | None) -> pd.DataFrame:
    if diagnostic is None or not diagnostic.get("curse_summary"):
        return pd.DataFrame()
    rows = []
    for name, d in sorted(diagnostic["curse_summary"]["per_policy"].items()):
        rows.append({
            "policy": name,
            "grid_points": d["n_grid_points"],
            "tuning_utility": d["tuning_utility"],
            "holdout_utility": d["holdout_utility"],
            "selected_gap": d["delta"],
            "grid_mean_gap": d["grid_mean_delta"],
            "curse": d["excess_over_grid"],
            "curse_lo": d["excess_ci95"][0],
            "curse_hi": d["excess_ci95"][1],
        })
    return pd.DataFrame(rows).sort_values("curse", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------


def panel_provenance(arts: RerankTuningArtifacts) -> pd.DataFrame:
    """Identity of both arms, plus the invariants the protocol guarantees."""

    identity = arts.identity
    print("Artifacts:")
    for arm, path in arts.paths.items():
        print(f"  {arm:<12} {path.name}")
    print()

    checks = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": "PASS" if ok else "FAIL", "detail": detail})

    prod = identity.loc["production"]
    check(
        "fit frame is the runner's train split",
        bool(prod["fit_rows"] == 6594 and prod["fit_users"] == 290),
        f"{prod['fit_rows']} rows / {prod['fit_users']} users",
    )
    check(
        "pool overlap removed from train",
        bool(prod["dropped_overlap_rows"] == 366),
        f"dropped {prod['dropped_overlap_rows']} rows",
    )
    check(
        "objective matches the runner's utility",
        prod["objective"] == "mean_hit_rate",
        f"{prod['objective']} at k={prod['policy_k']}",
    )
    check(
        "production tuned on every evaluation user",
        bool(prod["tuning_users"] == prod["eval_users_in_run"]),
        f"{prod['tuning_users']} of {prod['eval_users_in_run']}",
    )
    check(
        "leakage is declared, not implied",
        bool(prod["overlaps_eval_users"]),
        "validation labels ARE the evaluation labels",
    )
    if len(identity) > 1:
        diag = identity.loc["diagnostic"]
        check(
            "diagnostic arms partition the users",
            bool(diag["tuning_users"] + diag["holdout_users"] == diag["eval_users_in_run"]),
            f"{diag['tuning_users']} tuned + {diag['holdout_users']} held out",
        )
        check(
            "both arms share a search space",
            prod["search_spaces_sha256"] == diag["search_spaces_sha256"],
            f"sha {prod['search_spaces_sha256']}, revision {prod['search_space_revision']}",
        )
    frame = pd.DataFrame(checks).set_index("check")
    display_identity = identity.T
    print(display_identity.to_string())
    print()
    return frame


def panel_selected_roster(arts: RerankTuningArtifacts) -> pd.DataFrame:
    """The deliverable: what goes into `configs/policies/preference.json`.

    `boundary_params` is the one tuning defect that looks exactly like a normal
    result — a selection on the edge of its choice list means the optimum is
    probably outside the box, so that policy is reported at a handicap.
    """

    table = arts.selection
    flagged = table[table["n_boundary_params"] > 0]
    if len(flagged):
        print(
            f"{len(flagged)} of {len(table)} policies selected on a grid boundary "
            "(reported at a handicap):"
        )
        for _, row in flagged.iterrows():
            print(f"  {row['policy']}: {row['boundary_params']}")
    else:
        print("No boundary selections: every optimum is interior to its box.")
    print()

    figure, ax = plt.subplots(figsize=(FIGSIZE[0], 0.5 * len(table) + 1.6))
    figure.set_label("selected_roster")
    ordered = table.sort_values("objective")
    colors = [TUNED_COLOR if t else BASELINE_COLOR for t in ordered["tuned"]]
    bars = ax.barh(ordered["policy"], ordered["objective"], color=colors)
    for bar, (_, row) in zip(bars, ordered.iterrows()):
        mark = " !" if row["n_boundary_params"] else ""
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{row['objective']:.4f}  (grid {row['grid_size']}){mark}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel(f"{arts.identity.loc['production', 'objective']} on the tuning users")
    ax.set_title("Selected roster — grey policies have nothing to tune")
    ax.set_xlim(0, max(ordered["objective"]) * 1.25)
    figure.tight_layout()
    return table


def panel_grid_landscape(arts: RerankTuningArtifacts, arm: str = "production"):
    """Marginal effect of each hyperparameter, one row of axes per policy.

    Each point is the mean objective over every grid point sharing that value,
    so a flat line means the axis did not matter and the search spent its budget
    there for nothing. The selected value is marked.
    """

    trials = arts.trials[arts.trials["arm"] == arm]
    policies = [p for p in arts.tuned_policies if len(trials[trials["policy"] == p])]
    param_sets = {
        p: sorted(
            c for c in arts.grid(p).columns
            if c.startswith("param__") and trials[trials["policy"] == p][c].nunique() > 1
        )
        for p in policies
    }
    ncols = max(len(v) for v in param_sets.values())
    figure, axes = plt.subplots(
        len(policies), ncols,
        figsize=(3.0 * ncols, 2.6 * len(policies)),
        squeeze=False,
    )
    figure.set_label("grid_landscape")

    rows = []
    for r, policy in enumerate(policies):
        table = trials[trials["policy"] == policy]
        selected = table[table["is_selected"]].iloc[0]
        for c in range(ncols):
            ax = axes[r][c]
            if c >= len(param_sets[policy]):
                ax.set_visible(False)
                continue
            column = param_sets[policy][c]
            marginal = table.groupby(column)["objective"].agg(["mean", "max", "count"])
            ax.plot(range(len(marginal)), marginal["mean"], "o-", color=TUNED_COLOR, label="mean")
            ax.plot(range(len(marginal)), marginal["max"], "^--", color=CURSE_COLOR,
                    alpha=0.7, label="max")
            chosen = selected[column]
            if chosen in marginal.index:
                ax.axvline(list(marginal.index).index(chosen), color="black",
                           lw=1.0, ls=":", alpha=0.8)
            ax.set_xticks(range(len(marginal)))
            ax.set_xticklabels([str(v) for v in marginal.index], fontsize=7, rotation=45)
            ax.set_title(f"{policy.replace('Policy','')} · {column[len('param__'):]}", fontsize=9)
            ax.tick_params(labelsize=7)
            if c == 0:
                ax.set_ylabel("objective", fontsize=8)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="best")
            spread = float(marginal["mean"].max() - marginal["mean"].min())
            rows.append({
                "policy": policy,
                "param": column[len("param__"):],
                "selected": chosen,
                "marginal_spread": spread,
                "n_values": len(marginal),
            })
    figure.suptitle(
        "Grid landscape — dotted line is the selected value; a flat curve is a wasted axis",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    summary = pd.DataFrame(rows).sort_values(
        ["policy", "marginal_spread"], ascending=[True, False]
    ).reset_index(drop=True)
    return summary


def panel_winners_curse(arts: RerankTuningArtifacts) -> pd.DataFrame:
    """How much of the selected score is selection luck rather than quality.

    Keeping the best of G noisy estimates biases the winner's score upward. Here
    that bias lands in the reported reference ordering, because the labels
    selected on are the labels reported against.

    The estimate subtracts the *same* tuned-minus-holdout gap averaged over the
    policy's whole grid. That average never conditions on which point won, so it
    isolates the two halves' difficulty difference and leaves selection bias
    behind. A one-point grid therefore reports exactly zero — nothing was
    selected, so nothing could be inflated — which is a free correctness check.
    """

    if arts.curse.empty:
        print(
            "No diagnostic artifact: run with --tuning-user-fraction 0.5 to "
            "measure the curse."
        )
        return arts.curse

    table = arts.curse
    figure, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(FIGSIZE[0], 0.46 * len(table) + 2.4),
        gridspec_kw={"width_ratios": [1.15, 1]},
    )
    figure.set_label("winners_curse")

    ordered = table.sort_values("curse")
    y = np.arange(len(ordered))
    colors = [BASELINE_COLOR if g <= 1 else CURSE_COLOR for g in ordered["grid_points"]]
    # One call per row: `ecolor` takes a single colour, and handing it a list is
    # only rejected at draw time, so it survives a headless smoke test.
    for yi, (_, row), color in zip(y, ordered.iterrows(), colors):
        ax_left.errorbar(
            [row["curse"]], [yi],
            xerr=[[row["curse"] - row["curse_lo"]], [row["curse_hi"] - row["curse"]]],
            fmt="o", color=color, ecolor=color, elinewidth=1.6, capsize=3, zorder=3,
        )
    ax_left.axvline(0, color="black", lw=0.9)
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(
        [f"{p.replace('Policy','')}  (G={g})" for p, g in
         zip(ordered["policy"], ordered["grid_points"])]
    )
    ax_left.set_xlabel("winner's curse (selected gap − grid mean gap)")
    ax_left.set_title("Measured curse, 95% CI\ngrey = nothing to select, exactly 0", fontsize=10)

    for _, row in table.iterrows():
        if row["grid_points"] <= 1:
            continue
        ax_right.plot(
            [0, 1], [row["tuning_utility"], row["holdout_utility"]],
            "o-", label=row["policy"].replace("Policy", ""),
        )
    ax_right.set_xticks([0, 1])
    ax_right.set_xticklabels(["tuned on", "held out"])
    ax_right.set_ylabel("utility of the selected setting")
    ax_right.set_title("The selected setting, scored on both halves", fontsize=10)
    ax_right.legend(fontsize=8)
    figure.tight_layout()

    n_tuning = int(arts.identity.loc["diagnostic", "tuning_users"])
    n_all = int(arts.identity.loc["diagnostic", "eval_users_in_run"])
    out = table.copy()
    # The curse tracks the standard error of the maximized quantity, which falls
    # as 1/sqrt(n). This is an extrapolation from the half-sized selection, not
    # a second measurement.
    out["curse_at_production"] = out["curse"] * np.sqrt(n_tuning / n_all)
    print(
        f"Measured for a selection on {n_tuning} users; `curse_at_production` "
        f"rescales to the {n_all}-user selection by sqrt({n_tuning}/{n_all}) "
        f"= {np.sqrt(n_tuning / n_all):.3f}. That is an extrapolation.\n"
    )
    return out


def gap_table(arts: RerankTuningArtifacts, policy: str) -> pd.DataFrame:
    """Every grid point of one policy, scored on both diagnostic halves.

    The worked example behind `panel_winners_curse`: the winner is the point
    whose tuning-half score most overstates its held-out score.
    """

    table = arts.trials[
        (arts.trials["arm"] == "diagnostic") & (arts.trials["policy"] == policy)
    ].dropna(axis=1, how="all")
    params = [c for c in table.columns if c.startswith("param__")]
    out = table[[*params, "tuning_utility", "holdout_utility", "gap", "is_selected"]].copy()
    out.columns = [
        *(c[len("param__"):] for c in params),
        "U(tuning)", "U(holdout)", "gap", "selected",
    ]
    return out.sort_values("gap", ascending=False).reset_index(drop=True)


def panel_curse_vs_margins(
    arts: RerankTuningArtifacts,
    run_dir: str | Path,
    *,
    baseline: str = "PopularityPolicy",
    n_bootstrap: int = 2000,
) -> pd.DataFrame:
    """The decisive comparison: is each policy's lead bigger than its own bias?

    Utilities come from the run's `predictions.parquet` with a user-level
    bootstrap, so they are the numbers the experiment actually reports. The
    curse is overlaid because `baseline` has no hyperparameters and therefore
    carries none of it — the asymmetry lands exactly on the comparison of
    interest.
    """

    predictions = pd.read_parquet(Path(run_dir) / "predictions.parquet")
    per_user = predictions.groupby(["policy", "user_id"])["target"].mean().unstack(0)
    policies = sorted(per_user.columns)
    matrix = per_user[policies].to_numpy()
    if np.isnan(matrix).any():
        raise ValueError("a policy is missing users; utilities would not be comparable")

    rng = np.random.default_rng(0)
    index = rng.integers(0, matrix.shape[0], size=(n_bootstrap, matrix.shape[0]))
    boot = matrix[index].mean(axis=1)
    point = matrix.mean(axis=0)
    base = policies.index(baseline)

    curse = (
        arts.curse.set_index("policy")["curse"] * np.sqrt(
            arts.identity.loc["diagnostic", "tuning_users"]
            / arts.identity.loc["diagnostic", "eval_users_in_run"]
        )
        if not arts.curse.empty
        else pd.Series(dtype=float)
    )

    rows = []
    for i, policy in enumerate(policies):
        diff = boot[:, i] - boot[:, base]
        rows.append({
            "policy": policy,
            "utility": point[i],
            "ci_lo": float(np.percentile(boot[:, i], 2.5)),
            "ci_hi": float(np.percentile(boot[:, i], 97.5)),
            f"margin_over_{baseline.replace('Policy', '')}": point[i] - point[base],
            "margin_lo": float(np.percentile(diff, 2.5)),
            "margin_hi": float(np.percentile(diff, 97.5)),
            "curse": float(curse.get(policy, 0.0)),
        })
    table = pd.DataFrame(rows).sort_values("utility", ascending=False).reset_index(drop=True)
    margin_column = f"margin_over_{baseline.replace('Policy', '')}"

    # Two independent ways a lead can fail, and clearing one is not enough:
    # the paired bootstrap CI may include zero (indistinguishable at this sample
    # size), or the lead may be no bigger than the bias that produced it.
    def verdict(row: pd.Series) -> str:
        if row["policy"] == baseline:
            return "-"
        if row[margin_column] <= 0:
            return "behind"
        if row["margin_lo"] <= 0:
            return "not distinguishable"
        if row[margin_column] <= row["curse"]:
            return "within own curse"
        return "established"

    table["verdict"] = table.apply(verdict, axis=1)

    contenders = table[(table["policy"] != baseline) & (table[margin_column] > 0)]
    figure, ax = plt.subplots(figsize=(FIGSIZE[0], 0.55 * len(contenders) + 2.2))
    figure.set_label("curse_vs_margins")
    y = np.arange(len(contenders))
    ax.barh(y - 0.18, contenders[margin_column], height=0.34,
            color=TUNED_COLOR, label=f"lead over {baseline.replace('Policy', '')}")
    ax.barh(y + 0.18, contenders["curse"], height=0.34,
            color=CURSE_COLOR, label="own winner's curse")
    ax.set_yticks(y)
    ax.set_yticklabels([p.replace("Policy", "") for p in contenders["policy"]])
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlabel("utility")
    ax.set_title(
        f"A lead shorter than its own bias bar is not established.\n"
        f"{baseline.replace('Policy', '')} has nothing to tune, so it carries no curse.",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    figure.tight_layout()
    return table


def panel_roster_comparison(
    run_dirs: dict[str, str | Path],
    *,
    n_bootstrap: int = 2000,
) -> pd.DataFrame:
    """Utilities and separability across rosters, e.g. untuned vs tuned.

    `run_dirs` maps a label to a run directory. Separability is the Gate C
    question: how many of the C(n,2) policy pairs have non-overlapping 95% CIs.
    A rank-correlation over pairs that are mostly unresolvable is mostly
    measuring noise.
    """

    frames = []
    summary = []
    for label, run_dir in run_dirs.items():
        predictions = pd.read_parquet(Path(run_dir) / "predictions.parquet")
        per_user = predictions.groupby(["policy", "user_id"])["target"].mean().unstack(0)
        policies = sorted(per_user.columns)
        matrix = per_user[policies].to_numpy()
        rng = np.random.default_rng(0)
        index = rng.integers(0, matrix.shape[0], size=(n_bootstrap, matrix.shape[0]))
        boot = matrix[index].mean(axis=1)
        point = matrix.mean(axis=0)
        lo = np.percentile(boot, 2.5, axis=0)
        hi = np.percentile(boot, 97.5, axis=0)
        for i, policy in enumerate(policies):
            frames.append({
                "roster": label, "policy": policy, "utility": point[i],
                "ci_lo": lo[i], "ci_hi": hi[i],
            })
        resolvable = sum(
            1
            for a in range(len(policies))
            for b in range(a + 1, len(policies))
            if hi[a] < lo[b] or hi[b] < lo[a]
        )
        total = len(policies) * (len(policies) - 1) // 2
        non_random = [i for i, p in enumerate(policies) if p != "RandomPolicy"]
        summary.append({
            "roster": label,
            "spread": float(point.max() - point.min()),
            "spread_excl_random": float(point[non_random].max() - point[non_random].min()),
            "mean_ci_half_width": float(np.mean((hi - lo) / 2)),
            "resolvable_pairs": resolvable,
            "total_pairs": total,
            "top": policies[int(np.argmax(point))],
        })

    table = pd.DataFrame(frames)
    figure, ax = plt.subplots(figsize=FIGSIZE)
    figure.set_label("roster_comparison")
    labels = list(run_dirs)
    policies = sorted(table["policy"].unique())
    width = 0.8 / len(labels)
    for j, label in enumerate(labels):
        subset = table[table["roster"] == label].set_index("policy").loc[policies]
        x = np.arange(len(policies)) + (j - (len(labels) - 1) / 2) * width
        ax.errorbar(
            x, subset["utility"],
            yerr=[subset["utility"] - subset["ci_lo"], subset["ci_hi"] - subset["utility"]],
            fmt="o", capsize=3, label=label,
        )
    ax.set_xticks(np.arange(len(policies)))
    ax.set_xticklabels([p.replace("Policy", "") for p in policies], rotation=20)
    ax.set_ylabel("real utility (mean target over the slate)")
    ax.set_title("Roster comparison, user-level bootstrap 95% CIs", fontsize=10)
    ax.legend(fontsize=9)
    figure.tight_layout()

    print(pd.DataFrame(summary).set_index("roster").to_string())
    print()
    return table


def panel_cost(arts: RerankTuningArtifacts) -> pd.DataFrame:
    """Where the tuning budget went. Sizes any further box revision."""

    rows = []
    for arm in arts.trials["arm"].unique():
        subset = arts.trials[arts.trials["arm"] == arm]
        for policy, group in subset.groupby("policy"):
            rows.append({
                "arm": arm,
                "policy": policy,
                "points": len(group),
                "total_seconds": group["seconds"].sum(),
                "seconds_per_point": group["seconds"].mean(),
                "max_fallback_users": group["fallback_users"].max(),
            })
    table = pd.DataFrame(rows).sort_values(
        ["arm", "total_seconds"], ascending=[True, False]
    ).reset_index(drop=True)
    for arm, group in table.groupby("arm"):
        print(f"{arm}: {group['points'].sum()} points, "
              f"{group['total_seconds'].sum() / 60:.1f} min of fit+rank")
    print()
    return table
