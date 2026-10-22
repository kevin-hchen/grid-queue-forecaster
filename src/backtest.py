"""Stage 3 — Backtest: pretend it's an earlier date and score the forecast against reality.

One job: measure how wrong the Stage-2 odds are on outcomes the model couldn't
see (D6). This gates and sizes the Stage-4 outlook.

  1. Rewind the dataset to a cutoff: keep projects queued by then; anything built
     or withdrawn after the cutoff is relabelled "waiting", censored at the cutoff.
  2. Fit the Stage-2 curves on that rewound data only.
  3. For each project still waiting at the cutoff, having already waited d years:
         P(built in the next H years) = (F_built(d + H) - F_built(d)) / S(d)
     i.e. the share of projects still waiting at d that were built between d and
     d + H. Stage 4 uses exactly the same conditioning.
  4. Compare with what actually happened.

Recalibration (D13): the raw odds are overconfident — high predictions too
high, low ones too low. A two-parameter logistic map (Platt scaling),
    logit(p') = a + b * logit(p),
learned from earlier cutoffs' predictions vs outcomes, corrects that. b < 1
means "pull the odds toward the average". The main backtest learns it only from
cutoffs whose 5-year window closed by 2018, so 2020-2025 stays unseen.

Two tests:
  * Main backtest (D6): cutoff end-2019, scored on 2020-2025. Calibration table
    (predicted vs actual build rate per bin), Brier score vs a naive constant
    rate, expected vs actual GW.
  * Rolling backtest (D12): the same at every year-end cutoff 2008-2023 with a
    5-year horizon (the outlook's), recording actual / expected GW at each. The
    spread of those ratios is the model's real-world error; Stage 4 uses it to
    set the width of its ranges.
  * Range check (D14): at each test date 2014-2023, rebuild the whole method —
    recalibration, error range and its centre — from what was known then, make
    the 5-year forecast, and check whether actual GW landed inside P10-P90. A
    well-sized range catches the actual about 8 times in 10. A plain Monte
    Carlo (one average build rate, independent coin flips) is scored the same
    way for comparison.

Known leak: imputed event dates (Stage 1) used group-median lags computed from
the full dataset, including post-cutoff dates. It affects timing only, and the
Stage-2 imputation check shows timing barely moves the curves.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.clean import OBS_END
from src.survival import (FIG_DIR, GRID, INK, INK_2, OUT_DIR, _style, fit_curves, load_clean,
                          lookup, size_band)

CUTOFF = pd.Timestamp("2019-12-31")  # D6
CALIBRATION_TOLERANCE = 0.05          # proposed: within 5 percentage points per bin
N_BINS = 10
FIT_KW = {"by_size": True}            # the model configuration used everywhere (D11)

ROLLING_CUTOFFS = list(range(2008, 2024))     # D12: every year-end with >= 2 years of outcomes
ROLLING_HORIZON = 5                            # years — matches the 2026-2030 outlook
RECAL_CUTOFFS_FOR_BACKTEST = list(range(2008, 2014))  # windows close by 2018: unseen 2020-2025
RANGE_CHECK_CUTOFFS = list(range(2014, 2024))  # D14
RECENT_CUTOFFS = 3                             # D12: centre the error on the latest 3 cutoffs
MIN_SEEN_YEARS = 2                             # a cutoff needs >= 2 years of outcomes to score
Z90 = 1.2815515655446004                       # standard-normal 90th percentile
RECAL_EPS = 0.01                               # clip before logit so p = 0 or 1 stays finite


def years_between(a, b) -> np.ndarray:
    return ((b - a).dt.days / 365.25).to_numpy()


def rewind(df: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF) -> pd.DataFrame:
    """The dataset as it would have looked at `cutoff` — no information after it."""
    d = df[df["q_date"] <= cutoff].copy()
    ended = d["outcome"].isin(["built", "withdrawn"]) & (d["event_date"] <= cutoff)
    d.loc[~ended, "outcome"] = "waiting"
    d.loc[~ended, "event_date"] = cutoff
    d["duration_years"] = years_between(d["q_date"], d["event_date"]).clip(min=0)
    return d


def conditional_build_prob(curve, waited, horizon) -> np.ndarray:
    """P(built within `horizon` years | still waiting after `waited` years in the queue)."""
    s0, fb0, _ = curve.at(waited)
    _, fb1, _ = curve.at(np.asarray(waited) + horizon)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(s0 > 0, (fb1 - fb0) / s0, 0.0)
    return np.clip(p, 0, 1)


def build_probs_by_year(curves: dict, projects: pd.DataFrame, waited: pd.Series,
                        start: pd.Timestamp, years: list[int]) -> np.ndarray:
    """(n_projects, n_years): P(built in calendar year y | waiting at `start`).

    Used by the backtest (yearly totals) and by the Stage-4 simulation.
    """
    out = np.zeros((len(projects), len(years)))
    sizes = size_band(projects)
    edges = [(pd.Timestamp(f"{y}-12-31") - start).days / 365.25 for y in years]
    for (tech, region, size), g in projects.groupby([projects["tech_group"], projects["region"], sizes]):
        pos = projects.index.get_indexer(g.index)
        c = lookup(curves, tech, region, size)
        cum = np.column_stack([conditional_build_prob(c, waited[g.index], e) for e in edges])
        out[pos] = np.diff(np.column_stack([np.zeros(len(g)), cum]), axis=1)
    return out


def predict(df: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF, end: pd.Timestamp = OBS_END,
            **fit_kw) -> pd.DataFrame:
    """One row per project waiting at `cutoff`: predicted vs actual build by `end`."""
    fit_kw = {**FIT_KW, **fit_kw}
    past = rewind(df, cutoff)
    curves = fit_curves(past, as_of=cutoff, **fit_kw)
    waiting = past[past["outcome"] == "waiting"]
    truth = df.loc[waiting.index]

    years = list(range(cutoff.year + 1, end.year + 1))
    by_year = build_probs_by_year(curves, waiting, waiting["duration_years"], cutoff, years)

    out = waiting[["q_id", "region", "tech_group", "q_year", "mw_gen", "mw_storage"]].copy()
    out["waited"] = waiting["duration_years"]
    out["p_built"] = by_year.sum(axis=1)
    out["built"] = (truth["outcome"] == "built") & (truth["event_date"] > cutoff) & (truth["event_date"] <= end)
    out["built_year"] = truth["event_date"].dt.year.where(out["built"])

    # Naive baseline: the share of already-resolved projects that were built,
    # applied to everyone. This is the ratio people usually quote.
    resolved = past[past["outcome"] != "waiting"]
    out.attrs["naive_rate"] = float((resolved["outcome"] == "built").mean())
    out.attrs["by_year"] = pd.DataFrame(by_year, index=out.index, columns=years)
    return out


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), RECAL_EPS, 1 - RECAL_EPS)
    return np.log(p / (1 - p))


def fit_recalibration(p, y, iters: int = 100) -> tuple[float, float]:
    """Platt scaling: (a, b) maximising the likelihood of logit(p') = a + b * logit(p).

    Damped Newton steps — many raw predictions are exactly 0, which makes the
    undamped fit diverge.
    """
    X = np.column_stack([np.ones(len(p)), _logit(p)])
    y = np.asarray(y, float)
    w = np.array([0.0, 1.0])
    for _ in range(iters):
        q = 1 / (1 + np.exp(-X @ w))
        grad = X.T @ (q - y)
        hess = (X * (q * (1 - q))[:, None]).T @ X
        w -= np.clip(np.linalg.solve(hess + np.eye(2), grad), -1, 1)
    return float(w[0]), float(w[1])


def recalibrate(p, ab: tuple[float, float]) -> np.ndarray:
    a, b = ab
    return 1 / (1 + np.exp(-(a + b * _logit(p))))


def learn_recalibration(df: pd.DataFrame, cutoffs, horizon=ROLLING_HORIZON) -> tuple[float, float]:
    """Fit Platt scaling on pooled predictions from several historical cutoffs."""
    preds = [predict(df, pd.Timestamp(f"{y}-12-31"), pd.Timestamp(f"{y + horizon}-12-31")) for y in cutoffs]
    pooled = pd.concat(preds)
    return fit_recalibration(pooled["p_built"], pooled["built"])


def apply_recalibration(pred: pd.DataFrame, ab: tuple[float, float]) -> pd.DataFrame:
    """Recalibrate total build odds; spread the change across years in proportion.

    Projects whose raw odds are 0 (waited past where the curve has data) get the
    recalibrated odds spread evenly across the years.
    """
    out = pred.copy()
    by_year = pred.attrs["by_year"]
    raw = by_year.sum(axis=1).to_numpy()
    new = recalibrate(raw, ab)
    even = np.full(by_year.shape, 1 / by_year.shape[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(raw[:, None] > 0, by_year.to_numpy() / raw[:, None], even)
    out["p_raw"] = raw
    out["p_built"] = new
    out.attrs = {**pred.attrs, "by_year": pd.DataFrame(share * new[:, None], index=by_year.index,
                                                       columns=by_year.columns)}
    return out


def calibration_table(pred: pd.DataFrame, n_bins: int = N_BINS) -> pd.DataFrame:
    """Equal-count bins of predicted probability: mean predicted vs actual build rate."""
    bins = pd.qcut(pred["p_built"].rank(method="first"), n_bins, labels=False)
    t = pred.groupby(bins).agg(n=("built", "size"), predicted=("p_built", "mean"), actual=("built", "mean"),
                               p_low=("p_built", "min"), p_high=("p_built", "max"))
    t["gap"] = t["actual"] - t["predicted"]
    t["pass"] = t["gap"].abs() <= CALIBRATION_TOLERANCE
    t.index.name = "bin"
    return t


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def yearly_build(pred: pd.DataFrame) -> pd.DataFrame:
    """Expected vs actual projects and GW of generation built per calendar year."""
    by_year = pred.attrs["by_year"]
    mw = pred["mw_gen"].clip(lower=0)
    rows = []
    for year in by_year.columns:
        p = by_year[year]
        hit = pred["built_year"] == year
        rows.append({"year": year, "expected_n": p.sum(), "actual_n": int(hit.sum()),
                     "expected_gw": (p * mw).sum() / 1000, "actual_gw": mw[hit].sum() / 1000})
    return pd.DataFrame(rows)


def rolling_backtest(df: pd.DataFrame, ab: tuple[float, float], cutoffs=ROLLING_CUTOFFS,
                     horizon=ROLLING_HORIZON, obs_end: pd.Timestamp = OBS_END) -> pd.DataFrame:
    """Actual / expected GW built within `horizon` years at each historical cutoff,
    for the raw model and the recalibrated one.

    Recent cutoffs (2021, 2023) only have 4 and 2 years of outcomes so far. They
    are still forecast over the full 5 years, recalibrated, and then scored on
    the years that have happened — so they stay comparable and carry the most
    recent evidence about drift.
    """
    rows = []
    for y in cutoffs:
        cut = pd.Timestamp(f"{y}-12-31")
        raw = predict(df, cut, pd.Timestamp(f"{y + horizon}-12-31"))
        rec = apply_recalibration(raw, ab)
        seen = [c for c in raw.attrs["by_year"].columns if c <= obs_end.year]
        mw = raw["mw_gen"].clip(lower=0)
        act_gw = mw[raw["built"]].sum() / 1000  # truth only exists through OBS_END
        raw_gw = (raw.attrs["by_year"][seen].sum(axis=1) * mw).sum() / 1000
        rec_gw = (rec.attrs["by_year"][seen].sum(axis=1) * mw).sum() / 1000
        rows.append({"cutoff": y, "scored_through": min(y + horizon, obs_end.year), "years": len(seen),
                     "actual_gw": act_gw, "raw_expected_gw": raw_gw, "raw_ratio": act_gw / raw_gw,
                     "recal_expected_gw": rec_gw, "recal_ratio": act_gw / rec_gw})
    return pd.DataFrame(rows)


def error_distribution(rolling: pd.DataFrame, recent: int | None = RECENT_CUTOFFS) -> tuple[float, float]:
    """(mu, sd) of log(actual / expected GW) for the recalibrated model.

    mu: mean over the most recent `recent` cutoffs (None = all); sd: all cutoffs.
    """
    logs = np.log(rolling.sort_values("cutoff")["recal_ratio"].to_numpy())
    mu = logs[-recent:].mean() if recent else logs.mean()
    return float(mu), float(logs.std(ddof=1))


def range_check(df: pd.DataFrame, test_cutoffs=RANGE_CHECK_CUTOFFS) -> pd.DataFrame:
    """Out-of-sample check of the whole forecasting method (D14).

    At each test date c, using only data known at c: learn the recalibration
    from cutoffs whose 5-year window had closed, measure the error ratios at
    every earlier cutoff with >= 2 years of outcomes, centre and size the range
    exactly as Stage 4 does, then forecast the next 5 years (fewer for recent
    test dates) and compare with what actually happened.

    The range is the lognormal error band applied to the expected GW; the
    project-level coin flips the simulation adds are negligible next to it.
    """
    rows = []
    for c in test_cutoffs:
        cut = pd.Timestamp(f"{c}-12-31")
        known = rewind(df, cut)
        full = [y for y in ROLLING_CUTOFFS if y + ROLLING_HORIZON <= c]
        ab = learn_recalibration(known, full)
        past = [y for y in ROLLING_CUTOFFS if y <= c - MIN_SEEN_YEARS]
        roll = rolling_backtest(known, ab, past, obs_end=cut)

        end = min(pd.Timestamp(f"{c + ROLLING_HORIZON}-12-31"), OBS_END)
        fc = apply_recalibration(predict(df, cut, pd.Timestamp(f"{c + ROLLING_HORIZON}-12-31")), ab)
        seen = [y for y in fc.attrs["by_year"].columns if y <= end.year]
        mw = fc["mw_gen"].clip(lower=0)
        expected = (fc.attrs["by_year"][seen].sum(axis=1) * mw).sum() / 1000
        actual = mw[fc["built"]].sum() / 1000
        row = {"test_cutoff": c, "years": len(seen), "actual_gw": actual, "model_expected_gw": expected}
        for name, recent in [("recent", RECENT_CUTOFFS), ("all", None)]:
            mu, sd = error_distribution(roll, recent)
            row[f"{name}_p10"] = expected * np.exp(mu - Z90 * sd)
            row[f"{name}_p50"] = expected * np.exp(mu)
            row[f"{name}_p90"] = expected * np.exp(mu + Z90 * sd)
            row[f"{name}_hit"] = row[f"{name}_p10"] <= actual <= row[f"{name}_p90"]

        # Plain Monte Carlo: every waiting project gets the same chance — the
        # share of resolved projects that were built — spread evenly over 5
        # years, with independent draws (normal approximation to the sum).
        resolved = known[known["outcome"] != "waiting"]
        rate = float((resolved["outcome"] == "built").mean()) * len(seen) / ROLLING_HORIZON
        mean, sd_mw = rate * mw.sum() / 1000, np.sqrt((mw ** 2).sum() * rate * (1 - rate)) / 1000
        row.update({"plain_p10": mean - Z90 * sd_mw, "plain_p50": mean, "plain_p90": mean + Z90 * sd_mw})
        row["plain_hit"] = row["plain_p10"] <= actual <= row["plain_p90"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_range_check(rc: pd.DataFrame) -> pd.DataFrame:
    """Hit rate, typical P50 error and typical range width for each method."""
    out = {}
    for name in ["recent", "all", "plain"]:
        err = np.log(rc[f"{name}_p50"] / rc["actual_gw"])
        out[name] = {"hits": f"{int(rc[f'{name}_hit'].sum())}/{len(rc)}",
                     "median_abs_p50_error": float(np.exp(np.median(np.abs(err))) - 1),
                     "median_range_width_pct": float(np.median((rc[f"{name}_p90"] - rc[f"{name}_p10"])
                                                               / rc[f"{name}_p50"]))}
    return pd.DataFrame(out).T


def plot_backtest(cal: pd.DataFrame, rolling: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    lim = max(cal["predicted"].max(), cal["actual"].max()) * 100 * 1.1
    ax.fill_between([0, lim], [-5, lim - 5], [5, lim + 5], color=GRID, alpha=0.6, linewidth=0,
                    label="within 5 points")
    ax.plot([0, lim], [0, lim], color=INK_2, linewidth=1, linestyle="--", label="perfect calibration")
    ax.plot(cal["predicted"] * 100, cal["actual"] * 100, "o-", color="#2a78d6", linewidth=2,
            markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=1.5, label="model (10 equal-size bins)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f"Predicted % built {CUTOFF.year + 1}-{OBS_END.year}", color=INK_2, fontsize=9)
    ax.set_ylabel("Actual % built", color=INK_2, fontsize=9)
    ax.set_title(f"Project-level calibration, fit through {CUTOFF.year}", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style(ax)

    ax = axes[1]
    ax.axhline(1, color=INK_2, linewidth=1, linestyle="--")
    for col, color, label in [("raw_ratio", "#eb6834", "raw model"),
                              ("recal_ratio", "#2a78d6", "recalibrated")]:
        ax.plot(rolling["cutoff"], rolling[col], "o-", color=color, linewidth=2, markersize=7,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=label)
    for _, r in rolling.dropna(subset=["recal_ratio"]).iterrows():
        ax.annotate(f"{r.recal_ratio:.2f}", (r.cutoff, r.recal_ratio), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK_2)
    ax.set_ylim(0, max(1.6, rolling[["raw_ratio", "recal_ratio"]].max().max() * 1.15))
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_xticks(rolling["cutoff"][::2])
    ax.set_xlabel("Cutoff year (model fit on data up to here)", color=INK_2, fontsize=9)
    ax.set_ylabel("Actual / expected GW, next 5 years", color=INK_2, fontsize=9)
    ax.set_title("Aggregate error at each historical cutoff (1 = perfect)", loc="left", fontsize=11, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    df = load_clean()
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    raw = predict(df)
    ab_early = learn_recalibration(df, RECAL_CUTOFFS_FOR_BACKTEST)
    pred = apply_recalibration(raw, ab_early)
    cal = calibration_table(pred)
    cal_raw = calibration_table(raw)
    yearly = yearly_build(pred)
    naive = pred.attrs["naive_rate"]
    b_model = brier(pred["p_built"], pred["built"])
    b_raw = brier(raw["p_built"], raw["built"])
    b_naive = brier(np.full(len(pred), naive), pred["built"])

    # Final recalibration for the outlook: every cutoff with a full 5-year window.
    full = [y for y in ROLLING_CUTOFFS if y + ROLLING_HORIZON <= OBS_END.year]
    ab_final = learn_recalibration(df, full)
    rolling = rolling_backtest(df, ab_final)
    pd.Series({"a": ab_final[0], "b": ab_final[1], "cutoffs": " ".join(map(str, full))}).to_csv(
        OUT_DIR / "recalibration.csv", header=False)
    rc = range_check(df)
    rc.to_csv(OUT_DIR / "range_check.csv", index=False)
    rc_summary = summarise_range_check(rc)

    pred.to_csv(OUT_DIR / "backtest_predictions.csv", index=False)
    cal.to_csv(OUT_DIR / "backtest_calibration.csv")
    yearly.to_csv(OUT_DIR / "backtest_yearly.csv", index=False)
    rolling.to_csv(OUT_DIR / "backtest_rolling.csv", index=False)
    plot_backtest(cal, rolling, FIG_DIR / "backtest.png")

    show = cal.copy()
    for c in ["predicted", "actual", "gap", "p_low", "p_high"]:
        show[c] = (show[c] * 100).round(1)
    print(f"Main backtest: fit through {CUTOFF.date()}, scored on {CUTOFF.year + 1}-{OBS_END.year}.")
    print(f"{len(pred):,} projects waiting at the cutoff; {int(pred['built'].sum()):,} built by {OBS_END.year}.\n")
    print(f"Recalibration learned from cutoffs {RECAL_CUTOFFS_FOR_BACKTEST}: "
          f"a = {ab_early[0]:.2f}, b = {ab_early[1]:.2f}\n")
    print("Calibration after recalibration (percent):")
    print(show.to_string())
    print(f"\n{int(cal['pass'].sum())}/{len(cal)} bins within {CALIBRATION_TOLERANCE * 100:.0f} points "
          f"(raw model: {int(cal_raw['pass'].sum())}/{len(cal_raw)}); "
          f"largest gap {cal['gap'].abs().max() * 100:.1f} points (raw: {cal_raw['gap'].abs().max() * 100:.1f}).")
    print(f"Brier: recalibrated {b_model:.4f}, raw {b_raw:.4f}, naive constant {naive:.1%} {b_naive:.4f} "
          f"-> skill vs naive {(1 - b_model / b_naive):+.1%} (raw {(1 - b_raw / b_naive):+.1%})")
    tot = yearly[["expected_n", "actual_n", "expected_gw", "actual_gw"]].sum()
    print(f"Totals: expected {tot.expected_n:,.0f} projects / {tot.expected_gw:,.1f} GW; "
          f"actual {tot.actual_n:,.0f} projects / {tot.actual_gw:,.1f} GW")

    print(f"\nRolling backtest (recalibration a = {ab_final[0]:.2f}, b = {ab_final[1]:.2f}, "
          f"learned from cutoffs {full}):")
    print(rolling.round(2).to_string(index=False))

    print("\nRange check (D14): method rebuilt at each test date from what was known then. GW of generation.")
    cols = ["test_cutoff", "years", "actual_gw", "recent_p10", "recent_p50", "recent_p90", "recent_hit",
            "all_p50", "all_hit", "plain_p10", "plain_p50", "plain_p90", "plain_hit"]
    print(rc[cols].round(0).to_string(index=False))
    print("\nSummary (recent = this project's method; all = centred on all history; plain = plain Monte Carlo):")
    print(rc_summary.to_string())


if __name__ == "__main__":
    main()
