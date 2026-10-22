"""Stage 4 — Monte Carlo outlook: how much waiting capacity comes online, 2026-2030.

One job: turn per-project odds into P10 / P50 / P90 GW by region and year.

  1. Fit the Stage-2 curves (with size bands) on all data through the snapshot.
  2. For each waiting project, having already waited d years, the chance it is
     built in each calendar year 2026-2030 given it hasn't resolved yet
     (same conditioning as the backtest).
  3. Recalibrate those odds with the Platt map learned in the backtest (D13).
  4. Each simulation run (D8: 10,000):
       - draws one aggregate multiplier m from the model's historical error —
         a lognormal on the backtest's actual / expected GW ratios (D12). Its
         centre is the model's recent bias (last 3 cutoffs: the
         model has grown more optimistic as the queue filled with speculative
         projects); its spread comes from all cutoffs.
         Projects in one queue share policy, prices and supply chains, so they
         succeed or fail together; independent coin flips alone would give
         ranges far narrower than the model's real track record;
       - scales every project's odds by m, then draws whether and in which year
         each project is built;
       - sums generation MW by region and year.
  5. P10 / P50 / P90 across runs. Rerunning with a new seed checks stability.

Capacity counted is generation MW (mw_gen); storage is reported separately so
batteries aren't counted as new supply.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import build_probs_by_year, error_distribution, recalibrate, rolling_backtest
from src.clean import OBS_END
from src.survival import (FIG_DIR, GRID, INK, INK_2, OUT_DIR, REGIONS, TECHS, _style, fit_curves,
                          load_clean)

YEARS = list(range(2026, 2031))
N_RUNS = 10_000  # D8
CHUNK = 250
QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}


def load_recalibration(path: Path = OUT_DIR / "recalibration.csv") -> tuple[float, float]:
    s = pd.read_csv(path, header=None, index_col=0)[1]
    return float(s["a"]), float(s["b"])


def project_odds(df: pd.DataFrame, ab: tuple[float, float]) -> tuple[pd.DataFrame, np.ndarray]:
    """Waiting projects and their recalibrated (n, len(YEARS)) build-by-year odds."""
    curves = fit_curves(df, by_size=True, as_of=OBS_END)
    waiting = df[df["outcome"] == "waiting"].copy()
    raw = build_probs_by_year(curves, waiting, waiting["duration_years"], OBS_END, YEARS)
    total = raw.sum(axis=1)
    new = recalibrate(total, ab)
    even = np.full(raw.shape, 1 / len(YEARS))
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total[:, None] > 0, raw / total[:, None], even)
    return waiting, share * new[:, None]


def simulate(odds: np.ndarray, mw: np.ndarray, groups: np.ndarray, n_groups: int,
             mu: float, sd: float, n_runs: int = N_RUNS, seed: int = 0) -> np.ndarray:
    """(n_runs, n_groups, n_years) MW built. groups: integer group id per project."""
    rng = np.random.default_rng(seed)
    n, k = odds.shape
    out = np.zeros((n_runs, n_groups, k))
    onehot = np.zeros((n, n_groups))
    onehot[np.arange(n), groups] = 1.0
    weights = onehot * mw[:, None]                                     # (n, n_groups)
    for start in range(0, n_runs, CHUNK):
        r = min(CHUNK, n_runs - start)
        m = np.exp(rng.normal(mu, sd, size=r))                         # aggregate multiplier
        p = odds[None, :, :] * m[:, None, None]                        # (r, n, k)
        over = p.sum(axis=2, keepdims=True)
        p = np.where(over > 1, p / over, p)                            # odds can't exceed 1
        cum = np.cumsum(p, axis=2)
        u = rng.random((r, n, 1))
        year_idx = (u > cum).sum(axis=2)                               # k = not built by 2030
        for j in range(k):
            out[start:start + r, :, j] = (year_idx == j).astype(float) @ weights
    return out


def summarise(sims: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Quantiles in GW per group x year, plus a cumulative 2026-2030 column."""
    rows = []
    cum = sims.sum(axis=2)
    for gi, label in enumerate(labels):
        for j, year in enumerate(YEARS + ["2026-2030"]):
            v = sims[:, gi, j] if j < len(YEARS) else cum[:, gi]
            row = {"group": label, "year": year, "mean": v.mean() / 1000}
            row.update({q: np.quantile(v, x) / 1000 for q, x in QUANTILES.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def plot_outlook(by_region: pd.DataFrame, national: pd.DataFrame, queue_gw: pd.Series, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    cum = by_region[by_region["year"] == "2026-2030"].set_index("group").loc[REGIONS]
    order = cum["p50"].sort_values().index
    y = np.arange(len(order))
    ax.barh(y, queue_gw.loc[order], color=GRID, height=0.7, label="waiting in queue (generation)")
    ax.barh(y, cum.loc[order, "p50"], color="#2a78d6", height=0.7, label="P50 built by 2030")
    ax.errorbar(cum.loc[order, "p50"], y,
                xerr=[cum.loc[order, "p50"] - cum.loc[order, "p10"], cum.loc[order, "p90"] - cum.loc[order, "p50"]],
                fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, label="P10-P90")
    for yi, region in zip(y, order):
        ax.annotate(f"{cum.loc[region, 'p50']:.0f} of {queue_gw.loc[region]:.0f} GW", (queue_gw.loc[region], yi),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, order)
    ax.set_xlim(0, queue_gw.max() * 1.3)
    ax.set_xlabel("GW of generation", color=INK_2, fontsize=9)
    ax.set_title("Queued today vs likely built by 2030, by region", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)

    ax = axes[1]
    nat = national[national["year"] != "2026-2030"]
    x = nat["year"].astype(int).to_numpy()
    ax.fill_between(x, nat["p10"], nat["p90"], color="#86b6ef", alpha=0.5, linewidth=0, label="P10-P90")
    ax.plot(x, nat["p50"], "o-", color="#2a78d6", linewidth=2, markersize=7,
            markeredgecolor="#fcfcfb", markeredgewidth=1.5, label="P50")
    for xi, v in zip(x, nat["p50"]):
        ax.annotate(f"{v:.0f}", (xi, v), xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=8.5, color=INK_2)
    ax.set_xticks(x)
    ax.set_ylim(0, nat["p90"].max() * 1.15)
    ax.set_ylabel("GW of generation coming online", color=INK_2, fontsize=9)
    ax.set_title("US total from today's queue, by year", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    df = load_clean()
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    ab = load_recalibration()
    rolling = rolling_backtest(df, ab)
    mu, sd = error_distribution(rolling)
    mu_all, _ = error_distribution(rolling, recent=None)
    waiting, odds = project_odds(df, ab)
    mw_gen = waiting["mw_gen"].clip(lower=0).to_numpy()
    mw_sto = waiting["mw_storage"].clip(lower=0).to_numpy()
    region_id = pd.Categorical(waiting["region"], categories=REGIONS).codes
    zeros = np.zeros(len(waiting), dtype=int)

    sims = simulate(odds, mw_gen, region_id, len(REGIONS), mu, sd, seed=0)
    by_region = summarise(sims, REGIONS)
    national = summarise(sims.sum(axis=1, keepdims=True), ["US"])
    storage = summarise(simulate(odds, mw_sto, zeros, 1, mu, sd, seed=0), ["US storage"])
    reseed = summarise(simulate(odds, mw_gen, zeros, 1, mu, sd, seed=1), ["US"])

    # D12 sensitivity: centre the error on all cutoffs instead of the recent ones.
    all_hist = summarise(simulate(odds, mw_gen, zeros, 1, mu_all, sd, seed=0), ["US"])

    # D3 sensitivity: suspended projects treated as withdrawn (never built).
    keep = ~waiting["suspended_flag"].to_numpy()
    no_susp = summarise(simulate(odds[keep], mw_gen[keep], zeros[keep], 1, mu, sd, seed=0), ["US"])

    table = pd.concat([by_region, national, storage])
    table.to_csv(OUT_DIR / "outlook.csv", index=False)

    # By technology: generation MW, except storage-only projects, counted in
    # storage MW (their generation MW is zero). Hybrids count their generation.
    is_sto = (waiting["tech_group"] == "storage").to_numpy()
    mw_tech = np.where(is_sto, mw_sto, mw_gen)
    tech_id = pd.Categorical(waiting["tech_group"], categories=TECHS).codes
    by_tech = summarise(simulate(odds, mw_tech, tech_id, len(TECHS), mu, sd, seed=0), TECHS)
    by_tech["queue_gw"] = by_tech["group"].map(pd.Series(mw_tech, index=waiting.index)
                                               .groupby(waiting["tech_group"]).sum() / 1000)
    by_tech.to_csv(OUT_DIR / "outlook_by_tech.csv", index=False)

    # By state (generation MW); projects with no US state are left out.
    states = sorted(s for s in waiting["state"].dropna().unique() if s != "MX")
    state_id = pd.Categorical(waiting["state"], categories=states).codes
    has_state = state_id >= 0
    by_state = summarise(simulate(odds[has_state], mw_gen[has_state], state_id[has_state], len(states),
                                  mu, sd, seed=0), states)
    by_state["queue_gw"] = by_state["group"].map(waiting.assign(mw=mw_gen).groupby("state")["mw"].sum() / 1000)
    by_state.to_csv(OUT_DIR / "outlook_by_state.csv", index=False)
    queue_gw = waiting.groupby("region")["mw_gen"].sum().clip(lower=0) / 1000
    plot_outlook(by_region, national, queue_gw, FIG_DIR / "outlook.png")

    total_queue = mw_gen.sum() / 1000
    tot = national[national["year"] == "2026-2030"].iloc[0]
    print(f"Error model: actual/expected GW ~ lognormal, median x{np.exp(mu):.2f}, "
          f"P10-P90 x{np.exp(mu - 1.2816 * sd):.2f} to x{np.exp(mu + 1.2816 * sd):.2f}")
    print(f"Waiting in queue: {len(waiting):,} projects, {total_queue:,.0f} GW generation "
          f"+ {mw_sto.sum() / 1000:,.0f} GW storage\n")
    print(f"HEADLINE: of {total_queue:,.0f} GW of queued generation, {tot.p50:,.0f} GW (P50) is likely online "
          f"by 2030 — range {tot.p10:,.0f} to {tot.p90:,.0f} GW (P10-P90), "
          f"{tot.p50 / total_queue:.0%} of the queue.\n")
    show = table.copy()
    show[["mean", "p10", "p50", "p90"]] = show[["mean", "p10", "p50", "p90"]].round(1)
    print("GW of generation (storage row = GW of storage):")
    print(show.pivot(index="group", columns="year", values="p50").round(1).to_string())
    print("\nCumulative 2026-2030, P10 / P50 / P90 GW:")
    cum = show[show["year"] == "2026-2030"].set_index("group")[["p10", "p50", "p90"]]
    cum["queue_gw"] = queue_gw.reindex(cum.index).round(0)
    print(cum.to_string())

    r = reseed[reseed["year"] == "2026-2030"].iloc[0]
    drift = max(abs(r[q] / tot[q] - 1) for q in QUANTILES)
    print(f"\nReseed check (seed 1 vs 0): P10/P50/P90 {r.p10:.1f}/{r.p50:.1f}/{r.p90:.1f} GW, "
          f"max difference {drift:.1%}")
    h = all_hist[all_hist["year"] == "2026-2030"].iloc[0]
    print(f"Error centred on all cutoffs (x{np.exp(mu_all):.2f}) instead of recent: "
          f"P10/P50/P90 {h.p10:.0f}/{h.p50:.0f}/{h.p90:.0f} GW")
    hist = df[df["outcome"] == "built"]
    recent = hist[hist["event_date"].dt.year.between(2021, 2025)]
    print(f"Sanity: queue projects actually built 2021-2025 averaged "
          f"{recent['mw_gen'].clip(lower=0).sum() / 5000:.0f} GW/yr; outlook P50 averages {tot.p50 / 5:.0f} GW/yr")
    s = no_susp[no_susp["year"] == "2026-2030"].iloc[0]
    print(f"Suspended treated as withdrawn (D3): P50 {s.p50:.1f} GW vs {tot.p50:.1f} GW base")


if __name__ == "__main__":
    main()
