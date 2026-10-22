"""Stage 2 — Competing-risks curves: share built and share withdrawn by years in queue.

One job: for each technology x region group, estimate how the queue resolves over
time. At every point t (years since queue entry) each group has three shares that
sum to 1: built by t, withdrawn by t, and still waiting at t.

Method — Aalen-Johansen (D5). At each time an event happens:
    hazard_k = (# projects with outcome k at t) / (# still at risk at t)
    S(t)     = product over event times of (1 - hazard_built - hazard_withdrawn)
    F_k(t)   = sum over event times of S(just before t) * hazard_k
F_built is the cumulative share built. Withdrawal is a competing outcome, not
censoring: a withdrawn project leaves the risk set and can never be built.
Waiting projects are censored at the snapshot date: they count as at risk up to
the time they've waited, then drop out without an outcome.

Groups (D7): technology x region; groups under MIN_GROUP_N projects borrow the
technology's all-region curve instead of being fitted on their own. With
by_size=True (D11) each group is further split into size bands; a band under
MIN_GROUP_N falls back to its tech x region curve, then to the national one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.clean import CLEAN_PATH, OBS_END

OUT_DIR = Path("outputs")
FIG_DIR = Path("figures")

MIN_GROUP_N = 100  # D7: smaller groups fall back to the technology-level curve
ALL_REGIONS = "ALL"

EVENT_CODE = {"waiting": 0, "built": 1, "withdrawn": 2}

# D11: big projects are built less often (backtest finding). Bands in MW of
# generation, or of storage for storage-only projects.
SIZE_EDGES = [-np.inf, 100, 400, np.inf]
SIZE_LABELS = ["<100MW", "100-400MW", ">400MW"]

TECHS = ["solar", "wind", "storage", "hybrid", "gas", "other"]
REGIONS = ["CAISO", "ERCOT", "ISO-NE", "MISO", "NYISO", "PJM", "SPP", "Southeast", "West"]

# Fixed colour per technology (reference palette, light mode, slots 1-6 in order).
TECH_COLOR = dict(zip(TECHS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


@dataclass
class Curve:
    """Step-function Aalen-Johansen estimate, evaluable at any duration."""

    t: np.ndarray            # distinct event/censoring times (years), ascending
    surv: np.ndarray         # S(t): share still waiting just after t
    f_built: np.ndarray      # F_built(t): cumulative share built by t
    f_withdrawn: np.ndarray  # F_withdrawn(t)
    n: int

    def at(self, x) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(S, F_built, F_withdrawn) at duration(s) x. Before the first time: (1, 0, 0)."""
        x = np.asarray(x, dtype=float)
        i = np.searchsorted(self.t, x, side="right") - 1
        ok = i >= 0
        i = np.clip(i, 0, None)
        s = np.where(ok, self.surv[i], 1.0)
        fb = np.where(ok, self.f_built[i], 0.0)
        fw = np.where(ok, self.f_withdrawn[i], 0.0)
        return s, fb, fw


def aalen_johansen(durations, events, entry=None) -> Curve:
    """Fit the competing-risks curves. events: 0 = censored, 1 = built, 2 = withdrawn.

    entry (optional): duration at which each project starts being observed (left
    truncation). Used by the recent-window fit: a project queued in 2012 only
    counts from the years it had reached when the window opened.

    Ties: at a shared time, events are counted before censoring, so a project
    censored at t is still at risk for events at t (the standard convention).
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events)
    entry = np.zeros_like(durations) if entry is None else np.asarray(entry, dtype=float)
    t, idx = np.unique(durations, return_inverse=True)
    d_built = np.bincount(idx, weights=(events == 1), minlength=len(t))
    d_wd = np.bincount(idx, weights=(events == 2), minlength=len(t))
    n_leaving = np.bincount(idx, minlength=len(t))
    still_in = n_leaving[::-1].cumsum()[::-1]                     # exit >= t
    not_yet_entered = len(entry) - np.searchsorted(np.sort(entry), t, side="left")  # entry >= t
    at_risk = still_in - not_yet_entered                          # entry < t <= exit
    at_risk = np.where(at_risk > 0, at_risk, np.inf)              # no one observed: zero hazard

    h_built = d_built / at_risk
    h_wd = d_wd / at_risk
    surv = np.cumprod(1 - h_built - h_wd)
    surv_before = np.concatenate([[1.0], surv[:-1]])
    f_built = np.cumsum(surv_before * h_built)
    f_wd = np.cumsum(surv_before * h_wd)
    return Curve(t, surv, f_built, f_wd, len(durations))


def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["q_date", "event_date"])


def size_band(df: pd.DataFrame) -> pd.Series:
    mw = df["mw_gen"].where(df["mw_gen"] > 0, df["mw_storage"])
    return pd.cut(mw, SIZE_EDGES, labels=SIZE_LABELS, right=False).astype(str)


def window_entry(df: pd.DataFrame, window_years: float | None, as_of: pd.Timestamp) -> pd.DataFrame:
    """Restrict to experience in the last `window_years` before `as_of` (period approach).

    Each project gets an `entry` duration: how long it had already been queued
    when the window opened (0 if queued inside the window). Projects that ended
    before the window opened contribute nothing and are dropped.
    """
    d = df.copy()
    if window_years is None:
        d["entry"] = 0.0
        return d
    start = as_of - pd.Timedelta(days=round(window_years * 365.25))
    d["entry"] = ((start - d["q_date"]).dt.days / 365.25).clip(lower=0)
    return d[d["duration_years"] >= d["entry"]]


def fit_curves(df: pd.DataFrame, min_n: int = MIN_GROUP_N, window_years: float | None = None,
               as_of: pd.Timestamp = OBS_END, by_size: bool = False) -> dict[tuple, Curve]:
    """One curve per large tech x region group, plus an all-region curve per technology.

    window_years: if set, estimate only from what happened in the last
    `window_years` before `as_of` (period approach). Tested for drift in D10 and
    not adopted — it didn't improve the backtest. min_n counts projects in the window.
    """
    d = window_entry(df, window_years, as_of)
    events = d["outcome"].map(EVENT_CODE)

    def fit(g):
        return aalen_johansen(g["duration_years"], events[g.index], g["entry"])

    curves = {}
    for tech, g in d.groupby("tech_group"):
        curves[(tech, ALL_REGIONS)] = fit(g)
    for (tech, region), g in d.groupby(["tech_group", "region"]):
        if len(g) >= min_n:
            curves[(tech, region)] = fit(g)
    if by_size:
        d["size_band"] = size_band(d)
        for (tech, size), g in d.groupby(["tech_group", "size_band"]):
            if len(g) >= min_n:
                curves[(tech, ALL_REGIONS, size)] = fit(g)
        for (tech, region, size), g in d.groupby(["tech_group", "region", "size_band"]):
            if len(g) >= min_n:
                curves[(tech, region, size)] = fit(g)
    return curves


def lookup(curves: dict, tech: str, region: str, size: str | None = None) -> Curve:
    """Most specific fitted curve: tech x region x size -> tech x region -> tech x size -> tech."""
    for key in [(tech, region, size), (tech, region), (tech, ALL_REGIONS, size)]:
        if key in curves:
            return curves[key]
    return curves[(tech, ALL_REGIONS)]


# ---- sensitivity scenarios (D3 and the imputed-date check) ----------------

def scenario(df: pd.DataFrame, name: str, seed: int = 0) -> pd.DataFrame:
    """base | suspended_as_withdrawn | imputed_resampled."""
    if name == "base":
        return df
    if name == "suspended_as_withdrawn":
        # D3 sensitivity. Their withdrawal date is unknown, so the event sits at
        # the time they've waited (the snapshot) — a late, conservative placement.
        out = df.copy()
        out.loc[out["suspended_flag"], "outcome"] = "withdrawn"
        return out
    if name == "imputed_resampled":
        # Imputed-date check. Median imputation gives every imputed project in a
        # group the same duration (visible as steps in the curves). Here each
        # imputed duration is instead drawn from the real durations of the same
        # outcome and tech x region group. Outcomes are unchanged — only timing.
        # (Dropping imputed rows instead would remove mostly withdrawals and
        # inflate the built share, so it isn't a fair test.)
        rng = np.random.default_rng(seed)
        out = df.copy()
        real = out[out["event_date_source"].isin(["on_date", "prop_date", "wd_date"])]
        pools = real.groupby(["outcome", "tech_group", "region"])["duration_years"].apply(np.asarray)
        fallback = real.groupby("outcome")["duration_years"].apply(np.asarray)
        imputed = out.index[out["event_date_source"] == "imputed"]
        for i in imputed:
            o, t, r = out.at[i, "outcome"], out.at[i, "tech_group"], out.at[i, "region"]
            pool = pools.get((o, t, r), fallback[o])
            # can't end after the snapshot: cap at the time actually observable
            cap = (OBS_END - out.at[i, "q_date"]).days / 365.25
            out.at[i, "duration_years"] = min(rng.choice(pool), cap)
        return out
    raise ValueError(name)


SCENARIOS = ["base", "suspended_as_withdrawn", "imputed_resampled"]
HORIZONS = [3, 5, 10]


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Share built / withdrawn / waiting at 3, 5 and 10 years, per scenario and group."""
    rows = []
    for name in SCENARIOS:
        d = scenario(df, name)
        curves = fit_curves(d)
        for (tech, region), c in curves.items():
            s, fb, fw = c.at(HORIZONS)
            for h, si, b, w in zip(HORIZONS, s, fb, fw):
                rows.append({"scenario": name, "tech_group": tech, "region": region, "n": c.n,
                             "years": h, "built": b, "withdrawn": w, "waiting": si})
    return pd.DataFrame(rows)


# ---- plots ----------------------------------------------------------------

def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _dodge(ends: dict[str, float], gap: float) -> dict[str, float]:
    """Spread end-of-line label positions so they sit at least `gap` apart."""
    out, last = {}, -np.inf
    for name, y in sorted(ends.items(), key=lambda kv: kv[1]):
        out[name] = last = max(y, last + gap)
    return out


def plot_by_tech(curves: dict, path: Path) -> None:
    """National curves: share built (left) and withdrawn (right) by years in queue."""
    grid = np.linspace(0, 15, 301)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, k, title in [(axes[0], 1, "Share built"), (axes[1], 2, "Share withdrawn")]:
        ends = {}
        for tech in TECHS:
            vals = curves[(tech, ALL_REGIONS)].at(grid)[k]
            ax.plot(grid, vals * 100, color=TECH_COLOR[tech], linewidth=2, label=tech)
            ends[tech] = vals[-1] * 100
        for tech, y in _dodge(ends, gap=3.2).items():
            ax.annotate(tech, (grid[-1], y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK_2)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Years since queue entry", color=INK_2, fontsize=9)
        ax.set_xlim(0, 17)
        _style(ax)
    axes[0].set_ylabel("% of projects", color=INK_2, fontsize=9)
    axes[0].set_ylim(0, 100)
    fig.suptitle("How US queue projects resolve, by technology (Aalen-Johansen, all regions)",
                 x=0.06, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def plot_built_by_region(curves: dict, path: Path) -> None:
    """Small multiples: share built by years in queue, one panel per region."""
    grid = np.linspace(0, 12, 241)
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True)
    for ax, region in zip(axes.flat, REGIONS):
        for tech in TECHS:
            own = (tech, region) in curves
            vals = lookup(curves, tech, region).at(grid)[1]
            ax.plot(grid, vals * 100, color=TECH_COLOR[tech], linewidth=1.8 if own else 1.0,
                    linestyle="-" if own else ":", label=tech)
        ax.set_title(region, loc="left", fontsize=10, color=INK)
        _style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Years since queue entry", color=INK_2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("% built", color=INK_2, fontsize=9)
    axes[0, 0].set_ylim(0, 60)
    axes[0, 2].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Share built by years in queue, technology x region "
                 "(dotted = group under 100 projects, uses the national curve)",
                 x=0.06, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    df = load_clean()
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    curves = fit_curves(df)
    plot_by_tech(curves, FIG_DIR / "curves_by_tech.png")
    plot_built_by_region(curves, FIG_DIR / "curves_built_by_region.png")

    table = summary_table(df)
    table.to_csv(OUT_DIR / "survival_summary.csv", index=False)

    national = table[(table["region"] == ALL_REGIONS) & (table["years"] == 10)]
    view = national.pivot(index="tech_group", columns="scenario", values="built").loc[TECHS, SCENARIOS]
    print("Share built within 10 years of queue entry, by technology and scenario:")
    print((view * 100).round(1).to_string())
    print(f"\nWrote {OUT_DIR/'survival_summary.csv'} and 2 figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
