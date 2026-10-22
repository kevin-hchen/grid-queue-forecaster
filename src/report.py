"""README figures — the findings at a glance.

One job: turn the stage outputs into the charts a reader sees first. Run after
src.simulate (it reads outputs/*.csv and the clean dataset).

  headline.png        queued today vs likely built by 2030 vs built in the last 5 years
  outlook_by_tech.png queued vs likely built, by technology
  state_map.png       tile map: GW likely built by 2030, by state
  drift.png           queue entries per year, and the share built within 5 years
  track_record.png    the range check: our range vs a plain Monte Carlo vs actual
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from src.survival import (EVENT_CODE, FIG_DIR, GRID, INK, INK_2, OUT_DIR, TECHS, _style,
                          aalen_johansen, load_clean)

SURFACE = "#fcfcfb"
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#b9b8b2"
# Sequential blue ramp (reference palette steps 150-650), light -> dark.
RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

# Tile-grid positions (row, col): each state one square, roughly geographic.
TILES = {
    "AK": (0, 0), "ME": (0, 11),
    "WI": (1, 6), "VT": (1, 10), "NH": (1, 11),
    "WA": (2, 1), "ID": (2, 2), "MT": (2, 3), "ND": (2, 4), "MN": (2, 5), "IL": (2, 6), "MI": (2, 7),
    "NY": (2, 9), "MA": (2, 10),
    "OR": (3, 1), "NV": (3, 2), "WY": (3, 3), "SD": (3, 4), "IA": (3, 5), "IN": (3, 6), "OH": (3, 7),
    "PA": (3, 8), "NJ": (3, 9), "CT": (3, 10), "RI": (3, 11),
    "CA": (4, 1), "UT": (4, 2), "CO": (4, 3), "NE": (4, 4), "MO": (4, 5), "KY": (4, 6), "WV": (4, 7),
    "VA": (4, 8), "MD": (4, 9), "DE": (4, 10),
    "AZ": (5, 2), "NM": (5, 3), "KS": (5, 4), "AR": (5, 5), "TN": (5, 6), "NC": (5, 7), "SC": (5, 8),
    "DC": (5, 9),
    "OK": (6, 4), "LA": (6, 5), "MS": (6, 6), "AL": (6, 7), "GA": (6, 8),
    "HI": (7, 0), "TX": (7, 4), "FL": (7, 9),
}
MAP_BINS = [0, 1, 2, 4, 8, 16, np.inf]  # GW


def _cum(path, group_col="group") -> pd.DataFrame:
    t = pd.read_csv(OUT_DIR / path)
    return t[t["year"] == "2026-2030"].set_index(group_col)


def _save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_headline(df: pd.DataFrame) -> None:
    us = _cum("outlook.csv").loc["US"]
    queue = df.loc[df["outcome"] == "waiting", "mw_gen"].clip(lower=0).sum() / 1000
    built = df[(df["outcome"] == "built") & df["event_date"].dt.year.between(2021, 2025)]
    past = built["mw_gen"].clip(lower=0).sum() / 1000

    rows = [("Waiting in US queues today", queue, MUTED),
            ("Likely built from it, 2026-2030", us["p50"], BLUE),
            ("Actually built from queues, 2021-2025", past, INK_2)]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    y = np.arange(len(rows))[::-1]
    for yi, (label, v, color) in zip(y, rows):
        ax.barh(yi, v, color=color, height=0.62)
        ax.text(-20, yi, label, ha="right", va="center", fontsize=10, color=INK)
    ax.errorbar(us["p50"], y[1], xerr=[[us["p50"] - us["p10"]], [us["p90"] - us["p50"]]], fmt="none",
                ecolor=INK, elinewidth=1.3, capsize=4)
    ax.text(queue + 15, y[0], f"{queue:,.0f} GW", va="center", fontsize=10, color=INK)
    ax.text(us["p90"] + 15, y[1], f"{us['p50']:,.0f} GW  (range {us['p10']:,.0f}-{us['p90']:,.0f})",
            va="center", fontsize=10, color=INK)
    ax.text(past + 15, y[2], f"{past:,.0f} GW", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, queue * 1.18)
    ax.set_ylim(-0.6, 2.6)
    ax.axis("off")
    fig.text(0.02, 0.93, f"Only about 1 in {queue / us['p50']:.0f} GW waiting in US interconnection queues "
             "is likely to be online by 2030", fontsize=12.5, color=INK)
    fig.text(0.02, 0.85, "GW of generation (storage excluded). Range = P10-P90 from a backtested Monte Carlo.",
             fontsize=9, color=INK_2)
    fig.subplots_adjust(left=0.33, right=0.98, top=0.8, bottom=0.04)
    _save(fig, "headline.png")


def plot_by_tech() -> None:
    t = _cum("outlook_by_tech.csv").loc[TECHS].sort_values("p50")
    labels = {"storage": "storage only*", "hybrid": "solar/wind + storage", "other": "other"}
    fig, ax = plt.subplots(figsize=(9, 3.8))
    y = np.arange(len(t))
    ax.barh(y, t["queue_gw"], color=GRID, height=0.7, label="waiting in queue")
    ax.barh(y, t["p50"], color=BLUE, height=0.7, label="P50 built by 2030")
    ax.errorbar(t["p50"], y, xerr=[t["p50"] - t["p10"], t["p90"] - t["p50"]], fmt="none", ecolor=INK,
                elinewidth=1.2, capsize=3, label="P10-P90")
    for yi, (tech, r) in zip(y, t.iterrows()):
        ax.annotate(f"{r.p50:.0f} of {r.queue_gw:.0f} GW", (r.queue_gw, yi), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, [labels.get(i, i) for i in t.index])
    ax.set_xlim(0, t["queue_gw"].max() * 1.3)
    ax.set_xlabel("GW (generation; *storage-only projects in GW of storage)", color=INK_2, fontsize=9)
    ax.set_title("Queued today vs likely built by 2030, by technology", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    _save(fig, "outlook_by_tech.png")


def plot_state_map() -> None:
    s = _cum("outlook_by_state.csv")
    fig, ax = plt.subplots(figsize=(10, 6.6))
    for state, (r, c) in TILES.items():
        if state in s.index:
            v = s.loc[state, "p50"]
            k = int(np.digitize(v, MAP_BINS[1:-1]))
            color, text = RAMP[k], ("#ffffff" if k >= 3 else INK)
            value = f"{v:.0f}" if v >= 1 else (f"{v:.1f}" if v >= 0.05 else "<0.1")
        else:
            color, text, value = "#efeeea", INK_2, "–"
        ax.add_patch(Rectangle((c, -r), 0.92, 0.92, facecolor=color, edgecolor=SURFACE, linewidth=2))
        ax.text(c + 0.46, -r + 0.6, state, ha="center", va="center", fontsize=9, color=text, weight="bold")
        ax.text(c + 0.46, -r + 0.28, value, ha="center", va="center", fontsize=8.5, color=text)
    # legend
    edges = ["<1", "1-2", "2-4", "4-8", "8-16", "16+"]
    for i, (color, lab) in enumerate(zip(RAMP, edges)):
        ax.add_patch(Rectangle((2.2 + i * 0.85, 0.35), 0.8, 0.35, facecolor=color, edgecolor=SURFACE))
        ax.text(2.6 + i * 0.85, 0.25, lab, ha="center", va="top", fontsize=8.5, color=INK_2)
    ax.text(2.2, 0.85, "GW likely online by 2030 (P50)", fontsize=9, color=INK_2)
    ax.set_xlim(-0.2, 12.2)
    ax.set_ylim(-7.3, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.02, 0.95, "Where the new capacity is likely to come online, 2026-2030", fontsize=12.5, color=INK)
    fig.text(0.02, 0.91, "GW of generation from projects in today's queues; tiles are states, roughly in place. "
             "– = no queued projects in the data.", fontsize=9, color=INK_2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01)
    _save(fig, "state_map.png")


def plot_drift(df: pd.DataFrame) -> None:
    """Projects entering the queue each year, and the share of them built within 5 years."""
    years = list(range(2000, 2021))  # cohorts with a full 5 years observed by end-2025
    entries, built5 = [], []
    for y in years:
        g = df[df["q_year"] == y]
        entries.append(len(g))
        built5.append(aalen_johansen(g["duration_years"], g["outcome"].map(EVENT_CODE)).at(5)[1] * 100)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.bar(years, entries, color=MUTED, width=0.8)
    ax.set_title("Projects entering US queues each year", loc="left", fontsize=11, color=INK)
    _style(ax)
    ax = axes[1]
    ax.plot(years, built5, "o-", color=BLUE, linewidth=2, markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5)
    for y, v in zip(years, built5):
        if y in (2003, 2013, 2020):
            ax.annotate(f"{v:.0f}%", (y, v), xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=9, color=INK_2)
    ax.set_ylim(0, max(built5) * 1.25)
    ax.set_title("Share of each year's entrants built within 5 years", loc="left", fontsize=11, color=INK)
    _style(ax)
    for a in axes:
        a.set_xlabel("Year the project entered the queue", color=INK_2, fontsize=9)
        a.set_xticks(range(2000, 2021, 5))
    fig.tight_layout()
    _save(fig, "drift.png")


def plot_track_record() -> None:
    rc = pd.read_csv(OUT_DIR / "range_check.csv")
    x = rc["test_cutoff"].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.vlines(x - 0.13, rc["recent_p10"], rc["recent_p90"], color=BLUE, linewidth=9, alpha=0.55,
              label=f"this project's range (caught actual {int(rc['recent_hit'].sum())}/10)")
    ax.vlines(x + 0.13, rc["plain_p10"], rc["plain_p90"], color=ORANGE, linewidth=9, alpha=0.75,
              label=f"plain Monte Carlo range (caught {int(rc['plain_hit'].sum())}/10)")
    ax.plot(x, rc["actual_gw"], "o", color=INK, markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="actually built", zorder=3)
    ax.set_xticks(x, [f"{c}\n({n} yr)" if n < 5 else str(c) for c, n in zip(x, rc["years"])])
    ax.set_ylim(0, max(rc["recent_p90"].max(), rc["plain_p90"].max()) * 1.1)
    ax.set_xlabel("Forecast made at end of year (using only data known then); scored on the next 5 years",
                  color=INK_2, fontsize=9)
    ax.set_ylabel("GW of generation built", color=INK_2, fontsize=9)
    ax.set_title("Track record: forecast ranges vs what actually got built", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style(ax)
    fig.tight_layout()
    _save(fig, "track_record.png")


def main() -> None:
    df = load_clean()
    FIG_DIR.mkdir(exist_ok=True)
    plot_headline(df)
    plot_by_tech()
    plot_state_map()
    plot_drift(df)
    plot_track_record()
    print(f"Wrote 5 figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
