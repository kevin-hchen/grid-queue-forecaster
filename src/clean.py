"""Stage 1 — Clean the Berkeley Lab queue data and define competing-risks outcomes.

One job: map every project to an outcome (built / withdrawn / waiting), give it a
start date and an event/censoring date, and attach a region and technology group.

Decisions implemented (see docs/DECISIONS.md):
  D1  Clock starts at queue-entry date (q_date).
  D2  "Built" = status operational; event date is the operational date (on_date).
  D3  Suspended projects are treated as still-waiting (censored); a suspended_flag
      is carried so a sensitivity run can re-label them as withdrawn.
  D4  Observation ends 2025-12-31 (the data snapshot); waiting projects are
      right-censored there.
  Missing dates: ~31% of built and ~36% of withdrawn lack an exact terminal date.
      We keep them (we need the whole dataset) and impute the event date as
      q_date + the group-median observed lag for that outcome, flagged in
      event_date_source so the imputation is auditable and can be excluded in a
      sensitivity run. Build dates fall back to prop_date (the planned online
      date) when on_date is missing: a reported date used as a stand-in for the
      actual one, flagged separately from imputed dates.
  D7  Technology groups: solar, wind, storage, gas, hybrid (generation co-located
      with storage), other. Region is taken from the file's `region` column.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/lbnl_ix_queue_data_file_thru2025.xlsx")
RAW_SHEET = "03. Complete Queue Data"
CLEAN_PATH = Path("data/clean/projects.csv")

# D4: the data snapshot date. Waiting projects are censored here.
OBS_END = pd.Timestamp("2025-12-31")

DATE_COLS = ["q_date", "prop_date", "on_date", "wd_date", "ia_date"]

# Map the file's q_status to our three competing-risks outcomes.
# 'suspended' -> waiting by default (D3); 'unknown' is dropped.
OUTCOME_MAP = {
    "operational": "built",
    "withdrawn": "withdrawn",
    "active": "waiting",
    "suspended": "waiting",
}

STORAGE_TOKENS = {"Battery", "Other Storage"}


def tech_group(type_clean: str) -> str:
    """Collapse the file's 48 technology labels into 6 groups (D7).

    Rule, in order:
      - generation co-located with storage  -> hybrid   (e.g. Solar+Battery)
      - storage only                        -> storage  (Battery, Battery+Other Storage)
      - a single generation technology      -> solar / wind / gas / other
      - mixed generation, no storage (rare) -> other    (e.g. Gas+Oil, Solar+Wind)
    """
    tokens = [t.strip() for t in str(type_clean).split("+")]
    has_storage = any(t in STORAGE_TOKENS for t in tokens)
    gens = [t for t in tokens if t not in STORAGE_TOKENS]

    if has_storage and gens:
        return "hybrid"
    if has_storage and not gens:
        return "storage"
    if len(gens) == 1:
        t = gens[0]
        if t == "Solar":
            return "solar"
        if t in ("Wind", "Offshore Wind"):
            return "wind"
        if t == "Gas":
            return "gas"
        return "other"
    return "other"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the project-level sheet (its header is on row 2, under a banner row)."""
    df = pd.read_excel(path, sheet_name=RAW_SHEET, header=1).dropna(how="all")
    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _group_median_lag_years(df: pd.DataFrame, outcome: str, event_col: str) -> tuple[pd.Series, float]:
    """Median observed lag (event_date - q_date) in years, per tech_group x region.

    Computed only from rows of this outcome that have BOTH a q_date and a real
    event date. Used to impute the event date for rows missing it.
    """
    obs = df[(df["outcome"] == outcome) & df["q_date"].notna() & df[event_col].notna()].copy()
    obs["lag"] = (obs[event_col] - obs["q_date"]).dt.days / 365.25
    obs = obs[obs["lag"] >= 0]
    by_group = obs.groupby(["tech_group", "region"])["lag"].median()
    global_median = obs["lag"].median()
    return by_group, global_median


def _impute_event_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Fill event_date and event_date_source for built and withdrawn projects.

    built:     on_date -> prop_date -> q_date + group-median build lag
    withdrawn: wd_date  ->            q_date + group-median withdrawal lag
    waiting:   no event; censored at OBS_END.
    """
    df = df.copy()
    df["event_date"] = pd.NaT
    df["event_date_source"] = pd.NA

    # --- built ---
    built = df["outcome"] == "built"
    df.loc[built & df["on_date"].notna(), "event_date"] = df.loc[built & df["on_date"].notna(), "on_date"]
    df.loc[built & df["on_date"].notna(), "event_date_source"] = "on_date"
    # fall back to the planned online date (a stand-in, flagged as prop_date)
    need = built & df["event_date"].isna() & df["prop_date"].notna()
    df.loc[need, "event_date"] = df.loc[need, "prop_date"]
    df.loc[need, "event_date_source"] = "prop_date"

    # --- withdrawn ---
    wd = df["outcome"] == "withdrawn"
    df.loc[wd & df["wd_date"].notna(), "event_date"] = df.loc[wd & df["wd_date"].notna(), "wd_date"]
    df.loc[wd & df["wd_date"].notna(), "event_date_source"] = "wd_date"

    # --- impute the remainder from group-median lags ---
    build_lags, build_global = _group_median_lag_years(df, "built", "on_date")
    wd_lags, wd_global = _group_median_lag_years(df, "withdrawn", "wd_date")

    def impute(row, lags, global_lag):
        lag = lags.get((row["tech_group"], row["region"]), global_lag)
        if pd.isna(lag):
            lag = global_lag
        return row["q_date"] + pd.to_timedelta(lag * 365.25, unit="D")

    for outcome, lags, global_lag in [("built", build_lags, build_global),
                                      ("withdrawn", wd_lags, wd_global)]:
        mask = (df["outcome"] == outcome) & df["event_date"].isna() & df["q_date"].notna()
        df.loc[mask, "event_date"] = df.loc[mask].apply(lambda r: impute(r, lags, global_lag), axis=1)
        df.loc[mask, "event_date_source"] = "imputed"

    # waiting projects are censored at the snapshot date
    waiting = df["outcome"] == "waiting"
    df.loc[waiting, "event_date"] = OBS_END
    df.loc[waiting, "event_date_source"] = "censored"

    return df


def clean(path: Path = RAW_PATH) -> pd.DataFrame:
    """Full Stage-1 pipeline: raw sheet -> tidy project-level analysis table."""
    df = load_raw(path)

    # Outcome (drop the 10 'unknown' rows and anything with no queue-entry date).
    df["outcome"] = df["q_status"].map(OUTCOME_MAP)
    df["suspended_flag"] = df["q_status"].eq("suspended")
    df = df[df["outcome"].notna() & df["q_date"].notna()].copy()

    df["tech_group"] = df["type_clean"].map(tech_group)

    df = _impute_event_dates(df)

    # Clip event dates into [q_date, OBS_END] and compute non-negative duration.
    df["event_date"] = df["event_date"].clip(upper=OBS_END)
    df["event_date"] = df[["event_date", "q_date"]].max(axis=1)
    df["duration_years"] = (df["event_date"] - df["q_date"]).dt.days / 365.25
    df["duration_years"] = df["duration_years"].clip(lower=0)

    # Total nameplate across the (up to 3) co-located technologies, split into
    # generation and storage so storage isn't counted as new supply in Stage 4.
    df["mw_total"] = df[["mw_1", "mw_2", "mw_3"]].sum(axis=1, min_count=1)
    df["mw_storage"] = sum(
        df[f"mw_{i}"].where(df[f"type_{i}"].isin(STORAGE_TOKENS), 0).fillna(0) for i in (1, 2, 3)
    )
    df["mw_gen"] = df["mw_total"].fillna(0) - df["mw_storage"]

    out = df[[
        "q_id", "region", "state", "tech_group", "type_clean",
        "outcome", "suspended_flag",
        "q_date", "q_year", "ia_date", "event_date", "event_date_source", "duration_years",
        "mw_1", "mw_2", "mw_3", "mw_total", "mw_gen", "mw_storage",
    ]].reset_index(drop=True)
    return out


def main() -> None:
    df = clean()
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Wrote {len(df):,} projects -> {CLEAN_PATH}")
    print("\nOutcome counts:")
    print(df["outcome"].value_counts().to_string())
    print("\nEvent-date source (how each terminal date was resolved):")
    print(df["event_date_source"].value_counts().to_string())


if __name__ == "__main__":
    main()
