"""Day-1 exit check — do our project counts match the Berkeley Lab report?

Compares project-level counts by region and status (from sheet '03. Complete
Queue Data') against LBNL's own published summary table (sheet '02. Data Sample
by Region'). The proposed target is agreement within 2% per region/status.

Finding (end-2025 data): operational and withdrawn counts match the published
table exactly in every region; active matches exactly in 8 of 9 regions. ERCOT
active is 1,796 in the project-level file vs 1,527 in the summary table (+269).
Those 269 rows are unique project IDs with capacity and valid IA phases, so the
gap is a definitional difference between the row-level file and LBNL's summary
tab, not duplicates or empty records. We treat the project-level file as
authoritative and surface the discrepancy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/lbnl_ix_queue_data_file_thru2025.xlsx")
TOLERANCE = 0.02  # proposed: within 2%

# official summary label -> project-level file label
REGION_ALIAS = {"Southeast (non-ISO)": "Southeast", "West (non-ISO)": "West"}
# official summary column -> project-level q_status
STATUS_ALIAS = {"Active": "active", "Operational": "operational", "Withdrawn": "withdrawn"}


def official_counts(path: Path = RAW_PATH) -> pd.DataFrame:
    """Published counts per region x status from sheet '02. Data Sample by Region'."""
    df = pd.read_excel(path, sheet_name="02. Data Sample by Region", header=3, nrows=9)
    df = df.rename(columns={df.columns[0]: "region"})
    df["region"] = df["region"].replace(REGION_ALIAS)
    # the three 'n' columns are the count columns (capacity columns sit between them)
    out = pd.DataFrame({"region": df["region"]})
    out["active"] = df["n"]
    out["operational"] = df["n.1"]
    out["withdrawn"] = df["n.2"]
    return out.set_index("region").sort_index()


def our_counts(path: Path = RAW_PATH) -> pd.DataFrame:
    """Our project-level counts per region x status."""
    raw = pd.read_excel(path, sheet_name="03. Complete Queue Data", header=1).dropna(how="all")
    piv = raw.pivot_table(index="region", columns="q_status", values="q_id", aggfunc="count", fill_value=0)
    return piv[["active", "operational", "withdrawn"]].sort_index()


def compare(path: Path = RAW_PATH) -> pd.DataFrame:
    """Per region/status: ours, official, and % difference."""
    ours, off = our_counts(path), official_counts(path)
    rows = []
    for region in off.index:
        for status in ["active", "operational", "withdrawn"]:
            o = int(ours.loc[region, status])
            r = int(off.loc[region, status])
            pct = (o - r) / r if r else float("nan")
            rows.append({"region": region, "status": status, "ours": o,
                         "official": r, "pct_diff": pct, "pass": abs(pct) <= TOLERANCE})
    return pd.DataFrame(rows)


def main() -> None:
    cmp = compare()
    show = cmp.copy()
    show["pct_diff"] = (show["pct_diff"] * 100).round(1).astype(str) + "%"
    print(show.to_string(index=False))
    n_pass = int(cmp["pass"].sum())
    print(f"\n{n_pass}/{len(cmp)} region-status cells within {TOLERANCE:.0%}.")
    fails = cmp[~cmp["pass"]]
    if len(fails):
        print("Outside tolerance:")
        for _, r in fails.iterrows():
            print(f"  {r.region} {r.status}: ours={r.ours} official={r.official} ({r.pct_diff*100:+.1f}%)")


if __name__ == "__main__":
    main()
