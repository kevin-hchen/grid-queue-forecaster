# Data dictionary — `data/clean/projects.csv`

One row per interconnection project (37,527 rows), produced by `src/clean.py`
from the Berkeley Lab *Queued Up: 2026 Edition* project-level file
(`03. Complete Queue Data`), snapshot end-2025.

| Column | Type | Meaning | Source / derivation |
|--------|------|---------|---------------------|
| `q_id` | str | Queue identifier (region-local, not globally unique) | file `q_id` |
| `region` | str | Grid region, 9 LBNL groups (CAISO, ERCOT, ISO-NE, MISO, NYISO, PJM, SPP, Southeast, West) | file `region` |
| `state` | str | US state | file `state` |
| `tech_group` | str | solar / wind / storage / gas / hybrid / other (D7). Hybrid = generation co-located with storage | derived from `type_clean` by `tech_group()` |
| `type_clean` | str | Original LBNL technology label (48 values) | file `type_clean` |
| `outcome` | str | Competing-risks outcome: **built** / **withdrawn** / **waiting** | `operational`→built, `withdrawn`→withdrawn, `active`+`suspended`→waiting (D2, D3) |
| `suspended_flag` | bool | True if original status was `suspended` (for the D3 sensitivity rerun) | file `q_status == 'suspended'` |
| `q_date` | date | Queue-entry date — the clock start (D1) | file `q_date` |
| `q_year` | int | Queue-entry year | file `q_year` |
| `event_date` | date | Date the project was built / withdrawn, or the censoring date for waiting projects. Clipped to `[q_date, 2025-12-31]` | see `event_date_source` |
| `event_date_source` | str | How `event_date` was resolved: `on_date`, `prop_date`, `wd_date`, `imputed`, or `censored` | `src.clean._impute_event_dates` |
| `duration_years` | float | Years from `q_date` to `event_date` (time-to-event or time-at-risk). Median 1.78, mean 2.54 | derived |
| `mw_1`,`mw_2`,`mw_3` | float | Nameplate MW of the up-to-3 co-located technologies | file `mw_1..3` |
| `mw_total` | float | Sum of `mw_1..3` (nonzero for all rows) | derived |

## `event_date_source` distribution

| source | n | meaning |
|--------|---|---------|
| `wd_date` | 15,373 | withdrawn, real withdrawal date |
| `censored` | 9,091 | waiting, censored at 2025-12-31 |
| `imputed` | 8,862 | terminal event, no exact date → `q_date` + group-median lag |
| `on_date` | 3,146 | built, real operational date |
| `prop_date` | 1,055 | built, no `on_date` → proposed online date (a real date) |

So **92%** of *built* and **64%** of *withdrawn* projects carry a real date; the
rest are imputed and flagged. Exclude `event_date_source == 'imputed'` for a
sensitivity run.

## Known caveats

- **ERCOT active +269 vs the published summary table** — see `docs/DECISIONS.md`.
- **Right-censoring** — recent projects haven't finished; survival methods
  (Stage 2) handle this, but naive built/total ratios would understate build-out.
- **Hybrid capacity** — `mw_total` sums generation and storage MW; the capacity
  outlook (Stage 4) will need a generation-only definition to avoid counting
  storage as supply.
