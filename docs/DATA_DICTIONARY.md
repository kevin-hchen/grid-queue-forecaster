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
| `ia_date` | date | Interconnection agreement date, where recorded (sparse and uneven by region; not used by the model) | file `ia_date` |
| `event_date` | date | Date the project was built / withdrawn, or the censoring date for waiting projects. Clipped to `[q_date, 2025-12-31]` | see `event_date_source` |
| `event_date_source` | str | How `event_date` was resolved: `on_date`, `prop_date`, `wd_date`, `imputed`, or `censored` | `src.clean._impute_event_dates` |
| `duration_years` | float | Years from `q_date` to `event_date` (time-to-event or time-at-risk). Median 1.78, mean 2.54 | derived |
| `mw_1`,`mw_2`,`mw_3` | float | Nameplate MW of the up-to-3 co-located technologies | file `mw_1..3` |
| `mw_total` | float | Sum of `mw_1..3`. 16 rows are zero, negative or missing in the source; Stage 4 treats negatives as 0 | derived |
| `mw_gen` | float | Generation MW: `mw_total` minus storage. Used for the supply outlook | derived from `type_1..3` |
| `mw_storage` | float | Storage MW (Battery, Other Storage) among the up-to-3 technologies. Reported separately so batteries aren't counted as supply | derived from `type_1..3` |

## `event_date_source` distribution

| source | n | meaning |
|--------|---|---------|
| `wd_date` | 15,373 | withdrawn, real withdrawal date |
| `censored` | 9,091 | waiting, censored at 2025-12-31 |
| `imputed` | 8,862 | terminal event, no exact date → `q_date` + group-median lag |
| `on_date` | 3,146 | built, real operational date |
| `prop_date` | 1,055 | built, no `on_date` → planned online date, used as a stand-in |

So **92%** of *built* projects carry an operational or planned date and **64%**
of *withdrawn* projects carry a real withdrawal date; the rest are imputed and
flagged. Dropping imputed rows would mostly drop withdrawals and bias the build
share upward, so the sensitivity run (`src.survival`, scenario
`imputed_resampled`) instead redraws their durations from real ones.

## Known caveats

- **ERCOT active +269 vs the published summary table** — see `docs/DECISIONS.md`.
- **Right-censoring** — recent projects haven't finished; survival methods
  (Stage 2) handle this, but naive built/total ratios would understate build-out.
- **Hybrid capacity** — `mw_total` sums generation and storage MW; the outlook
  uses `mw_gen` and reports `mw_storage` separately.
