# Decision log

Every modelling choice, with its reason. This is the interview-prep table.

| #  | Decision | Choice | Why | Status |
|----|----------|--------|-----|--------|
| D1 | Start of the clock | Queue-entry date (`q_date`) | Same starting point for every project; makes regions comparable | Agreed (Day 1) |
| D2 | What counts as built | Status `operational`; event date = `on_date` | Operational is the only outcome that delivers power | Agreed (Day 1) |
| D3 | Suspended projects | Treat as still-waiting (censored); `suspended_flag` carried for a withdrawn-sensitivity rerun | Suspended projects can resume; showing both bounds the effect | Agreed (Day 1) |
| D4 | End of observation | 2025-12-31 (data snapshot); waiting projects right-censored here | Last date we actually know outcomes | Agreed (Day 1) |
| DM | Missing terminal dates | Keep all projects; impute event date = `q_date` + group-median observed lag (tech×region), flagged in `event_date_source`. Built dates fall back to `prop_date` (a real date) before imputing. | ~31% of built and ~36% of withdrawn lack an exact terminal date; we need the whole dataset, and the flag keeps it auditable / sensitivity-testable | Agreed (Day 1) |
| D5 | Core model | Aalen-Johansen competing-risks curves by group; cause-specific Cox for size/entry-year | Handles two competing outcomes and stays explainable | Proposed (Day 2) |
| D6 | Backtest cutoff | Fit as of end-2019, score 2020–2025 | Six years of real outcomes to check against | Proposed (Day 3) |
| D7 | Groups | Technology × region; 9 LBNL regions; tech = solar/wind/storage/gas/hybrid/other (hybrid = generation co-located with storage) | Matches how Berkeley Lab reports; avoids noisy tiny groups | Agreed (Day 1) |
| D8 | Simulation runs | 10,000 (verify by reseeding) | Enough for stable P10/P90 | Proposed (Day 3) |
| D9 | Who writes code | Claude writes; Kevin decides, reviews, explains back | Fits the 4-day deadline and the TPM story | Agreed |

## Day-1 data finding (exit check)

Project counts by region and status match LBNL's published summary table
(sheet `02. Data Sample by Region`) in **26 of 27 region×status cells, exactly**.
Operational and withdrawn match in every region; active matches in 8 of 9.

The one exception: **ERCOT active = 1,796 in the project-level file vs 1,527 in
the summary table (+269, +17.6%)**. Those 269 rows have unique IDs, nonzero
capacity, and valid interconnection-agreement phases — so the gap is a
definitional difference between the row-level file and LBNL's summary tab, not
duplicates or empty records. We treat the project-level file as authoritative
and report this discrepancy. Verify with `uv run python -m src.validate`.

Cleaning drops 674 rows with no queue-entry date (can't start the clock) and 10
`unknown`-status rows → **37,527 projects** in `data/clean/projects.csv`.
