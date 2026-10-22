# Decision log

Every modelling choice, with its reason. This is the interview-prep table.

| #  | Decision | Choice | Why | Status |
|----|----------|--------|-----|--------|
| D1 | Start of the clock | Queue-entry date (`q_date`) | Same starting point for every project; makes regions comparable | Agreed (Day 1) |
| D2 | What counts as built | Status `operational`; event date = `on_date` | Operational is the only outcome that delivers power | Agreed (Day 1) |
| D3 | Suspended projects | Treat as still-waiting (censored); `suspended_flag` carried for a withdrawn-sensitivity rerun | Suspended projects can resume; showing both bounds the effect | Agreed (Day 1). Sensitivity: 10-year build shares move ≤1.6 pts; 2030 outlook P50 153 → 141 GW |
| D4 | End of observation | 2025-12-31 (data snapshot); waiting projects right-censored here | Last date we actually know outcomes | Agreed (Day 1) |
| DM | Missing terminal dates | Keep all projects; impute event date = `q_date` + group-median observed lag (tech×region), flagged in `event_date_source`. Built dates fall back to `prop_date` (the planned online date, a stand-in) before imputing. | ~31% of built and ~36% of withdrawn lack an exact terminal date; we need the whole dataset, and the flag keeps it auditable / sensitivity-testable | Agreed (Day 1). Sensitivity: redrawing imputed durations from real ones moves 10-year build shares ≤0.5 pts |
| D5 | Core model | Aalen-Johansen competing-risks curves by group (own numpy implementation, matches `lifelines` to 1e-15) | Handles two competing outcomes and stays explainable. The planned Cox model was replaced by size bands (D11) + recalibration (D13), which the backtest could test directly | Agreed (Day 3) |
| D6 | Backtest cutoff | Fit as of end-2019, score 2020–2025 | Six years of real outcomes to check against | Agreed (Day 3) — see results below |
| D7 | Groups | Technology × region; 9 LBNL regions; tech = solar/wind/storage/gas/hybrid/other (hybrid = generation co-located with storage). Groups under 100 projects use the technology's all-region curve | Matches how Berkeley Lab reports; avoids noisy tiny groups. Borrowing the national curve keeps the technology signal instead of lumping into "other" | Agreed (Day 1); merge rule built (Day 2): 4 of 54 groups fall back |
| D8 | Simulation runs | 10,000, seed 0; verified by rerunning with seed 1 | Enough for stable P10/P90 | Agreed (Day 3): reseed moves P10/P50/P90 ≤1.3% |
| D9 | Who writes code | Claude writes; Kevin decides, reviews, explains back | Fits the 4-day deadline and the TPM story | Agreed |
| D10 | Drift fix: recent-window fit | Tested estimating curves only from the last 3/5/8 years of experience (left truncation). **Not adopted** | Chosen on a 2015 validation cutoff to keep the 2019 test clean; no window improved it | Rejected (Day 3) |
| D11 | Size bands | Split groups into <100 / 100–400 / >400 MW (generation, or storage for storage-only); a band under 100 projects falls back to tech×region, then tech | Projects over 400 MW were built ~half as often as the model predicted, at every cutoff | Agreed (Day 3) |
| D12 | Width of the outlook ranges | Each simulation run scales all odds by one shared multiplier drawn from a lognormal on the backtest's actual/expected GW ratios at every year-end cutoff 2008–2023 (16). Centre = the last 3 cutoffs (×0.55); spread = all 16 | Projects in one queue rise and fall together (policy, prices, supply chains). Independent coin flips alone give ranges far narrower than the model's real track record. The model has grown more optimistic since 2015, so the recent bias is the better centre | Agreed (Day 3). Sensitivity: centring on all 16 cutoffs (×0.79) gives P50 221 GW. Range check (D14): recent centring caught 6/10, all-history 5/10 |
| D13 | Recalibration | Platt scaling on 5-year build odds: logit(p′) = a + b·logit(p), learned from historical cutoffs (backtest: 2008–13 only; outlook: 2008–20). b ≈ 0.2–0.26 | Raw odds were overconfident (71% predictions came true 35% of the time). b < 1 pulls odds toward the average | Agreed (Day 3) |
| D14 | How the backtest is judged | (1) Project level: beat the naive constant-rate baseline on Brier score; report per-bin calibration as a limitation. (2) Totals: at each test date 2014–2023, rebuild the whole method from what was known then and check the P10–P90 range catches actual GW about 8 times in 10 | The outlook reports totals with ranges, so that's what must be tested. Restated **after** seeing the original per-bin target fail — said openly here and in the README | Agreed (Day 3). Result: (1) passes, +2.8%; (2) 6/10 — short of 8/10. Ranges left as-is rather than widened 1.1× to hit 8/10, since that would be tuned on the same tests |

## Day-1 data finding (exit check)

Project counts by region and status match LBNL's published summary table
(sheet `02. Data Sample by Region`) in **26 of 27 region×status cells, exactly**.
Operational and withdrawn match in every region; active matches in 8 of 9.

The one exception: **ERCOT active = 1,796 in the project-level file vs 1,527 in
the summary table (+269, +17.6%)**. Those 269 rows have unique IDs, nonzero
capacity, and valid interconnection-agreement phases — so the gap is most likely
a definitional difference between the row-level file and LBNL's summary tab, not
duplicates or empty records (the exact cause is unconfirmed). We treat the
project-level file as authoritative and report this discrepancy. **Signed off
(Day 3):** the Day-1 data target is accepted with this one documented exception. Verify with
`uv run python -m src.validate`.

Cleaning drops 674 of 38,201 rows — 664 with no queue-entry date and 10 with
`unknown` status → **37,527 projects** in `data/clean/projects.csv`.

## Day-2 findings (survival curves)

1. **About 18% of queued projects are built within 10 years; 75% withdraw.** A naive
   built ÷ total ratio says 12%, because it treats every still-waiting project as a failure.
2. **What dominates the queue is least likely to be built.** Within 10 years: solar
   15%, storage 11%, hybrid 9% — vs gas 32% and other 35%.
3. **ERCOT builds at 2–3× other regions' rates** (e.g. solar 27% vs 5% in MISO).

## Day-3 gate: backtest result

The plan's target — every calibration bin within 5 points, and beating a naive
baseline — **was not fully met**:

| Check (fit through 2019, scored 2020–2025) | Raw model | After D11 + D13 |
|---|---|---|
| Bins within 5 points | 3 / 10 | 3 / 10 |
| Largest bin gap | 36 pts | 12 pts |
| Brier skill vs naive constant rate | −7% | **+2.8%** |
| Expected vs actual GW, 2020–2025 | 180 vs 135 | 151 vs 135 |

Why: build rates drift. Projects queued in 2000–04 were built within 5 years
22% of the time; those queued in 2018–20, 8%. Tech, region, size and time
waited carry only weak project-level signal (AUC ≈ 0.66). The outlook reports
totals, so the criterion was restated (D14) and the full method checked out of
sample.

## Day-3 range check (D14)

At each test date 2014–2023 the method was rebuilt from what was known then,
and compared with a **plain Monte Carlo** (every project gets the same average
build rate, independent coin flips). Full table: `outputs/range_check.csv`.

| Method | Range caught actual | Median miss on P50 | Range width (P10–P90 ÷ P50) |
|---|---|---|---|
| This project | **6 / 10** | 40% | 83% |
| Centred on all history | 5 / 10 | 46% | 83% |
| Plain Monte Carlo | 2 / 10 | 30% | 13% |

- **Ranges:** this method's are honest far more often (6/10 vs 2/10), but still
  short of the 8/10 a P10–P90 range should hit. Making them 1.1× wider would
  reach 8/10 on these tests — not applied, because it would be tuned on the
  same tests.
- **Central number:** over all 10 tests the plain method's P50 misses less
  (30% vs 40%), because of 2014–15, when both were too low. At the three most
  recent test dates — the most like today — this project misses by 52%, 55%
  and 27%, the plain method by 58%, 95% and 94%.
- Both methods miss in the same direction in the same years: system-wide
  drift, not project-level error, dominates. Averaging them doesn't help.

## Day-3 outlook (2026–2030, GW of generation)

Of **1,415 GW** of generation waiting in queues at end-2025, **153 GW (P50)** is
likely online by 2030 — **P10–P90: 100–235 GW**, about 11% of the queue. That
averages 31 GW/yr vs 29 GW/yr actually built from queues in 2021–25. Storage:
57 GW (37–88) of 578 GW queued. Full table: `outputs/outlook.csv`.

Given the range check, read the range as "likely" rather than a strict 80%
interval: historically it caught the actual 6 times in 10.
