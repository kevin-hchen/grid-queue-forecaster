# Grid Queue Forecaster

Of the US power projects queued today, how much will realistically get built,
where, and when — with calibrated uncertainty.

**Status:** Day 1 complete — cleaned dataset + data dictionary, counts validated
against the Berkeley Lab report. Model, backtest, and outlook follow (Days 2–4).

## Data
Berkeley Lab, *Queued Up: 2026 Edition* (project-level file, snapshot end-2025).
Not committed. Download to `data/raw/`:
~~~
curl -L -A "Mozilla/5.0" \
  "https://eta-publications.lbl.gov/sites/default/files/2026-05/lbnl_ix_queue_data_file_thru2025.xlsx" \
  -o data/raw/lbnl_ix_queue_data_file_thru2025.xlsx
~~~

## Run
~~~
uv sync
uv run python -m src.clean      # Stage 1: clean + define outcomes -> data/clean/projects.csv
uv run python -m src.validate   # exit check: counts vs the LBNL report
uv run pytest                   # tests
~~~

## Pipeline
1. **Clean** (`src/clean.py`) — map every project to built / withdrawn / waiting with dates. ✅
2. **Survival** — Aalen-Johansen competing-risks curves by tech × region. _(Day 2)_
3. **Backtest** — fit as of 2019, score 2020–2025; calibration + Brier. _(Day 3)_
4. **Outlook** — Monte Carlo P10/P50/P90 new MW by region and year, 2026–2030. _(Day 3)_

See `docs/DECISIONS.md` for every modelling choice and `docs/DATA_DICTIONARY.md`
for the cleaned schema.
