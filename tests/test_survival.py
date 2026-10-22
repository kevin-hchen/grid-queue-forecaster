"""Tests for the Stage-2 curves and Stage-3 backtest logic (synthetic data, no raw file needed)."""

import numpy as np
import pandas as pd
import pytest
from lifelines import AalenJohansenFitter

from src.backtest import conditional_build_prob, fit_recalibration, recalibrate, rewind
from src.simulate import simulate
from src.survival import ALL_REGIONS, aalen_johansen, fit_curves, lookup, size_band


# ---- Aalen-Johansen ----------------------------------------------------------

def test_withdrawal_is_competing_not_censored():
    # 80 withdraw at year 2, 20 are built at year 5: the true build share is 20%.
    # Treating withdrawal as censoring would report 100%.
    durations = [2.0] * 80 + [5.0] * 20
    events = [2] * 80 + [1] * 20
    s, fb, fw = aalen_johansen(durations, events).at(10)
    assert fb == pytest.approx(0.20)
    assert fw == pytest.approx(0.80)
    assert s == pytest.approx(0.0)


def test_shares_sum_to_one():
    rng = np.random.default_rng(0)
    c = aalen_johansen(rng.exponential(3, 500), rng.choice([0, 1, 2], 500))
    s, fb, fw = c.at(np.linspace(0, 20, 50))
    assert np.allclose(s + fb + fw, 1.0)


def test_matches_lifelines():
    rng = np.random.default_rng(1)
    t, e = rng.exponential(3, 400), rng.choice([0, 1, 2], 400, p=[0.3, 0.2, 0.5])
    ref = AalenJohansenFitter(calculate_variance=False).fit(t, e, event_of_interest=1).cumulative_density_
    ours = aalen_johansen(t, e).at(ref.index.values)[1]
    assert np.allclose(ours, ref.iloc[:, 0].values)


def test_left_truncation_ignores_time_before_entry():
    # Two projects: one built at year 1 (observed from 0), one built at year 4 but
    # only observed from year 3. Before year 3 only the first is at risk, so the
    # year-1 build is 100% of the risk set at that time.
    c = aalen_johansen([1.0, 4.0], [1, 1], entry=[0.0, 3.0])
    assert c.at(1.5)[1] == pytest.approx(1.0)


# ---- groups and the backtest ---------------------------------------------------

def _toy(n_big=150, n_small=20):
    rows = []
    for i in range(n_big):
        rows.append(("solar", "PJM", "built" if i % 3 == 0 else "withdrawn", 1 + i % 5))
    for i in range(n_small):
        rows.append(("solar", "SPP", "waiting", 2.0))
    df = pd.DataFrame(rows, columns=["tech_group", "region", "outcome", "duration_years"])
    df["q_date"] = pd.Timestamp("2015-01-01")
    df["event_date"] = df["q_date"] + pd.to_timedelta(df["duration_years"] * 365.25, unit="D")
    return df


def test_small_groups_fall_back_to_national_curve():
    curves = fit_curves(_toy())
    assert ("solar", "PJM") in curves
    assert ("solar", "SPP") not in curves  # 20 projects < 100
    assert lookup(curves, "solar", "SPP") is curves[("solar", ALL_REGIONS)]


def test_rewind_hides_the_future():
    df = _toy()
    cutoff = pd.Timestamp("2017-01-01")
    past = rewind(df, cutoff)
    assert (past.loc[past["outcome"] != "waiting", "event_date"] <= cutoff).all()
    assert (past.loc[past["outcome"] == "waiting", "event_date"] == cutoff).all()


def test_conditional_probability_uses_time_already_waited():
    # Everyone left at year 2 has the same fate: built at year 5. So a project
    # that has waited 3 years is built within 3 more years with certainty.
    df = _toy(n_small=0)
    df["duration_years"] = [2.0] * 80 + [5.0] * 70
    df["outcome"] = ["withdrawn"] * 80 + ["built"] * 70
    c = lookup(fit_curves(df), "solar", "PJM")
    assert conditional_build_prob(c, [3.0], 3.0)[0] == pytest.approx(1.0)
    assert conditional_build_prob(c, [0.0], 6.0)[0] == pytest.approx(70 / 150)


def test_size_bands_and_fallback():
    df = _toy()
    df["mw_gen"] = [50.0] * 120 + [500.0] * 30 + [50.0] * 20
    df["mw_storage"] = 0.0
    assert list(size_band(df.iloc[[0, 130]])) == ["<100MW", ">400MW"]
    curves = fit_curves(df, by_size=True)
    assert ("solar", "PJM", "<100MW") in curves          # 120 projects
    assert ("solar", "PJM", ">400MW") not in curves      # 30 projects -> falls back
    assert lookup(curves, "solar", "PJM", ">400MW") is curves[("solar", "PJM")]


# ---- recalibration and simulation -------------------------------------------

def test_recalibration_pulls_overconfident_odds_toward_the_average():
    # Predictions of 10% and 90% that both actually come true 50% of the time.
    p = np.array([0.1] * 1000 + [0.9] * 1000)
    y = np.array([0, 1] * 1000)
    a, b = fit_recalibration(p, y)
    assert abs(b) < 0.1
    assert recalibrate([0.1, 0.9], (a, b)) == pytest.approx([0.5, 0.5], abs=0.02)


def test_simulation_matches_expected_value_without_aggregate_error():
    # 1,000 identical projects, 100 MW each, 20% chance per year for 2 years.
    odds = np.full((1000, 2), 0.2)
    mw = np.full(1000, 100.0)
    sims = simulate(odds, mw, np.zeros(1000, dtype=int), 1, mu=0.0, sd=0.0, n_runs=500, seed=0)
    assert sims[:, 0, :].mean(axis=0) == pytest.approx([20_000, 20_000], rel=0.02)


def test_simulation_never_builds_a_project_twice():
    odds = np.full((50, 3), 0.6)  # deliberately sums past 1; must be capped
    mw = np.ones(50)
    sims = simulate(odds, mw, np.zeros(50, dtype=int), 1, mu=0.0, sd=0.5, n_runs=200, seed=0)
    assert (sims.sum(axis=2) <= 50).all()
