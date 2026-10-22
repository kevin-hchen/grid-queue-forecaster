"""Tests for Stage-1 cleaning logic and the data-counts exit check."""

import pandas as pd
import pytest

from src.clean import OBS_END, RAW_PATH, clean, tech_group
from src.validate import compare

# The raw Berkeley Lab file isn't committed; on a fresh clone these tests skip.
needs_raw = pytest.mark.skipif(not RAW_PATH.exists(), reason="raw LBNL file not downloaded")


# ---- technology grouping (D7) --------------------------------------------

@pytest.mark.parametrize("label,expected", [
    ("Solar", "solar"),
    ("Wind", "wind"),
    ("Offshore Wind", "wind"),
    ("Gas", "gas"),
    ("Battery", "storage"),
    ("Other Storage", "storage"),
    ("Battery+Other Storage", "storage"),   # storage-only combo stays storage
    ("Solar+Battery", "hybrid"),            # generation + storage
    ("Wind+Battery", "hybrid"),
    ("Solar+Wind+Battery", "hybrid"),
    ("Nuclear", "other"),
    ("Hydro", "other"),
    ("Gas+Oil", "other"),                   # mixed generation, no storage
    ("Solar+Wind", "other"),
])
def test_tech_group(label, expected):
    assert tech_group(label) == expected


# ---- cleaned dataset invariants ------------------------------------------

@pytest.fixture(scope="module")
def clean_df():
    if not RAW_PATH.exists():
        pytest.skip("raw LBNL file not downloaded")
    return clean()


def test_outcomes_are_the_three_categories(clean_df):
    assert set(clean_df["outcome"].unique()) == {"built", "withdrawn", "waiting"}


def test_duration_is_non_negative(clean_df):
    assert (clean_df["duration_years"] >= 0).all()


def test_event_date_within_observation_window(clean_df):
    assert (pd.to_datetime(clean_df["event_date"]) <= OBS_END).all()
    assert (pd.to_datetime(clean_df["event_date"]) >= pd.to_datetime(clean_df["q_date"])).all()


def test_waiting_projects_are_censored_at_snapshot(clean_df):
    waiting = clean_df[clean_df["outcome"] == "waiting"]
    assert (waiting["event_date_source"] == "censored").all()
    assert (pd.to_datetime(waiting["event_date"]) == OBS_END).all()


def test_terminal_events_have_a_date_source(clean_df):
    terminal = clean_df[clean_df["outcome"].isin(["built", "withdrawn"])]
    assert terminal["event_date_source"].isin({"on_date", "prop_date", "wd_date", "imputed"}).all()


def test_no_missing_queue_entry_dates(clean_df):
    assert clean_df["q_date"].notna().all()


# ---- exit check: counts match the Berkeley Lab report --------------------

@needs_raw
def test_counts_match_report_except_known_ercot_gap():
    cmp = compare()
    fails = cmp[~cmp["pass"]]
    # Exactly one documented discrepancy: ERCOT active (definitional, not a bug).
    assert list(zip(fails["region"], fails["status"])) == [("ERCOT", "active")]


@needs_raw
def test_operational_and_withdrawn_match_report_everywhere():
    cmp = compare()
    subset = cmp[cmp["status"].isin(["operational", "withdrawn"])]
    assert subset["pass"].all()
