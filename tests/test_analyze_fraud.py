from pathlib import Path

import pandas as pd
import pytest

from analyze_fraud import score_transactions, summarize_results


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scored(rows):
    """Build a minimal scored DataFrame for summarize_results tests."""
    return pd.DataFrame(rows, columns=["transaction_id", "amount_usd", "risk_label"])


def make_chargebacks(transaction_ids):
    return pd.DataFrame({"transaction_id": transaction_ids})


# ---------------------------------------------------------------------------
# summarize_results — metric correctness
# ---------------------------------------------------------------------------

def test_chargeback_rate_is_one_when_all_flagged():
    scored = make_scored([
        (1, 100.0, "high"),
        (2, 200.0, "high"),
    ])
    cb = make_chargebacks([1, 2])
    result = summarize_results(scored, cb)
    high_row = result[result["risk_label"] == "high"].iloc[0]
    assert high_row["chargeback_rate"] == 1.0


def test_chargeback_rate_is_zero_when_none_flagged():
    scored = make_scored([
        (1, 100.0, "low"),
        (2, 200.0, "low"),
    ])
    cb = make_chargebacks([])
    result = summarize_results(scored, cb)
    low_row = result[result["risk_label"] == "low"].iloc[0]
    assert low_row["chargeback_rate"] == 0.0


def test_chargeback_rate_partial():
    scored = make_scored([
        (1, 100.0, "high"),
        (2, 200.0, "high"),
        (3, 150.0, "high"),
        (4, 50.0, "high"),
    ])
    cb = make_chargebacks([1, 2])
    result = summarize_results(scored, cb)
    high_row = result[result["risk_label"] == "high"].iloc[0]
    assert high_row["chargeback_rate"] == pytest.approx(0.5)


def test_transaction_count_per_label():
    scored = make_scored([
        (1, 10.0, "high"),
        (2, 20.0, "medium"),
        (3, 30.0, "medium"),
        (4, 40.0, "low"),
        (5, 50.0, "low"),
        (6, 60.0, "low"),
    ])
    cb = make_chargebacks([])
    result = summarize_results(scored, cb)
    counts = result.set_index("risk_label")["transactions"]
    assert counts["high"] == 1
    assert counts["medium"] == 2
    assert counts["low"] == 3


def test_total_amount_usd_per_label():
    scored = make_scored([
        (1, 100.0, "high"),
        (2, 200.0, "high"),
        (3, 50.0, "low"),
    ])
    cb = make_chargebacks([1])
    result = summarize_results(scored, cb)
    totals = result.set_index("risk_label")["total_amount_usd"]
    assert totals["high"] == pytest.approx(300.0)
    assert totals["low"] == pytest.approx(50.0)


def test_avg_amount_usd_per_label():
    scored = make_scored([
        (1, 100.0, "high"),
        (2, 300.0, "high"),
        (3, 60.0, "low"),
    ])
    cb = make_chargebacks([])
    result = summarize_results(scored, cb)
    avgs = result.set_index("risk_label")["avg_amount_usd"]
    assert avgs["high"] == pytest.approx(200.0)
    assert avgs["low"] == pytest.approx(60.0)


def test_chargeback_count_per_label():
    scored = make_scored([
        (1, 100.0, "high"),
        (2, 200.0, "high"),
        (3, 50.0, "low"),
    ])
    cb = make_chargebacks([1, 2])
    result = summarize_results(scored, cb)
    cb_counts = result.set_index("risk_label")["chargebacks"]
    assert cb_counts["high"] == 2
    assert cb_counts["low"] == 0


def test_chargeback_not_counted_twice_if_label_repeated():
    # Each transaction_id should only be counted once as a chargeback
    scored = make_scored([
        (1, 100.0, "high"),
        (2, 200.0, "high"),
    ])
    cb = make_chargebacks([1])
    result = summarize_results(scored, cb)
    high_row = result[result["risk_label"] == "high"].iloc[0]
    assert high_row["chargebacks"] == 1


def test_labels_present_in_output():
    scored = make_scored([
        (1, 100.0, "high"),
        (2, 50.0, "medium"),
        (3, 20.0, "low"),
    ])
    cb = make_chargebacks([1])
    result = summarize_results(scored, cb)
    assert set(result["risk_label"]) == {"high", "medium", "low"}


# ---------------------------------------------------------------------------
# End-to-end integration against actual CSV data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline_results():
    accounts = pd.read_csv(DATA_DIR / "accounts.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    chargebacks = pd.read_csv(DATA_DIR / "chargebacks.csv")
    scored = score_transactions(transactions, accounts)
    summary = summarize_results(scored, chargebacks)
    return scored, summary


def test_no_confirmed_chargeback_is_labeled_low(pipeline_results):
    scored, _ = pipeline_results
    chargebacks = pd.read_csv(DATA_DIR / "chargebacks.csv")
    fraud_ids = set(chargebacks["transaction_id"])
    fraud_rows = scored[scored["transaction_id"].isin(fraud_ids)]
    low_fraud = fraud_rows[fraud_rows["risk_label"] == "low"]
    assert len(low_fraud) == 0, (
        f"Confirmed chargebacks labeled 'low': {low_fraud['transaction_id'].tolist()}"
    )


def test_high_risk_bucket_chargeback_rate_is_100_percent(pipeline_results):
    _, summary = pipeline_results
    high_row = summary[summary["risk_label"] == "high"]
    assert len(high_row) == 1, "Expected a 'high' risk label row in summary"
    assert high_row["chargeback_rate"].iloc[0] == pytest.approx(1.0), (
        "Every transaction in the 'high' bucket should be a confirmed chargeback"
    )


def test_low_risk_bucket_chargeback_rate_is_zero(pipeline_results):
    _, summary = pipeline_results
    low_row = summary[summary["risk_label"] == "low"]
    assert len(low_row) == 1, "Expected a 'low' risk label row in summary"
    assert low_row["chargeback_rate"].iloc[0] == pytest.approx(0.0), (
        "No confirmed chargebacks should appear in the 'low' bucket"
    )


def test_total_confirmed_fraud_loss_is_correct(pipeline_results):
    scored, summary = pipeline_results
    chargebacks = pd.read_csv(DATA_DIR / "chargebacks.csv")
    expected_total_loss = chargebacks["loss_amount_usd"].sum()
    fraud_ids = set(chargebacks["transaction_id"])
    scored_fraud = scored[scored["transaction_id"].isin(fraud_ids)]
    actual_total = scored_fraud["amount_usd"].sum()
    assert actual_total == pytest.approx(expected_total_loss, rel=1e-3)


def test_all_transactions_are_scored(pipeline_results):
    scored, _ = pipeline_results
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    assert len(scored) == len(transactions)


def test_risk_scores_are_within_bounds(pipeline_results):
    scored, _ = pipeline_results
    assert (scored["risk_score"] >= 0).all()
    assert (scored["risk_score"] <= 100).all()


def test_risk_labels_are_valid_values(pipeline_results):
    scored, _ = pipeline_results
    assert set(scored["risk_label"]).issubset({"low", "medium", "high"})
