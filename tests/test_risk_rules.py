import pytest
from risk_rules import label_risk, score_transaction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def base_tx(**overrides):
    """Minimal low-risk transaction; override individual fields per test."""
    tx = {
        "device_risk_score": 10,
        "is_international": 0,
        "amount_usd": 50,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


# ---------------------------------------------------------------------------
# label_risk thresholds
# ---------------------------------------------------------------------------

def test_label_low():
    assert label_risk(0) == "low"
    assert label_risk(29) == "low"


def test_label_medium():
    assert label_risk(30) == "medium"
    assert label_risk(59) == "medium"


def test_label_high():
    assert label_risk(60) == "high"
    assert label_risk(100) == "high"


# ---------------------------------------------------------------------------
# Device risk score
# ---------------------------------------------------------------------------

def test_high_device_risk_adds_25():
    low = score_transaction(base_tx(device_risk_score=10))
    high = score_transaction(base_tx(device_risk_score=70))
    assert high - low == 25


def test_medium_device_risk_adds_10():
    low = score_transaction(base_tx(device_risk_score=10))
    mid = score_transaction(base_tx(device_risk_score=40))
    assert mid - low == 10


def test_device_risk_below_40_adds_nothing():
    s1 = score_transaction(base_tx(device_risk_score=0))
    s2 = score_transaction(base_tx(device_risk_score=39))
    assert s1 == s2


# ---------------------------------------------------------------------------
# International transactions
# ---------------------------------------------------------------------------

def test_international_adds_15():
    domestic = score_transaction(base_tx(is_international=0))
    intl = score_transaction(base_tx(is_international=1))
    assert intl - domestic == 15


def test_domestic_adds_nothing():
    s = score_transaction(base_tx(is_international=0))
    assert s == score_transaction(base_tx())


# ---------------------------------------------------------------------------
# Amount
# ---------------------------------------------------------------------------

def test_large_amount_adds_25():
    low = score_transaction(base_tx(amount_usd=50))
    high = score_transaction(base_tx(amount_usd=1000))
    assert high - low == 25


def test_medium_amount_adds_10():
    low = score_transaction(base_tx(amount_usd=50))
    mid = score_transaction(base_tx(amount_usd=500))
    assert mid - low == 10


def test_small_amount_adds_nothing():
    s1 = score_transaction(base_tx(amount_usd=1))
    s2 = score_transaction(base_tx(amount_usd=499))
    assert s1 == s2


# ---------------------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------------------

def test_high_velocity_adds_20():
    low = score_transaction(base_tx(velocity_24h=1))
    high = score_transaction(base_tx(velocity_24h=6))
    assert high - low == 20


def test_medium_velocity_adds_5():
    low = score_transaction(base_tx(velocity_24h=1))
    mid = score_transaction(base_tx(velocity_24h=3))
    assert mid - low == 5


def test_low_velocity_adds_nothing():
    s1 = score_transaction(base_tx(velocity_24h=1))
    s2 = score_transaction(base_tx(velocity_24h=2))
    assert s1 == s2


# ---------------------------------------------------------------------------
# Failed logins
# ---------------------------------------------------------------------------

def test_high_failed_logins_adds_20():
    low = score_transaction(base_tx(failed_logins_24h=0))
    high = score_transaction(base_tx(failed_logins_24h=5))
    assert high - low == 20


def test_medium_failed_logins_adds_10():
    low = score_transaction(base_tx(failed_logins_24h=0))
    mid = score_transaction(base_tx(failed_logins_24h=2))
    assert mid - low == 10


def test_no_failed_logins_adds_nothing():
    s1 = score_transaction(base_tx(failed_logins_24h=0))
    s2 = score_transaction(base_tx(failed_logins_24h=1))
    assert s1 == s2


# ---------------------------------------------------------------------------
# Prior chargebacks
# ---------------------------------------------------------------------------

def test_multiple_prior_chargebacks_adds_20():
    clean = score_transaction(base_tx(prior_chargebacks=0))
    repeat = score_transaction(base_tx(prior_chargebacks=2))
    assert repeat - clean == 20


def test_one_prior_chargeback_adds_5():
    clean = score_transaction(base_tx(prior_chargebacks=0))
    one = score_transaction(base_tx(prior_chargebacks=1))
    assert one - clean == 5


def test_no_prior_chargebacks_adds_nothing():
    s = score_transaction(base_tx(prior_chargebacks=0))
    assert s == score_transaction(base_tx())


# ---------------------------------------------------------------------------
# Score boundaries
# ---------------------------------------------------------------------------

def test_score_never_below_zero():
    assert score_transaction(base_tx()) >= 0


def test_score_never_above_100():
    tx = base_tx(
        device_risk_score=90,
        is_international=1,
        amount_usd=2000,
        velocity_24h=10,
        failed_logins_24h=10,
        prior_chargebacks=5,
    )
    assert score_transaction(tx) <= 100


# ---------------------------------------------------------------------------
# End-to-end: known fraud transaction (tx 50003)
# ---------------------------------------------------------------------------

def test_known_fraud_transaction_scores_high():
    """
    tx 50003: gift cards, Philippines, device risk 81, 6 txns/24h,
    5 failed logins, $1250 — confirmed chargeback. Must be labeled high.
    """
    tx = base_tx(
        device_risk_score=81,
        is_international=1,
        amount_usd=1250,
        velocity_24h=6,
        failed_logins_24h=5,
        prior_chargebacks=0,
    )
    score = score_transaction(tx)
    assert score >= 60, f"Expected high-risk score, got {score}"
    assert label_risk(score) == "high"


def test_clean_transaction_scores_low():
    """Ordinary domestic grocery purchase — should not be flagged."""
    tx = base_tx(
        device_risk_score=8,
        is_international=0,
        amount_usd=45,
        velocity_24h=1,
        failed_logins_24h=0,
        prior_chargebacks=0,
    )
    score = score_transaction(tx)
    assert score < 30, f"Expected low-risk score, got {score}"
    assert label_risk(score) == "low"
