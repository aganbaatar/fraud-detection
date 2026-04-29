import pandas as pd
import pytest
from features import build_model_frame


def make_transactions(**overrides):
    row = {
        "transaction_id": 1,
        "account_id": 100,
        "amount_usd": 50.0,
        "failed_logins_24h": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def make_accounts(**overrides):
    row = {
        "account_id": 100,
        "prior_chargebacks": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# prior_chargebacks comes from accounts, not transactions
# ---------------------------------------------------------------------------

def test_prior_chargebacks_sourced_from_accounts():
    txns = make_transactions()
    accts = make_accounts(prior_chargebacks=3)
    df = build_model_frame(txns, accts)
    assert df["prior_chargebacks"].iloc[0] == 3


def test_prior_chargebacks_not_from_transactions():
    # Transactions CSV has no prior_chargebacks column — confirmed it comes from the join
    txns = make_transactions()
    assert "prior_chargebacks" not in txns.columns
    accts = make_accounts(prior_chargebacks=0)
    df = build_model_frame(txns, accts)
    assert "prior_chargebacks" in df.columns


# ---------------------------------------------------------------------------
# is_large_amount derivation
# ---------------------------------------------------------------------------

def test_is_large_amount_true_at_threshold():
    df = build_model_frame(make_transactions(amount_usd=1000), make_accounts())
    assert df["is_large_amount"].iloc[0] == 1


def test_is_large_amount_true_above_threshold():
    df = build_model_frame(make_transactions(amount_usd=5000), make_accounts())
    assert df["is_large_amount"].iloc[0] == 1


def test_is_large_amount_false_below_threshold():
    df = build_model_frame(make_transactions(amount_usd=999.99), make_accounts())
    assert df["is_large_amount"].iloc[0] == 0


def test_is_large_amount_false_for_small_amounts():
    df = build_model_frame(make_transactions(amount_usd=10), make_accounts())
    assert df["is_large_amount"].iloc[0] == 0


# ---------------------------------------------------------------------------
# login_pressure binning
# ---------------------------------------------------------------------------

def test_login_pressure_none_for_zero_failures():
    df = build_model_frame(make_transactions(failed_logins_24h=0), make_accounts())
    assert str(df["login_pressure"].iloc[0]) == "none"


def test_login_pressure_low_for_one_failure():
    df = build_model_frame(make_transactions(failed_logins_24h=1), make_accounts())
    assert str(df["login_pressure"].iloc[0]) == "low"


def test_login_pressure_low_for_two_failures():
    df = build_model_frame(make_transactions(failed_logins_24h=2), make_accounts())
    assert str(df["login_pressure"].iloc[0]) == "low"


def test_login_pressure_high_for_three_failures():
    df = build_model_frame(make_transactions(failed_logins_24h=3), make_accounts())
    assert str(df["login_pressure"].iloc[0]) == "high"


def test_login_pressure_high_for_many_failures():
    df = build_model_frame(make_transactions(failed_logins_24h=10), make_accounts())
    assert str(df["login_pressure"].iloc[0]) == "high"


# ---------------------------------------------------------------------------
# Join behaviour
# ---------------------------------------------------------------------------

def test_join_merges_account_fields():
    txns = make_transactions()
    accts = make_accounts(prior_chargebacks=2)
    df = build_model_frame(txns, accts)
    assert len(df) == 1
    assert df["account_id"].iloc[0] == 100


def test_unmatched_account_produces_nan_not_error():
    txns = make_transactions(account_id=999)
    accts = make_accounts(account_id=100)
    df = build_model_frame(txns, accts)
    assert len(df) == 1
    assert pd.isna(df["prior_chargebacks"].iloc[0])


def test_multiple_transactions_same_account():
    txns = pd.DataFrame([
        {"transaction_id": 1, "account_id": 100, "amount_usd": 50.0, "failed_logins_24h": 0},
        {"transaction_id": 2, "account_id": 100, "amount_usd": 200.0, "failed_logins_24h": 3},
    ])
    accts = make_accounts(prior_chargebacks=1)
    df = build_model_frame(txns, accts)
    assert len(df) == 2
    assert (df["prior_chargebacks"] == 1).all()
