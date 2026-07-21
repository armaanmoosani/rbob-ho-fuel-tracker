from datetime import date

from audit_contract_history import assess_observation


SESSION = date(2026, 6, 26)


def test_verified_observation_requires_same_continuous_price():
    prices = {"/RBQ26": {SESSION: 2.8257}, "/RBN26": {SESSION: 3.0068}}
    result = assess_observation("RB", SESSION, 2.8257, 2.8257, prices)
    assert result["status"] == "verified"
    assert result["correction_eligible"] is False
    assert result["stored_matches"] == ["/RBQ26"]


def test_mismatch_is_eligible_only_with_unique_active_contract_match():
    prices = {"/RBQ26": {SESSION: 2.8257}, "/RBN26": {SESSION: 3.0068}}
    result = assess_observation("RB", SESSION, 3.0068, 2.8257, prices)
    assert result["status"] == "mismatch"
    assert result["stored_matches"] == ["/RBN26"]
    assert result["active_matches"] == ["/RBQ26"]
    assert result["correction_eligible"] is True


def test_ambiguous_contract_match_never_authorizes_correction():
    prices = {"/HOQ26": {SESSION: 3.1}, "/HON26": {SESSION: 3.1}}
    result = assess_observation("HO", SESSION, 3.2, 3.1, prices)
    assert result["status"] == "mismatch"
    assert result["correction_eligible"] is False


def test_missing_continuous_history_is_unverifiable():
    result = assess_observation("RB", SESSION, 2.8257, None, {})
    assert result["status"] == "unverifiable"
    assert result["correction_eligible"] is False
