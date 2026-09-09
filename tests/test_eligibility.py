import pytest

from rules.eligibility import check_crisis_eligibility, check_regular_eligibility


def _regular_case(**overrides):
    data = {
        "household_size": 1,
        "annual_income": 50000,
        "home_type": "SF",
        "heating_fuel_type": "gas",
        "disconnection_status": False,
        "arrearage_amount": 0.0,
        "fuel_tank_percent": None,
        "heat_included_in_rent": False,
    }
    data.update(overrides)
    return data


def test_check_regular_eligibility_income_50000_for_one_person_is_eligible():
    evidence = _regular_case()

    result = check_regular_eligibility(evidence)

    assert result["eligible"] is True
    assert result["income_limit"] == pytest.approx(61841.0)
    assert "61841" in result["rule_fired"]
    assert "annual_income_within_limit" in result["rule_fired"]


def test_check_regular_eligibility_income_65000_for_one_person_is_ineligible():
    evidence = _regular_case(annual_income=65000)

    result = check_regular_eligibility(evidence)

    assert result["eligible"] is False
    assert result["income_limit"] == pytest.approx(61841.0)
    assert "annual_income_exceeds_limit" in result["rule_fired"]
    assert "61841" in result["rule_fired"]


def test_check_regular_eligibility_boundary_income_for_three_person_household_is_eligible():
    evidence = _regular_case(household_size=3, annual_income=99897)

    result = check_regular_eligibility(evidence)

    assert result["eligible"] is True
    assert result["income_limit"] == pytest.approx(99897.0)
    assert "99897" in result["rule_fired"]
    assert "<=" in result["rule_fired"] or "within_limit" in result["rule_fired"]


def test_check_regular_eligibility_household_size_seven_uses_over_six_formula():
    # DOEE rule: for household_size > 6, add 3 percentage points to 132% per
    # additional person beyond 6, multiplied by the 4-person income limit.
    expected_limit = 118926.0 * (1.32 + 0.03 * 3)
    evidence = _regular_case(household_size=7, annual_income=expected_limit)

    result = check_regular_eligibility(evidence)

    assert result["eligible"] is True
    assert result["income_limit"] == pytest.approx(expected_limit)
    assert "household_size_7" in result["rule_fired"]
    assert str(expected_limit).split(".")[0] in result["rule_fired"]


def test_check_crisis_eligibility_disconnected_case_is_eligible_when_after_benefit_threshold_met():
    # SF, 1 person, $20,000 income, gas -> real DOEE benefit = $293
    # arrearage_after_benefit = 600 - 293 = 307
    evidence = _regular_case(
        annual_income=20000,
        arrearage_amount=600.0,
        disconnection_status=True,
        fuel_tank_percent=None,
    )

    result = check_crisis_eligibility(evidence)

    assert result["eligible"] is True
    assert result["arrearage_after_benefit"] == pytest.approx(307.0)
    assert "250" in result["rule_fired"]
    assert "disconnection_status" in result["rule_fired"]


def test_check_crisis_eligibility_without_disconnection_or_fuel_condition_is_not_eligible():
    evidence = _regular_case(
        annual_income=20000,
        arrearage_amount=600.0,
        disconnection_status=False,
        fuel_tank_percent=None,
    )

    result = check_crisis_eligibility(evidence)

    assert result["eligible"] is False
    assert result["arrearage_after_benefit"] == pytest.approx(307.0)
    assert "250" in result["rule_fired"]
    assert "not_met" in result["rule_fired"] or "neither" in result["rule_fired"]


def test_check_crisis_eligibility_exactly_250_after_benefit_is_eligible():
    # SF, 1 person, $20,000 income, gas -> benefit = $293
    # arrearage_amount set to 543 so arrearage_after_benefit lands exactly on 250
    evidence = _regular_case(
        annual_income=20000,
        arrearage_amount=543.0,
        disconnection_status=True,
        fuel_tank_percent=None,
    )

    result = check_crisis_eligibility(evidence)

    assert result["eligible"] is True
    assert result["arrearage_after_benefit"] == pytest.approx(250.0)
    assert "250" in result["rule_fired"]
    assert ">=" in result["rule_fired"] or ">=_250" in result["rule_fired"]


def test_check_crisis_eligibility_under_250_after_benefit_is_not_eligible():
    # arrearage_after_benefit = 499 - 293 = 206, under the 250 threshold
    evidence = _regular_case(
        annual_income=20000,
        arrearage_amount=499.0,
        disconnection_status=True,
        fuel_tank_percent=None,
    )

    result = check_crisis_eligibility(evidence)

    assert result["eligible"] is False
    assert result["arrearage_after_benefit"] == pytest.approx(206.0)
    assert "250" in result["rule_fired"]
    assert "below_threshold" in result["rule_fired"]


def test_check_crisis_eligibility_requires_regular_eligibility_first():
    evidence = _regular_case(annual_income=70000, arrearage_amount=600.0, disconnection_status=True)

    result = check_crisis_eligibility(evidence)

    assert result["eligible"] is False
    assert "regular_eligibility_failed_before_crisis_check" in result["rule_fired"]
    assert "annual_income_exceeds_limit" in result["rule_fired"]