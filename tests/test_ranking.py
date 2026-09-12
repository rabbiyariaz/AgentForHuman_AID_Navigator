from triage.ranking import (
    TIER_CONTRADICTION,
    TIER_CRISIS_ELIGIBLE,
    TIER_INCOMPLETE_NO_URGENCY,
    TIER_INELIGIBLE,
    TIER_REGULAR_ELIGIBLE,
    TIER_URGENT_INCOMPLETE,
    classify_case,
    rank_cases,
)


def _ready_case(**overrides):
    fields = {
        "household_size": 1,
        "annual_income": 20000,
        "home_type": "SF",
        "heating_fuel_type": "gas",
        "disconnection_status": False,
        "arrearage_amount": 0.0,
        "fuel_tank_percent": None,
        "heat_included_in_rent": False,
    }
    fields.update(overrides)
    audit = {
        field: {"value": value, "confidence": 0.9, "source": "text"}
        for field, value in fields.items()
    }
    return {
        "status": "ready_for_eligibility",
        "next_action": {"urgency_flag": False},
        "extraction_output": {"extracted": fields, "audit": audit},
    }


def _followup_case(urgent):
    return {
        "status": "needs_followup",
        "next_action": {
            "urgency_flag": urgent,
            "missing_fields": ["annual_income"],
        },
        "partial_audit": {
            "household_size": {"confidence": 0.8},
        },
    }


def test_classify_case_marks_crisis_eligible_case_as_tier_one():
    case = _ready_case(
        arrearage_amount=600.0,
        disconnection_status=True,
    )

    result = classify_case(case)

    assert result["tier"] == TIER_CRISIS_ELIGIBLE
    assert result["eligibility"]["crisis"]["eligible"] is True
    assert result["confidence"] == 0.9


def test_classify_case_marks_regular_eligible_case_as_tier_four():
    case = _ready_case(
        household_size=4,
        annual_income=35000,
    )

    result = classify_case(case)

    assert result["tier"] == TIER_REGULAR_ELIGIBLE
    assert result["eligibility"]["regular"]["eligible"] is True
    assert result["eligibility"]["crisis"]["eligible"] is False


def test_classify_case_marks_high_income_case_ineligible():
    case = _ready_case(
        household_size=2,
        annual_income=150000,
    )

    result = classify_case(case)

    assert result["tier"] == TIER_INELIGIBLE
    assert result["eligibility"]["regular"]["eligible"] is False


def test_classify_case_prioritizes_urgent_incomplete_case():
    result = classify_case(_followup_case(urgent=True))

    assert result["tier"] == TIER_URGENT_INCOMPLETE
    assert "urgent_signal_present" in result["reason"]
    assert result["eligibility"] is None


def test_classify_case_separates_nonurgent_incomplete_case():
    result = classify_case(_followup_case(urgent=False))

    assert result["tier"] == TIER_INCOMPLETE_NO_URGENCY
    assert "no_urgency_signal" in result["reason"]


def test_classify_case_prioritizes_contradictions_for_review():
    case = {
        "status": "needs_review",
        "next_action": {"urgency_flag": False},
        "partial_audit": {
            "annual_income": {"confidence": 0.7},
        },
        "contradictions": [{"field": "annual_income"}],
    }

    result = classify_case(case)

    assert result["tier"] == TIER_CONTRADICTION
    assert result["confidence"] == 0.7
    assert "annual_income" in result["reason"]


def test_rank_cases_orders_cases_by_tier_then_lowest_confidence():
    cases = [
        _ready_case(household_size=2, annual_income=150000),
        _ready_case(household_size=4, annual_income=35000),
        _followup_case(urgent=True),
        _ready_case(arrearage_amount=600.0, disconnection_status=True),
    ]

    ranked = rank_cases(cases)

    assert [case["triage"]["tier"] for case in ranked] == [
        TIER_CRISIS_ELIGIBLE,
        TIER_URGENT_INCOMPLETE,
        TIER_REGULAR_ELIGIBLE,
        TIER_INELIGIBLE,
    ]
    assert all("triage" in case for case in ranked)


def test_rank_cases_places_lower_confidence_case_first_within_same_tier():
    high_confidence = _ready_case(household_size=4, annual_income=35000)
    low_confidence = _ready_case(household_size=3, annual_income=35000)
    low_confidence["extraction_output"]["audit"]["annual_income"]["confidence"] = 0.55

    ranked = rank_cases([high_confidence, low_confidence])

    assert ranked[0]["triage"]["tier"] == TIER_REGULAR_ELIGIBLE
    assert ranked[0]["triage"]["confidence"] == 0.55
    assert ranked[1]["triage"]["confidence"] == 0.9
