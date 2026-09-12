import pytest

from agent import tools


def _llm_response(**field_overrides):
    fields = {
        "household_size": 2,
        "annual_income": 40000,
        "home_type": "SF",
        "heating_fuel_type": "gas",
        "disconnection_status": False,
        "arrearage_amount": 0.0,
        "fuel_tank_percent": None,
        "heat_included_in_rent": False,
        "stated_deadline_or_urgency": None,
        "applicant_name": "Alex Carter",
    }
    fields.update(field_overrides)
    confidence = {
        field: 0.9 if value is not None else 0.0
        for field, value in fields.items()
    }
    return {"fields": fields, "confidence": confidence}


def _evidence(**overrides):
    fields = {
        "household_size": 2,
        "annual_income": 40000,
        "home_type": "SF",
        "heating_fuel_type": "gas",
        "disconnection_status": False,
        "applicant_name": "Alex Carter",
    }
    fields.update(overrides)
    return {"extracted": fields}


def test_build_extraction_output_returns_partial_result_for_missing_field():
    response = _llm_response(annual_income=None)

    result = tools._build_extraction_output(response, "text")

    assert result["extracted"] is None
    assert result["next_action"]["status"] == "needs_followup"
    assert "annual_income" in result["next_action"]["missing_fields"]
    assert "household_size" in result["partial_audit"]


def test_build_extraction_output_treats_low_confidence_as_followup():
    response = _llm_response()
    response["confidence"]["home_type"] = 0.4

    result = tools._build_extraction_output(response, "text")

    assert result["next_action"]["status"] == "needs_followup"
    assert result["next_action"]["missing_fields"] == ["home_type"]


def test_build_extraction_output_validates_complete_result():
    result = tools._build_extraction_output(_llm_response(), "document")

    assert result["extracted"]["household_size"] == 2
    assert result["audit"]["annual_income"]["source"] == "document"
    assert result["next_action"]["status"] == "ready_for_eligibility"


def test_build_extraction_output_rejects_schema_invalid_complete_result():
    response = _llm_response(home_type="detached_house")

    with pytest.raises(ValueError, match="failed schema validation"):
        tools._build_extraction_output(response, "text")


def test_check_contradiction_flags_income_beyond_five_percent():
    tools.begin_tool_state()

    result = tools.check_contradiction(
        _evidence(annual_income=40000),
        _evidence(annual_income=50000),
    )

    assert result["count"] == 1
    assert result["fields"] == ["annual_income"]
    contradiction_ref = result["contradictions_ref"]
    assert tools.get_tool_state()["contradictions"][contradiction_ref][0]["text_value"] == 40000


def test_check_contradiction_allows_income_within_five_percent():
    tools.begin_tool_state()

    result = tools.check_contradiction(
        _evidence(annual_income=40000),
        _evidence(annual_income=41000),
    )

    assert result["count"] == 0
    assert result["fields"] == []


def test_check_contradiction_flags_exact_field_mismatch():
    tools.begin_tool_state()

    result = tools.check_contradiction(
        _evidence(home_type="SF"),
        _evidence(home_type="MF"),
    )

    assert result["count"] == 1
    assert result["fields"] == ["home_type"]


def test_check_contradiction_accepts_matching_names_by_substring():
    tools.begin_tool_state()

    result = tools.check_contradiction(
        _evidence(applicant_name="Alex Carter"),
        _evidence(applicant_name="Alex"),
    )

    assert result["count"] == 0


def test_check_contradiction_returns_empty_result_without_document():
    tools.begin_tool_state()

    result = tools.check_contradiction(_evidence())

    assert result["count"] == 0
    assert result["fields"] == []
    assert tools.get_tool_state()["contradictions"][result["contradictions_ref"]] == []
