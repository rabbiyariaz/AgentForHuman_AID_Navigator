from nba.next_best_action import (
    REQUIRED_FOR_ELIGIBILITY,
    check_urgency,
    evaluate_next_action,
    find_missing_required_fields,
)


def _complete_fields():
    return {
        "household_size": 2,
        "annual_income": 40000,
        "home_type": "SF",
        "heating_fuel_type": "gas",
        "disconnection_status": False,
    }


def _confidences(value=0.9):
    return {field: value for field in REQUIRED_FOR_ELIGIBILITY}


def test_find_missing_required_fields_returns_null_required_fields():
    fields = _complete_fields()
    fields["annual_income"] = None
    confidences = _confidences()

    missing = find_missing_required_fields(fields, confidences)

    assert missing == ["annual_income"]


def test_find_missing_required_fields_treats_low_confidence_as_missing():
    fields = _complete_fields()
    confidences = _confidences()
    confidences["home_type"] = 0.49

    missing = find_missing_required_fields(fields, confidences)

    assert missing == ["home_type"]


def test_check_urgency_prefers_structured_disconnection_signal():
    urgent, reason = check_urgency({"disconnection_status": True})

    assert urgent is True
    assert reason == "disconnection_status is true"


def test_check_urgency_falls_back_to_original_text():
    urgent, reason = check_urgency(
        {"disconnection_status": None},
        original_text="The utility will shut off service today.",
    )

    assert urgent is True
    assert "original text" in reason


def test_check_urgency_returns_false_without_signal():
    urgent, reason = check_urgency(
        {"disconnection_status": False},
        original_text="I want to learn about assistance.",
    )

    assert urgent is False
    assert reason is None


def test_evaluate_next_action_requests_questions_for_missing_fields():
    fields = _complete_fields()
    fields["annual_income"] = None

    result = evaluate_next_action(fields, _confidences())

    assert result.status == "needs_followup"
    assert result.missing_fields == ["annual_income"]
    assert len(result.followup_questions) == 1
    assert result.urgency_flag is False


def test_evaluate_next_action_marks_complete_urgent_case():
    result = evaluate_next_action(
        {**_complete_fields(), "disconnection_status": True},
        _confidences(),
    )

    assert result.status == "flag_urgent"
    assert result.missing_fields == []
    assert result.urgency_flag is True


def test_evaluate_next_action_marks_complete_nonurgent_case_ready():
    result = evaluate_next_action(_complete_fields(), _confidences())

    assert result.status == "ready_for_eligibility"
    assert result.missing_fields == []
    assert result.urgency_flag is False
