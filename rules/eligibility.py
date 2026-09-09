from __future__ import annotations

from typing import Any

from rules.benefit_lookup import lookup_benefit


REGULAR_INCOME_LIMITS = {
    1: 61841.0,
    2: 80869.0,
    3: 99897.0,
    4: 118926.0,
    5: 137954.0,
    6: 156982.0,
}


def _extract_fields(evidence: Any) -> dict[str, Any]:
    if hasattr(evidence, "extracted"):
        data = evidence.extracted
        if hasattr(data, "model_dump"):
            return data.model_dump()
        return dict(data)
    if isinstance(evidence, dict):
        if "extracted" in evidence:
            data = evidence["extracted"]
            if hasattr(data, "model_dump"):
                return data.model_dump()
            return dict(data)
        return evidence
    raise TypeError("evidence must be an ExtractionOutput model instance or a dict-like object")


def _regular_income_limit(household_size: int) -> float:
    """Return the applicable FY26 DC DOEE regular-income limit for household size."""
    if household_size <= 6:
        return float(REGULAR_INCOME_LIMITS.get(household_size, REGULAR_INCOME_LIMITS[6]))

    base_limit = REGULAR_INCOME_LIMITS[4]
    additional_people = max(0, household_size - 4)
    multiplier = 1.32 + (0.03 * additional_people)
    return float(base_limit * multiplier)


def check_regular_eligibility(evidence: Any) -> dict[str, Any]:
    """Evaluate whether an applicant meets the regular FY26 DC DOEE income rule.

    Source: DC DOEE FY26 LIHEAP income eligibility standards.
    """
    fields = _extract_fields(evidence)
    household_size = int(fields["household_size"])
    annual_income = float(fields["annual_income"])
    home_type = str(fields["home_type"])
    fuel_type = str(fields["heating_fuel_type"])

    income_limit = _regular_income_limit(household_size)
    if annual_income > income_limit:
        return {
            "eligible": False,
            "rule_fired": f"annual_income_exceeds_limit_for_household_size_{household_size}: {annual_income} > {income_limit}",
            "income_limit": float(income_limit),
            "benefit_amount": 0.0,
        }

    benefit_amount, source = lookup_benefit(
        income=annual_income,
        household_size=household_size,
        home_type=home_type,
        fuel_type=fuel_type,
    )

    return {
        "eligible": True,
        "rule_fired": f"annual_income_within_limit_for_household_size_{household_size}: {annual_income} <= {income_limit}",
        "income_limit": float(income_limit),
        "benefit_amount": float(benefit_amount),
    }


def check_crisis_eligibility(evidence: Any) -> dict[str, Any]:
    """Evaluate whether an applicant qualifies under the FY26 DC DOEE crisis rule.

    Source: DC DOEE FY26 LIHEAP crisis and emergency utility-assistance criteria.
    """
    fields = _extract_fields(evidence)
    regular_result = check_regular_eligibility(evidence)
    benefit_amount = float(regular_result["benefit_amount"])

    arrearage_amount = float(fields.get("arrearage_amount") or 0.0)
    disconnection_status = bool(fields.get("disconnection_status", False))
    fuel_tank_percent = fields.get("fuel_tank_percent")
    if fuel_tank_percent is not None:
        fuel_tank_percent = float(fuel_tank_percent)

    arrearage_after_benefit = arrearage_amount - benefit_amount

    if not regular_result["eligible"]:
        return {
            "eligible": False,
            "rule_fired": f"regular_eligibility_failed_before_crisis_check: {regular_result['rule_fired']}",
            "arrearage_after_benefit": float(arrearage_after_benefit),
        }

    if arrearage_after_benefit >= 250 and disconnection_status is True:
        return {
            "eligible": True,
            "rule_fired": (
                "arrearage_after_benefit_>=_250_and_disconnection_status_is_true: "
                f"{arrearage_after_benefit} >= 250 and disconnection_status=True"
            ),
            "arrearage_after_benefit": float(arrearage_after_benefit),
        }

    if arrearage_after_benefit >= 250 and fuel_tank_percent is not None and fuel_tank_percent <= 5:
        return {
            "eligible": True,
            "rule_fired": (
                "arrearage_after_benefit_>=_250_and_fuel_tank_percent_<=_5: "
                f"{arrearage_after_benefit} >= 250 and fuel_tank_percent={fuel_tank_percent} <= 5"
            ),
            "arrearage_after_benefit": float(arrearage_after_benefit),
        }

    if arrearage_after_benefit < 250:
        return {
            "eligible": False,
            "rule_fired": f"arrearage_after_benefit_below_threshold: {arrearage_after_benefit} < 250",
            "arrearage_after_benefit": float(arrearage_after_benefit),
        }

    return {
        "eligible": False,
        "rule_fired": (
            "crisis_conditions_not_met: arrearage_after_benefit_>=_250_but_"
            "neither_disconnection_status_is_true_nor_fuel_tank_percent_<=_5"
        ),
        "arrearage_after_benefit": float(arrearage_after_benefit),
    }
