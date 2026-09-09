from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MATRIX_PATH = Path(__file__).resolve().parent / "data" / "benefit_matrix.json"

FUEL_COLUMN_MAP = {
    "gas": "gas_benefit",
    "oil": "oil_benefit",
    "electric": "electric_benefit",
}


def _load_matrix_payload() -> dict[str, Any]:
    """Load the published FY26 DC DOEE benefit table from the project JSON source."""
    if not MATRIX_PATH.exists():
        raise FileNotFoundError(f"Benefit matrix not found at {MATRIX_PATH}")

    with MATRIX_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = payload.get("benefit_matrix", [])
    if not rows:
        raise ValueError("Benefit matrix is empty; no FY26 rows are available for lookup")

    return payload


def _normalise_home_type(home_type: str) -> str:
    value = str(home_type).strip().upper()
    if value in {"SINGLE_FAMILY", "SF"}:
        return "SF"
    if value in {"MULTI_FAMILY", "MF"}:
        return "MF"
    return value


def _normalise_fuel_type(fuel_type: str) -> str:
    value = str(fuel_type).strip().lower()
    if value in {"natural_gas", "ng", "propane"}:
        return "gas"
    return value


def lookup_benefit(income: float, household_size: int, home_type: str, fuel_type: str) -> tuple[float, str]:
    """Return the FY26 DC DOEE benefit amount for a matched household profile.

    Source: DC DOEE FY26 LIHEAP benefit matrix and household-income bracket rules.
    """
    income_value = float(income)
    household_value = int(household_size)
    home_value = _normalise_home_type(home_type)
    fuel_value = _normalise_fuel_type(fuel_type)

    if home_value not in {"SF", "MF"}:
        raise ValueError(f"Unsupported home_type '{home_type}' for FY26 lookup; expected 'SF' or 'MF'.")

    payload = _load_matrix_payload()

    # HIR is a flat constant, not looked up per-row
    if fuel_value == "hir":
        return float(payload["hir_benefit_flat"]), "published_table"

    if fuel_value not in FUEL_COLUMN_MAP:
        raise ValueError(
            f"No FY26 benefit rows matched home_type={home_value}, fuel_type={fuel_value}, "
            f"household_size={household_value}; unsupported fuel_type='{fuel_type}'."
        )

    if income_value > 30000:
        return 200.0, "extrapolated_minimum"

    target_size = min(household_value, 4)
    rows = payload["benefit_matrix"]

    candidates = [
        row
        for row in rows
        if _normalise_home_type(row["home_type"]) == home_value
        and int(row["household_size"]) == target_size
    ]

    if not candidates:
        raise ValueError(
            f"No FY26 benefit rows matched home_type={home_value}, household_size={target_size}"
        )

    # DOEE rule: use the bracket that is the nearest value <= combined income
    matched_row = None
    matched_bracket = None
    for row in candidates:
        bracket = float(row["income_bracket"])
        if bracket <= income_value:
            if matched_bracket is None or bracket > matched_bracket:
                matched_bracket = bracket
                matched_row = row

    if matched_row is None:
        matched_row = sorted(candidates, key=lambda row: float(row["income_bracket"]))[0]

    column = FUEL_COLUMN_MAP[fuel_value]
    amount = float(matched_row[column])
    return amount, "published_table"