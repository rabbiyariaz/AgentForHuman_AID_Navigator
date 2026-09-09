import pytest

from rules.benefit_lookup import lookup_benefit


def test_lookup_benefit_zero_income_mf_one_person_gas():
    amount, source = lookup_benefit(0, 1, "MF", "gas")

    assert amount == pytest.approx(1512.0)
    assert source == "published_table"


def test_lookup_benefit_ten_thousand_sf_four_people_gas():
    amount, source = lookup_benefit(10000, 4, "SF", "gas")

    assert amount == pytest.approx(1800.0)
    assert source == "published_table"


def test_lookup_benefit_above_table_uses_extrapolated_minimum():
    amount, source = lookup_benefit(35000, 2, "SF", "gas")

    assert amount == pytest.approx(200.0)
    assert source == "extrapolated_minimum"


def test_lookup_benefit_household_size_six_uses_four_person_row():
    amount_4, source_4 = lookup_benefit(5000, 4, "SF", "gas")
    amount_6, source_6 = lookup_benefit(5000, 6, "SF", "gas")

    assert amount_6 == pytest.approx(amount_4)
    assert source_6 == source_4 == "published_table"


def test_lookup_benefit_unknown_fuel_type_raises_clear_error():
    with pytest.raises(ValueError, match="No FY26 benefit rows matched|fuel_type"):
        lookup_benefit(5000, 2, "SF", "unknown_fuel")
