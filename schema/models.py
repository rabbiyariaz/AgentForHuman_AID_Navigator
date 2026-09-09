from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


HomeType = Literal["SF", "MF"]
HeatingFuelType = Literal["gas", "oil", "electric", "HIR"]
EvidenceSource = Literal["text", "document", "both"]


class EvidenceField(BaseModel):
    """Audit record for a single extracted field: value, confidence, and source."""

    value: int | float | str | bool | None
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: EvidenceSource


class ExtractedApplicantFields(BaseModel):
    """Raw extracted values from the applicant's text and/or supporting document."""

    household_size: int
    annual_income: Optional[float] = None
    home_type: HomeType
    heating_fuel_type: HeatingFuelType
    disconnection_status: bool
    arrearage_amount: Optional[float] = None
    fuel_tank_percent: Optional[float] = None
    heat_included_in_rent: Optional[bool] = None
    stated_deadline_or_urgency: Optional[str] = None
    applicant_name: Optional[str] = None

    @field_validator("household_size")
    @classmethod
    def validate_household_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError("household_size must be at least 1")
        return value

    @field_validator("annual_income")
    @classmethod
    def validate_annual_income(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("annual_income must be greater than or equal to 0")
        return value

    @field_validator("arrearage_amount")
    @classmethod
    def validate_arrearage_amount(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("arrearage_amount must be greater than or equal to 0")
        return value

    @field_validator("fuel_tank_percent")
    @classmethod
    def validate_fuel_tank_percent(cls, value: Optional[float]) -> Optional[float]:
        if value is not None:
            if value < 0 or value > 100:
                raise ValueError("fuel_tank_percent must be between 0 and 100")
        return value

    @field_validator("stated_deadline_or_urgency")
    @classmethod
    def validate_stated_deadline_or_urgency(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            value = value.strip()
            if not value:
                return None
        return value

    @field_validator("applicant_name")
    @classmethod
    def validate_applicant_name(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            value = value.strip()
            if not value:
                return None
        return value

class ExtractionAudit(BaseModel):
    """Per-field audit with confidence score and source metadata for each extracted field."""

    household_size: EvidenceField
    annual_income: EvidenceField | None = None
    home_type: EvidenceField
    heating_fuel_type: EvidenceField
    disconnection_status: EvidenceField
    arrearage_amount: EvidenceField | None = None
    fuel_tank_percent: EvidenceField | None = None
    heat_included_in_rent: EvidenceField | None = None
    stated_deadline_or_urgency: EvidenceField | None = None
    applicant_name: EvidenceField | None = None

class ExtractionOutput(BaseModel):
    """Top-level extraction contract between the Strands agent and the deterministic rule engine."""

    extracted: ExtractedApplicantFields
    audit: ExtractionAudit


class OrchestrationOutput(BaseModel):
    """Structured response contract returned by the Strands orchestrator."""

    text_evidence: dict[str, object]
    document_evidence: dict[str, object] | None = None
    contradictions: list[dict[str, object]] = Field(default_factory=list)
