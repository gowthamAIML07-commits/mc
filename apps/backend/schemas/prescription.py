"""Pydantic schemas for prescription extraction and medicine normalization."""
from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    name: Optional[str] = Field(None, description="Extracted patient name")
    age: Optional[int] = Field(None, description="Extracted patient age in years")
    gender: Optional[str] = Field(None, description="Extracted patient gender")


class DoctorInfo(BaseModel):
    name: Optional[str] = Field(None, description="Prescribing doctor name")
    registration: Optional[str] = Field(None, description="Medical council registration ID")
    clinic: Optional[str] = Field(None, description="Clinic or hospital name")


class MedicineResult(BaseModel):
    raw_text: str = Field(..., description="Raw text token as detected by OCR / handwriting recognizer")
    normalized_name: Optional[str] = Field(None, description="Official RxNorm standard clinical medicine name")
    rxnorm_id: Optional[str] = Field(None, description="Official RxNorm Concept Unique Identifier (RxCUI)")
    ingredient: Optional[str] = Field(None, description="Active pharmaceutical ingredient (INN)")
    strength: Optional[str] = Field(None, description="Dosage strength (e.g. 500 mg, 10 mg)")
    dose: Optional[str] = Field(None, description="Unit dose quantity (e.g. 1 tablet, 5 ml)")
    frequency: Optional[str] = Field(None, description="Administration schedule (e.g. 1-0-1 BD, OD, SOS)")
    duration: Optional[str] = Field(None, description="Course duration (e.g. 5 days, 1 month)")
    route: Optional[str] = Field("Oral", description="Administration route (e.g. Oral, Topical, IV)")
    ocr_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of text bounding box extraction")
    recognition_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of handwriting/OCR character recognition")
    normalization_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of RxNorm concept alignment")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Harmonized clinical confidence score")
    verification_status: Literal["verified", "unverified", "review_required"] = Field(
        ...,
        description="Clinical verification category: verified (high-confidence exact/alias/phonetic match), unverified (no match), or review_required (ambiguous/noisy/fuzzy match)"
    )
    review_reason: Optional[str] = Field(None, description="Explanation when human clinical pharmacist review is required")


class ProcessingTimings(BaseModel):
    upload_validation_ms: float = 0.0
    preprocessing_ms: float = 0.0
    ocr_ms: float = 0.0
    layout_association_ms: float = 0.0
    handwriting_recognition_ms: float = 0.0
    normalization_ms: float = 0.0
    total_pipeline_ms: float = 0.0


class PrescriptionResult(BaseModel):
    prescription_id: str = Field(..., description="Unique deterministic prescription transaction identifier")
    patient: Optional[PatientInfo] = Field(None, description="Extracted patient details")
    doctor: Optional[DoctorInfo] = Field(None, description="Extracted doctor and clinic credentials")
    date: Optional[str] = Field(None, description="Extracted prescription consultation date")
    medicines: List[MedicineResult] = Field(default_factory=list, description="Extracted and normalized medication items")
    instructions: List[str] = Field(default_factory=list, description="General physician advice and patient instructions")
    warnings: List[str] = Field(default_factory=list, description="Safety warnings, unverified flags, or review notices")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Aggregate confidence across all fields")
    timings: Optional[ProcessingTimings] = Field(None, description="Non-sensitive stage-by-stage pipeline execution latencies")


class PrescriptionResponse(BaseModel):
    success: bool = True
    request_id: str = Field(..., description="Correlation request ID")
    status: str = Field("completed", description="Status of processing: completed, review_required, or error")
    data: PrescriptionResult = Field(..., description="Extracted prescription entity data")
    warnings: List[str] = Field(default_factory=list, description="Top-level pipeline warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Model and versioning metadata")
