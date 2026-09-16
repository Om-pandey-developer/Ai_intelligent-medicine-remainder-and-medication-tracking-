"""
PillSync OCR Scanner Pydantic Schemas.

Defines request/response validation models for the OCR
Prescription Scanner Engine endpoints and prescription history.
"""

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


# ===================================================================
# OCR Scan Response
# ===================================================================

class OCRScanResponse(BaseModel):
    """
    Response returned after scanning a prescription image.

    Contains the extracted medicine details along with the raw
    OCR text and a confidence score for transparency.
    Includes all clinical enrichment fields for full pipeline compatibility.
    """

    medicine_name: Optional[str] = Field(
        None,
        examples=["Amoxicillin"],
        description="Extracted medicine name from the prescription",
    )
    dosage: Optional[str] = Field(
        None,
        examples=["500mg"],
        description="Extracted dosage (e.g., '500mg', '250mg')",
    )
    frequency: Optional[str] = Field(
        None,
        examples=["1-0-1"],
        description="Extracted frequency pattern (e.g., '1-0-1', 'twice daily')",
    )
    daily_frequency: Optional[int] = Field(
        1,
        examples=[2],
        description="Calculated integer number of times per day",
    )
    dosage_form: Optional[str] = Field(
        "Tablet",
        examples=["Tablet", "Capsule", "Syrup"],
        description="Dosage form (Tablet, Capsule, Syrup, Injection, Drops, Inhaler, Ointment)",
    )
    disease_category: Optional[str] = Field(
        "General Healthcare",
        examples=["Antibiotics", "Diabetes"],
        description="Inferred therapeutic category",
    )
    initial_quantity: Optional[int] = Field(
        30,
        examples=[30],
        description="Total units prescribed or calculated from duration",
    )
    quantity_per_dose: Optional[int] = Field(
        1,
        examples=[1],
        description="Number of units taken per dose",
    )
    instructions: Optional[str] = Field(
        None,
        examples=["Take after meals"],
        description="Administration instructions and dietary notes",
    )
    verified: bool = Field(
        False,
        description="Whether medicine was verified against a known catalog",
    )
    matched_medicine: Optional[str] = Field(
        None,
        description="Catalog-matched medicine name (if verified)",
    )
    generic_salt: Optional[str] = Field(
        None,
        description="Active pharmaceutical ingredient / generic salt name",
    )
    raw_text: str = Field(
        ...,
        description="Raw text extracted from the prescription image by OCR",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        examples=[0.85],
        description="OCR extraction confidence score (0.0 to 1.0)",
    )
    scan_id: Optional[str] = Field(
        None,
        description="MongoDB document ID of the saved scan result",
    )
    medicines: Optional[List[Any]] = Field(
        default=None,
        description="Full list of extracted medications (from AI Vision or multi-medicine parsing)",
    )


# ===================================================================
# Prescription History Schemas
# ===================================================================

class PrescriptionHistoryItem(BaseModel):
    """A single prescription scan record from MongoDB."""

    scan_id: str = Field(
        ...,
        description="MongoDB document ID",
    )
    filename: str = Field(
        ...,
        description="Original uploaded file name",
    )
    raw_text: str = Field(
        ...,
        description="Raw OCR-extracted text",
    )
    confidence_score: float = Field(
        ...,
        description="OCR extraction confidence (0.0 to 1.0)",
    )
    parsed_data: dict = Field(
        default_factory=dict,
        description="NLP-parsed fields (medicine_name, dosage, frequency)",
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of the scan",
    )


class PrescriptionHistoryResponse(BaseModel):
    """Paginated prescription scan history."""

    scans: list[PrescriptionHistoryItem]
    total: int
    page: int
    page_size: int


class PrescriptionDetailResponse(BaseModel):
    """Detailed view of a single prescription scan."""

    scan_id: str
    user_id: str
    filename: str
    raw_text: str
    confidence_score: float
    parsed_data: dict = Field(default_factory=dict)
    created_at: Optional[datetime] = None
