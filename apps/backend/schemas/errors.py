"""Pydantic schemas for standardized API error responses."""
from typing import Any, Optional
from pydantic import BaseModel, Field


class APIErrorResponse(BaseModel):
    success: bool = False
    error_code: str = Field(..., description="Standard machine-readable error code")
    message: str = Field(..., description="Human-readable safe error message")
    details: Optional[Any] = Field(None, description="Detailed validation error payload or field specifics")
