"""Prescription upload and inference API endpoints."""
import logging
import time
import uuid
from typing import Any, Dict
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from apps.backend.schemas.errors import APIErrorResponse
from apps.backend.schemas.prescription import PrescriptionResponse
from apps.backend.services.model_registry import ModelRegistry
from apps.backend.services.prescription_service import PrescriptionInferenceService
from apps.backend.services.upload_validator import ImageUploadValidator, UploadValidationError

logger = logging.getLogger("medicine_ai.api.prescriptions")
router = APIRouter(prefix="/prescriptions", tags=["Prescriptions"])


def get_inference_service() -> PrescriptionInferenceService:
    """Dependency injection for prescription inference service."""
    return PrescriptionInferenceService()


@router.post(
    "/upload-and-extract",
    response_model=PrescriptionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": APIErrorResponse, "description": "Invalid, corrupted, or oversized prescription image"},
        500: {"model": APIErrorResponse, "description": "Inference or model execution failure"},
        503: {"model": APIErrorResponse, "description": "Models not loaded or service unavailable"}
    },
    summary="Upload and extract structured prescription data",
    description="Validates a prescription image, executes Phase 4 OCR / handwriting recognition, associates layout fields, and normalizes medications against RxNorm/RxTerms."
)
async def upload_and_extract_prescription(
    file: UploadFile = File(..., description="Prescription image file (JPEG, PNG, WEBP, max 10MB)"),
    service: PrescriptionInferenceService = Depends(get_inference_service)
):
    """Secure endpoint for uploading prescription image and receiving verified clinical JSON."""
    request_id = f"req_{uuid.uuid4().hex[:10]}"
    t_start = time.time()
    logger.info(f"Incoming prescription upload (request_id={request_id}, filename={file.filename})")

    # 1. Check Model Readiness
    registry = ModelRegistry.get_instance()
    if not registry.is_ready():
        logger.error(f"Inference request {request_id} rejected: Models not initialized.")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=APIErrorResponse(
                error_code="MODELS_NOT_READY",
                message="Prescription inference models are not initialized or ready."
            ).model_dump()
        )

    # 2. Read and Validate File Payload
    try:
        file_bytes = await file.read()
        validated_image = ImageUploadValidator.validate_image_bytes(
            file_bytes=file_bytes,
            filename=file.filename or "upload.png"
        )
    except UploadValidationError as e:
        logger.warning(f"Upload validation failed for {request_id}: [{e.error_code}] {e.message}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=APIErrorResponse(
                error_code=e.error_code,
                message=e.message
            ).model_dump()
        )
    except Exception as e:
        logger.error(f"Unexpected upload processing error for {request_id}: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=APIErrorResponse(
                error_code="INVALID_UPLOAD",
                message="Failed to process uploaded file."
            ).model_dump()
        )

    # 3. Execute Inference Pipeline
    try:
        prescription_result = service.process_prescription(
            image=validated_image,
            filename=file.filename
        )
    except Exception as e:
        logger.error(f"Inference failure for {request_id}: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIErrorResponse(
                error_code="INFERENCE_ERROR",
                message="An error occurred during prescription extraction and normalization."
            ).model_dump()
        )

    # 4. Construct API Response
    status_label = "review_required" if any(m.verification_status == "review_required" for m in prescription_result.medicines) else "completed"
    
    elapsed_ms = round((time.time() - t_start) * 1000, 2)
    logger.info(f"Prescription extraction {request_id} completed in {elapsed_ms}ms (Status={status_label})")

    return PrescriptionResponse(
        success=True,
        request_id=request_id,
        status=status_label,
        data=prescription_result,
        warnings=prescription_result.warnings,
        metadata={
            "models": registry.get_metadata(),
            "execution_time_ms": elapsed_ms
        }
    )
