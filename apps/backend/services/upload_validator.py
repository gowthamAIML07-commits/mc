"""Prescription image upload security and structural validation service."""
import io
import os
from typing import Tuple
from PIL import Image

from apps.backend.core.config import settings


class UploadValidationError(Exception):
    """Base exception for upload validation failures."""
    def __init__(self, error_code: str, message: str):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class ImageUploadValidator:
    """Secure validator for prescription image payloads."""

    @classmethod
    def validate_image_bytes(cls, file_bytes: bytes, filename: str = "upload.png") -> Image.Image:
        """Thoroughly validate image bytes, dimensions, and integrity."""
        # 1. Check presence and non-emptiness
        if not file_bytes or len(file_bytes) == 0:
            raise UploadValidationError("EMPTY_FILE", "The uploaded file is empty.")

        # 2. Check File Size Limit
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise UploadValidationError(
                "FILE_TOO_LARGE",
                f"File size ({len(file_bytes) / (1024*1024):.2f} MB) exceeds maximum allowed limit of {settings.MAX_UPLOAD_SIZE_MB} MB."
            )

        # 3. Check Extension
        _, ext = os.path.splitext(filename.lower())
        if ext and ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
            raise UploadValidationError(
                "UNSUPPORTED_EXTENSION",
                f"File extension '{ext}' is not supported. Supported extensions: {', '.join(settings.ALLOWED_IMAGE_EXTENSIONS)}."
            )

        # 4. Decode and Verify with Pillow (Guards against fake image files)
        try:
            image_stream = io.BytesIO(file_bytes)
            img = Image.open(image_stream)
            img.verify() # Checks image header and structural integrity
        except Exception as e:
            raise UploadValidationError(
                "CORRUPTED_IMAGE",
                f"Corrupted or unreadable image file: {str(e)}"
            )

        # 5. Re-open for mode/dimension verification (Image.verify consumes the stream)
        try:
            image_stream.seek(0)
            img = Image.open(image_stream)
            img_format = (img.format or "").upper()
            
            allowed_formats = ["JPEG", "PNG", "WEBP", "MPO"]
            if img_format not in allowed_formats:
                raise UploadValidationError(
                    "UNSUPPORTED_FORMAT",
                    f"Decoded image format '{img_format}' is not supported. Supported formats: JPEG, PNG, WEBP."
                )

            # 6. Check Dimensions (Guards against decompression bombs / tiny icons)
            width, height = img.size
            min_dim = settings.MIN_IMAGE_DIMENSION
            max_dim = settings.MAX_IMAGE_DIMENSION

            if width < min_dim or height < min_dim:
                raise UploadValidationError(
                    "IMAGE_TOO_SMALL",
                    f"Image dimensions ({width}x{height} px) are too small for medical OCR. Minimum required: {min_dim}x{min_dim} px."
                )

            if width > max_dim or height > max_dim:
                raise UploadValidationError(
                    "IMAGE_TOO_LARGE",
                    f"Image dimensions ({width}x{height} px) exceed maximum permitted limit of {max_dim}x{max_dim} px."
                )

            # Convert to standard RGB to prevent alpha/palette issues
            img_rgb = img.convert("RGB")
            return img_rgb

        except UploadValidationError:
            raise
        except Exception as e:
            raise UploadValidationError("DECODING_FAILED", f"Failed to decode image: {str(e)}")
