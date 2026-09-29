"""FastAPI application entrypoint and middleware orchestration."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from apps.backend.api.v1.router import api_router
from apps.backend.core.config import settings
from apps.backend.core.security_headers import (
    CorrelationIdMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware
)
from apps.backend.schemas.errors import APIErrorResponse
from apps.backend.services.model_registry import ModelRegistry, ModelRegistryError

logger = logging.getLogger("medicine_ai.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager loading Phase 4 inference models at startup."""
    logger.info("Initializing AI Medicine Assistant inference engine...")
    try:
        registry = ModelRegistry.get_instance()
        registry.load_all_models()
        logger.info("✓ ModelRegistry initialized successfully.")
    except ModelRegistryError as e:
        logger.error(f"FATAL: ModelRegistry initialization failed: {e}")
        # In development/test, keep server running for diagnostics
    except Exception as e:
        logger.error(f"Unexpected error during model initialization: {e}")
    
    yield
    
    logger.info("Shutting down AI Medicine Assistant API...")


app = FastAPI(
    title=settings.APP_NAME,
    description="AI Medicine Assistant API: Prescription OCR, Medicine Recognition, Drug Verification, and Clinical RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Production Hardening Middlewares
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(RequestSizeLimitMiddleware, max_content_length=20 * 1024 * 1024)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach API v1 Router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Mount Static Frontend Workspace
static_dir = Path(__file__).resolve().parent.parent / "frontend" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", tags=["UI"])
    @app.get("/app", tags=["UI"])
    async def serve_frontend():
        """Serves the interactive Phase 7 Clinical Workspace."""
        return FileResponse(static_dir / "index.html")


@app.get("/health", tags=["Health"])
@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    """System liveness health check probe."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "api_prefix": settings.API_V1_STR
    }


@app.get("/ready", tags=["Health"])
@app.get("/api/v1/ready", tags=["Health"])
async def readiness_check():
    """System readiness probe checking model loading and inference availability."""
    registry = ModelRegistry.get_instance()
    if registry.is_ready():
        return {
            "status": "ready",
            "models_loaded": True,
            "metadata": registry.get_metadata()
        }
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "not_ready",
            "models_loaded": False,
            "message": "Required Phase 4 models are not loaded or failed validation."
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.backend.main:app", host="0.0.0.0", port=8000, reload=True)
