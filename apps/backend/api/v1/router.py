"""Consolidated API v1 routing registry."""
from fastapi import APIRouter
from apps.backend.api.v1.chat import router as chat_router
from apps.backend.api.v1.prescriptions import router as prescriptions_router

api_router = APIRouter()

# Attach Domain Routers
api_router.include_router(prescriptions_router)
api_router.include_router(chat_router)


@api_router.get("/status", tags=["System"])
async def get_api_status():
    """Returns the operational status of API v1."""
    return {"status": "online", "api_version": "v1"}
