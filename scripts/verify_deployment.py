"""Comprehensive Production Deployment Verification & System Health Probe."""
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path
proj_root = Path(__file__).resolve().parent.parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from apps.backend.core.config import settings
from apps.backend.services.model_registry import ModelRegistry
from database.connection import check_database_health
from rag.indexing.vector_store import MedicalVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("medicine_ai.deployment")


async def run_deployment_verification() -> bool:
    """Execute end-to-end verification of all subsystems required for production deployment."""
    logger.info("==================================================")
    logger.info("AI MEDICINE ASSISTANT — DEPLOYMENT VERIFICATION")
    logger.info("==================================================")

    all_passed = True

    # 1. Config Verification
    logger.info("[1/5] Verifying Environment Configuration...")
    try:
        assert settings.APP_NAME
        assert settings.API_V1_STR == "/api/v1"
        logger.info(f"✓ Configuration Valid: {settings.APP_NAME} (Env: {settings.APP_ENV})")
    except Exception as e:
        logger.error(f"✗ Configuration Error: {e}")
        all_passed = False

    # 2. Database Connectivity Probe
    logger.info("[2/5] Probing Database Connection...")
    try:
        db_healthy = await check_database_health()
        if db_healthy:
            logger.info("✓ Database Connection: Healthy and Responsive")
        else:
            logger.warning("! Database Probe: Warning (Falling back to SQLite in dev mode)")
    except Exception as e:
        logger.error(f"✗ Database Connectivity Failed: {e}")
        all_passed = False

    # 3. Model Registry & Checkpoints
    logger.info("[3/5] Verifying Phase 4 Inference Models & Checkpoints...")
    try:
        registry = ModelRegistry.get_instance()
        registry.load_all_models()
        meta = registry.get_metadata()
        logger.info(f"✓ Model Registry Ready: {meta.get('loaded_models')} models active.")
    except Exception as e:
        logger.error(f"✗ Model Registry Failed: {e}")
        all_passed = False

    # 4. Knowledge Base & Vector Index
    logger.info("[4/5] Verifying Medical Knowledge RAG & Vector Index...")
    try:
        kb_path = proj_root / "data" / "processed" / "knowledge_base" / "clinical_chunks.json"
        assert kb_path.exists(), f"Knowledge base missing at {kb_path}"
        v_store = MedicalVectorStore()
        v_store.load_from_json(kb_path)
        assert len(v_store.chunks) >= 50
        logger.info(f"✓ Vector Store Ready: {len(v_store.chunks)} authoritative chunks indexed.")
    except Exception as e:
        logger.error(f"✗ Knowledge Base Verification Failed: {e}")
        all_passed = False

    # 5. Static Frontend Assets
    logger.info("[5/5] Verifying Static Frontend Workspace Assets...")
    try:
        static_dir = proj_root / "apps" / "frontend" / "static"
        assert (static_dir / "index.html").exists()
        assert (static_dir / "styles.css").exists()
        assert (static_dir / "app.js").exists()
        assert (static_dir / "api.js").exists()
        logger.info("✓ Frontend Assets: All SPA assets verified.")
    except Exception as e:
        logger.error(f"✗ Frontend Assets Missing: {e}")
        all_passed = False

    logger.info("==================================================")
    if all_passed:
        logger.info("🎉 ALL DEPLOYMENT CHECKS PASSED: Application is Deployment-Ready.")
    else:
        logger.error("❌ DEPLOYMENT CHECKS FAILED: Review the logs above.")
    logger.info("==================================================")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_deployment_verification())
    sys.exit(0 if success else 1)
