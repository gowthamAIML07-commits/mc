"""Application settings and environment configuration."""
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    APP_NAME: str = "AI Medicine Assistant"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    
    # Security & Tokens
    JWT_SECRET_KEY: str = "insecure_dev_secret_key_change_in_production_32_bytes"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 # 24 hours
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Upload Limits & Validation
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]
    MIN_IMAGE_DIMENSION: int = 32
    MAX_IMAGE_DIMENSION: int = 8192
    
    # Storage & Persistence
    DATABASE_URL: str = "postgresql+asyncpg://medicine_user:medicine_secure_pass@localhost:5432/medicine_ai"
    REDIS_URL: str = "redis://localhost:6379/0"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    
    # Phase 4 Model Paths (Relative to project root)
    DEVICE: str = "cpu" # cpu, cuda, or auto
    HANDWRITING_CHECKPOINT_PATH: str = str(PROJECT_ROOT / "checkpoints" / "phase4" / "handwriting" / "best_handwriting_crnn.pt")
    OCR_CHECKPOINT_PATH: str = str(PROJECT_ROOT / "checkpoints" / "phase4" / "ocr" / "best_ocr_extractor.pt")
    LAYOUT_CHECKPOINT_PATH: str = str(PROJECT_ROOT / "checkpoints" / "phase4" / "layout" / "best_layout_parser.pt")
    NORMALIZATION_KB_PATH: str = str(PROJECT_ROOT / "data" / "processed" / "normalization" / "normalized_medicines.json")
    NORMALIZATION_INDEX_PATH: str = str(PROJECT_ROOT / "data" / "processed" / "normalization" / "lexical_index.json")

    # Future AI Models (Phase 6+)
    MODEL_NAME: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    RERANKER_MODEL: str = "BAAI/bge-reranker-large"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
