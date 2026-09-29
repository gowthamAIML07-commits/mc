"""Production Database Connection Pool and Session Management with Zero-Dependency Fallback."""
import logging
import sqlite3
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

logger = logging.getLogger("medicine_ai.database")

# Try importing SQLAlchemy if present; otherwise provide zero-dependency SQLite adapter
try:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.orm import declarative_base
    from apps.backend.core.config import settings

    Base = declarative_base()
    raw_db_url = getattr(settings, "DATABASE_URL", "sqlite+aiosqlite:///./data/medicine_ai.db")
    if raw_db_url.startswith("postgresql://"):
        raw_db_url = raw_db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine_kwargs: Dict[str, Any] = {"echo": False}
    if "sqlite" in raw_db_url:
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    else:
        engine_kwargs.update({
            "pool_size": 10,
            "max_overflow": 20,
            "pool_timeout": 30,
            "pool_pre_ping": True,
            "pool_recycle": 1800
        })

    async_engine = create_async_engine(raw_db_url, **engine_kwargs)
    AsyncSessionLocal = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    Base = object
    async_engine = None
    AsyncSessionLocal = None


async def get_db_session():
    """Dependency provider yielding managed database sessions."""
    if HAS_SQLALCHEMY and AsyncSessionLocal:
        async with AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    else:
        # Lightweight zero-dependency SQLite connection
        db_path = Path("data/medicine_ai.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


async def check_database_health() -> bool:
    """Probe database connectivity and health."""
    if HAS_SQLALCHEMY and async_engine:
        try:
            from sqlalchemy import text
            async with async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.warning(f"SQLAlchemy database health probe failed: {e}")
            return False
    else:
        try:
            db_path = Path("data/medicine_ai.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute("SELECT 1")
            res = cur.fetchone()
            conn.close()
            return res == (1,)
        except Exception as e:
            logger.warning(f"SQLite health probe failed: {e}")
            return False
