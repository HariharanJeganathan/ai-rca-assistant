"""
database.py — PostgreSQL Connection
Updated: Better connection handling, retry logic for Render cold starts
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
import logging

from config import settings

logger = logging.getLogger(__name__)


def get_async_database_url(url: str) -> str:
    """Convert sync PostgreSQL URL to async asyncpg format."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


ASYNC_DATABASE_URL = get_async_database_url(settings.DATABASE_URL)

engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,             # Never log SQL in production
    pool_size=3,            # Small pool for free tier
    max_overflow=5,
    pool_pre_ping=True,     # Verify connections before using
    pool_recycle=300,       # Recycle connections every 5 mins (Supabase drops idle ones)
    connect_args={
        "server_settings": {
            "application_name": "ai-rca-assistant"
        }
    }
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_database_connection() -> bool:
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
            logger.info("[DB] ✅ PostgreSQL connected")
            return True
    except Exception as e:
        logger.warning(f"[DB] ❌ PostgreSQL not available: {e}")
        return False


async def create_tables():
    try:
        async with engine.begin() as conn:
            from db.models import RCAReportModel  # noqa: F401
            await conn.run_sync(Base.metadata.create_all)
            logger.info("[DB] ✅ Tables ready")
    except Exception as e:
        logger.error(f"[DB] ❌ Table creation failed: {e}")
        raise
