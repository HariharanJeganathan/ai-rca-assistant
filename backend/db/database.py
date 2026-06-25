"""
database.py — PostgreSQL Connection & Session Management
=========================================================
This file handles CONNECTING to PostgreSQL.

WHAT IS SQLALCHEMY?
  SQLAlchemy is a Python library that lets you talk to databases
  using Python code instead of raw SQL.

  Without SQLAlchemy:
    cursor.execute("INSERT INTO reports (id, title) VALUES ('1', 'Payment down')")

  With SQLAlchemy:
    db.add(RCAReportModel(id="1", title="Payment down"))
    db.commit()

  Much cleaner, safer (no SQL injection), and works with any database.

WHAT IS ASYNC?
  Normal database call:  App WAITS for DB to respond (blocking)
  Async database call:   App does other work while DB responds (non-blocking)

  For a web API that handles many users at once, async is much faster.
  We use asyncpg (async PostgreSQL driver) + SQLAlchemy async mode.

CONNECTION POOL:
  Instead of opening a new DB connection for every request (slow),
  we keep a pool of open connections and reuse them (fast).
  Like a taxi rank — cabs wait ready instead of being called fresh each time.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
import logging

from config import settings

logger = logging.getLogger(__name__)


# ============================================================
# Convert DATABASE_URL to async format
# ============================================================
# SQLAlchemy async requires "postgresql+asyncpg://" prefix
# But .env usually has "postgresql://" (sync format)
# We convert it automatically here

def get_async_database_url(url: str) -> str:
    """
    Convert sync PostgreSQL URL to async format.

    postgresql://user:pass@host/db
    → postgresql+asyncpg://user:pass@host/db

    asyncpg = the fast async PostgreSQL driver
    """
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        # Supabase sometimes uses "postgres://" (without ql)
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


ASYNC_DATABASE_URL = get_async_database_url(settings.DATABASE_URL)


# ============================================================
# Create the Database Engine
# ============================================================
# Engine = the connection factory
# It manages the connection pool automatically

engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=settings.DEBUG,        # If DEBUG=true, prints all SQL queries to console
                                # Very useful for development, turn off in production
    pool_size=5,                # Keep 5 connections open and ready
    max_overflow=10,            # Allow up to 10 extra connections during peak load
    pool_pre_ping=True,         # Test connections before using them (handles dropped connections)
    pool_recycle=3600,          # Recycle connections every hour (prevents stale connections)
)


# ============================================================
# Session Factory
# ============================================================
# A "session" is one unit of work with the database.
# Think of it like a shopping basket:
#   - You add items (add records)
#   - You commit (save everything to DB)
#   - Or you rollback (cancel everything if something went wrong)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,     # Keep objects accessible after commit
    autocommit=False,           # We control when to commit manually
    autoflush=False,            # We control when to flush manually
)


# ============================================================
# Base Class for all database models
# ============================================================
class Base(DeclarativeBase):
    """
    All database table models inherit from this class.
    It provides the metadata SQLAlchemy needs to create tables.

    Usage:
        class RCAReportModel(Base):
            __tablename__ = "rca_reports"
            ...
    """
    pass


# ============================================================
# Dependency — get a database session
# ============================================================
async def get_db():
    """
    FastAPI dependency that provides a database session.

    DEPENDENCY INJECTION:
      Instead of creating a DB session inside every route function,
      FastAPI injects it automatically via Depends(get_db).

      Usage in a route:
        @router.get("/reports")
        async def list_reports(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(RCAReportModel))
            ...

    The 'async with' ensures the session is ALWAYS closed,
    even if an error occurs — no connection leaks!
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session           # Give the session to the route function
            await session.commit()  # If no errors, save changes
        except Exception:
            await session.rollback()  # If error, undo all changes
            raise
        finally:
            await session.close()   # Always close the session


# ============================================================
# Health Check — test if DB is reachable
# ============================================================
async def check_database_connection() -> bool:
    """
    Test if we can connect to PostgreSQL.
    Used by the /health endpoint.

    Returns True if connected, False if not.
    """
    try:
        async with AsyncSessionLocal() as session:
            # Simple query — if this works, DB is connected
            await session.execute(text("SELECT 1"))
            logger.info("[DB] ✅ PostgreSQL connection successful")
            return True
    except Exception as e:
        logger.error(f"[DB] ❌ PostgreSQL connection failed: {e}")
        return False


# ============================================================
# Create all tables
# ============================================================
async def create_tables():
    """
    Creates all database tables defined in models.py.
    Called once at app startup.

    In production, use Alembic migrations instead.
    For this project, create_tables() is fine for simplicity.
    """
    try:
        async with engine.begin() as conn:
            # Import models so SQLAlchemy knows about them
            from db.models import RCAReportModel  # noqa: F401
            await conn.run_sync(Base.metadata.create_all)
            logger.info("[DB] ✅ Tables created successfully")
    except Exception as e:
        logger.error(f"[DB] ❌ Failed to create tables: {e}")
        raise
