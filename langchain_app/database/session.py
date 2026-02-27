"""
PostgreSQL 세션 관리
SQLAlchemy 엔진 및 세션 팩토리 (Sync + Async)
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from config.settings import settings
from typing import Generator, AsyncGenerator
import logging

logger = logging.getLogger(__name__)

# ==================== 동기 엔진 (기존 스크립트용) ====================
engine = create_engine(
    settings.postgres_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    echo=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db() -> Generator:
    """
    동기 DB 세션 (스크립트용)
    
    Usage:
        db = SessionLocal()
        try:
            ...
        finally:
            db.close()
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


# ==================== 비동기 엔진 (FastAPI용) ====================
# PostgreSQL URL을 asyncpg용으로 변환
async_postgres_url = settings.postgres_url.replace("postgresql://", "postgresql+asyncpg://")

async_engine = create_async_engine(
    async_postgres_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    echo=True,
)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    비동기 DB 세션 (FastAPI용)
    
    Usage:
        @app.get("/items")
        async def read_items(db: AsyncSession = Depends(get_async_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Async database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


def init_db():
    """
    데이터베이스 초기화
    GraphRAG 필수 테이블만 생성 (users, maple_dictionary)
    """
    from .base import Base
    from .models import user, maple_dictionary  # 필수 모델만 import
    
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
        logger.info(f"   Tables: {list(Base.metadata.tables.keys())}")
        logger.info("   - users: 사용자 정보")
        logger.info("   - maple_dictionary: 용어 사전 (엔티티 추출)")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise
