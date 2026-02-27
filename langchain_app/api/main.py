"""
LangChain AI Platform - FastAPI 메인 애플리케이션
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from config.settings import settings
from database.session import get_async_db

# 로깅 설정
logging.basicConfig(
    level=settings.log_level.upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작 시 실행
    logger.info("🚀 LangChain AI Platform 시작...")
    logger.info(f"Ollama: {settings.OLLAMA_BASE_URL}")
    logger.info(f"PostgreSQL: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")
    logger.info(f"Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    logger.info(f"Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    
    # DB 테이블 초기화
    logger.info("🗄️ Initializing database tables...")
    try:
        from database.session import init_db
        init_db()
        logger.info("✅ Database tables initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
    
    yield
    
    # 종료 시 실행
    logger.info("👋 LangChain AI Platform 종료...")


# FastAPI 앱 생성 (lifespan 연결!)
app = FastAPI(lifespan=lifespan)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic 모델
class QuestionRequest(BaseModel):
    question: str
    category: Optional[str] = None
    limit: int = 10
    use_plan_execution: bool = True


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    confidence: float
    search_results: List[dict]


# 라우터 등록
@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "🤖 LangChain AI Platform API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "services": {
            "ollama": settings.OLLAMA_BASE_URL,
            "postgres": f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}",
            "milvus": f"{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
            "redis": f"{settings.REDIS_HOST}:{settings.REDIS_PORT}"
        }
    }


@app.post("/api/v1/qa", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Q&A 엔드포인트 (Async)

    RAG Chain을 통해 하이브리드 검색 + LLM 답변 생성

    Args:
        request: 질문 요청 (question, category, limit, use_plan_execution)
        db: 비동기 DB 세션

    Returns:
        QuestionResponse: 답변, 출처, 신뢰도, 검색 결과
    """
    try:
        from src.chains.rag_chain import MapleRAGChain

        chain = MapleRAGChain(db=db, use_milvus=True, use_neo4j=True)
        result = await chain.ainvoke(
            query=request.question,
            category=request.category,
            limit=request.limit,
            use_plan_execution=request.use_plan_execution,
        )

        return QuestionResponse(
            question=request.question,
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            search_results=result["search_results"],
        )

    except Exception as e:
        logger.error(f"Q&A 엔드포인트 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level
    )
