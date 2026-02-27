"""
LangChain AI Platform - 설정 관리
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, ClassVar
from pathlib import Path

# 환경변수 파일 경로 설정 (클래스 밖에서)
_BASE_PATH = Path(__file__).resolve().parent.parent.parent
_ENV_PATH = _BASE_PATH / ".env"

class Settings(BaseSettings):
    """config 설정"""
    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False  # 대소문자 구분 없이 읽기
    )

    """애플리케이션 설정"""
    # Ollama 설정
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL: str
    
    # Groq 설정 (Ollama fallback용)
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL_NAME: Optional[str] = None 
    
    # PostgreSQL 설정 (BIZ_POSTGRES_* 사용)
    BIZ_POSTGRES_HOST: str
    BIZ_POSTGRES_PORT: int
    BIZ_POSTGRES_DB: str
    BIZ_POSTGRES_USER: str
    BIZ_POSTGRES_PASSWORD: str
    
    # 하위 호환성을 위한 별칭 (deprecated, BIZ_POSTGRES_* 사용 권장)
    @property
    def POSTGRES_HOST(self) -> str:
        return self.BIZ_POSTGRES_HOST
    
    @property
    def POSTGRES_PORT(self) -> int:
        return self.BIZ_POSTGRES_PORT
    
    @property
    def POSTGRES_DB(self) -> str:
        return self.BIZ_POSTGRES_DB
    
    @property
    def POSTGRES_USER(self) -> str:
        return self.BIZ_POSTGRES_USER
    
    @property
    def POSTGRES_PASSWORD(self) -> str:
        return self.BIZ_POSTGRES_PASSWORD


    # Milvus 설정
    MILVUS_HOST: str
    MILVUS_PORT: int
    MILVUS_METRIC_PORT: int
    MILVUS_COLLECTION_NAME: str
    MILVUS_DIMENSION: int = 384  # sentence-transformers/all-MiniLM-L6-v2 기본값
    
    # Redis 설정
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str
    REDIS_DB: int

    # Langfuse 설정 (LLM Observability - Cloud 사용)
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_BASE_URL: str
    LANGFUSE_ENABLED: bool
    
    # API 설정
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    log_level: str = "info"
    
    # RAG 설정
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 5
    
    # 임베딩 모델
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # 리랭커
    RERANKER_API_URL: str
    
    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def async_postgres_url(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def redis_url(self) -> str:
        return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    



# 싱글톤 인스턴스
settings = Settings()
