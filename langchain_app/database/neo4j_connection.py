"""
Neo4j 연결 관리 (Sync + Async)
"""
from neo4j import GraphDatabase, AsyncGraphDatabase
from typing import Optional
import logging
import os
from pathlib import Path

# .env 로드 (파일 로드 시점에)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

logger = logging.getLogger(__name__)


# ==================== 동기 연결 (기존 스크립트용) ====================
class Neo4jConnection:
    """Neo4j 동기 연결 관리 클래스 (Singleton)"""
    
    _instance: Optional['Neo4jConnection'] = None
    _driver = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._driver is None:
            self._initialize_driver()
    
    def _initialize_driver(self):
        """Neo4j Driver 초기화"""
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "changeme_neo4j")
        
        try:
            self._driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_user, neo4j_password)
            )
            # 연결 테스트
            self._driver.verify_connectivity()
            logger.info(f"✅ Neo4j 연결 성공: {neo4j_uri}")
        except Exception as e:
            logger.error(f"❌ Neo4j 연결 실패: {e}")
            raise
    
    def get_session(self):
        """Neo4j Session 반환"""
        if self._driver is None:
            self._initialize_driver()
        return self._driver.session(database="neo4j")
    
    def close(self):
        """연결 종료"""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j 연결 종료")
            self._driver = None
    
    def execute_query(self, query: str, parameters: dict = None):
        """
        Cypher 쿼리 실행
        
        Args:
            query: Cypher 쿼리
            parameters: 쿼리 파라미터
            
        Returns:
            쿼리 결과
        """
        with self.get_session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def execute_write(self, query: str, parameters: dict = None):
        """
        쓰기 트랜잭션 실행
        
        Args:
            query: Cypher 쿼리
            parameters: 쿼리 파라미터
        """
        with self.get_session() as session:
            session.execute_write(lambda tx: tx.run(query, parameters or {}))


# ==================== 비동기 연결 (FastAPI용) ====================
class AsyncNeo4jConnection:
    """Neo4j 비동기 연결 관리 클래스 (Singleton)"""
    
    _instance: Optional['AsyncNeo4jConnection'] = None
    _driver = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._driver is None:
            self._initialize_driver()
    
    def _initialize_driver(self):
        """Neo4j Async Driver 초기화"""
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "changeme_neo4j")
        
        try:
            self._driver = AsyncGraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_user, neo4j_password)
            )
            logger.info(f"✅ Neo4j Async 연결 준비 완료: {neo4j_uri}")
        except Exception as e:
            logger.error(f"❌ Neo4j Async 연결 실패: {e}")
            raise
    
    async def verify_connectivity(self):
        """비동기 연결 테스트"""
        try:
            await self._driver.verify_connectivity()
            logger.info("✅ Neo4j Async 연결 확인 완료")
        except Exception as e:
            logger.error(f"❌ Neo4j Async 연결 확인 실패: {e}")
            raise
    
    def get_session(self):
        """Neo4j Async Session 반환"""
        if self._driver is None:
            self._initialize_driver()
        return self._driver.session(database="neo4j")
    
    async def close(self):
        """연결 종료"""
        if self._driver:
            await self._driver.close()
            logger.info("Neo4j Async 연결 종료")
            self._driver = None
    
    async def _reconnect(self):
        """드라이버 재연결 (이벤트 루프가 바뀌었을 때 사용)"""
        try:
            if self._driver:
                await self._driver.close()
        except Exception:
            pass
        self._driver = None
        self._initialize_driver()
        logger.info("✅ Neo4j Async 드라이버 재연결 완료")

    async def execute_query(self, query: str, parameters: dict = None):
        """
        비동기 Cypher 쿼리 실행 (연결 끊김 시 자동 재연결)

        Args:
            query: Cypher 쿼리
            parameters: 쿼리 파라미터

        Returns:
            쿼리 결과
        """
        try:
            async with self.get_session() as session:
                result = await session.run(query, parameters or {})
                records = await result.data()
                return records
        except Exception as e:
            # TCPTransport closed 등 연결 끊김 감지 → 재연결 후 재시도
            if "closed" in str(e).lower() or "handler is closed" in str(e).lower():
                logger.warning(f"Neo4j 연결 끊김 감지, 재연결 시도: {e}")
                await self._reconnect()
                async with self.get_session() as session:
                    result = await session.run(query, parameters or {})
                    records = await result.data()
                    return records
            raise
    
    async def execute_write(self, query: str, parameters: dict = None):
        """
        비동기 쓰기 트랜잭션 실행
        
        Args:
            query: Cypher 쿼리
            parameters: 쿼리 파라미터
        """
        async with self.get_session() as session:
            await session.execute_write(lambda tx: tx.run(query, parameters or {}))


# 싱글톤 인스턴스
neo4j_conn = Neo4jConnection()
async_neo4j_conn = AsyncNeo4jConnection()


def get_neo4j_session():
    """Neo4j Session 반환 (동기, 편의 함수)"""
    return neo4j_conn.get_session()


def execute_cypher(query: str, parameters: dict = None):
    """Cypher 쿼리 실행 (동기, 편의 함수)"""
    return neo4j_conn.execute_query(query, parameters)


def get_async_neo4j_session():
    """Neo4j Async Session 반환 (비동기, 편의 함수)"""
    return async_neo4j_conn.get_session()


async def execute_async_cypher(query: str, parameters: dict = None):
    """Cypher 쿼리 실행 (비동기, 편의 함수)"""
    return await async_neo4j_conn.execute_query(query, parameters)
