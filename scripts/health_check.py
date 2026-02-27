#!/usr/bin/env python3
"""
전체 서비스 Health Check
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LANGCHAIN_APP_DIR = PROJECT_ROOT / "langchain_app"
sys.path.insert(0, str(LANGCHAIN_APP_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import asyncio
import requests
from config.settings import settings
from database.session import AsyncSessionLocal
from database.neo4j_connection import async_neo4j_conn
from sqlalchemy import text
from pymilvus import connections, utility


def check_ollama():
    """Ollama 서버 확인"""
    try:
        response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama: {len(models)}개 모델")
            
            # 대상 모델 확인
            target_model = settings.OLLAMA_MODEL
            if any(target_model in m["name"] for m in models):
                print(f"   ✅ 모델 존재: {target_model}")
            else:
                print(f"   ⚠️ 모델 없음: {target_model}")
                print(f"      실행: ollama pull {target_model}")
            return True
        else:
            print(f"❌ Ollama: 응답 오류 ({response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Ollama: 연결 실패 - {e}")
        return False


async def check_postgres():
    """PostgreSQL 확인"""
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(text("SELECT COUNT(*) FROM maple_dictionary"))
            count = result.scalar()
        print(f"✅ PostgreSQL: {count}개 엔티티")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL: {e}")
        return False


def check_milvus():
    """Milvus 확인"""
    try:
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT
        )
        
        # 컬렉션 확인
        collections = utility.list_collections()
        print(f"✅ Milvus: {len(collections)}개 컬렉션 {collections}")
        
        if "maple_qa" in collections:
            from pymilvus import Collection
            col = Collection("maple_qa")
            print(f"   ✅ maple_qa: {col.num_entities}개 벡터")
        
        return True
    except Exception as e:
        print(f"❌ Milvus: {e}")
        return False


async def check_neo4j():
    """Neo4j 확인"""
    try:
        await async_neo4j_conn.verify_connectivity()
        
        # 노드 개수 확인
        result = await async_neo4j_conn.execute_query("MATCH (n) RETURN count(n) as count")
        count = result[0]["count"] if result else 0
        
        print(f"✅ Neo4j: {count}개 노드")
        return True
    except Exception as e:
        print(f"❌ Neo4j: {e}")
        return False


def check_redis():
    """Redis 확인"""
    try:
        import redis
        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            decode_responses=True
        )
        r.ping()
        print(f"✅ Redis: 연결 성공")
        return True
    except Exception as e:
        print(f"❌ Redis: {e}")
        return False


async def main():
    """전체 Health Check"""
    print("="*80)
    print("🏥 AI Platform Health Check")
    print("="*80)
    print()
    
    results = {}
    
    # 각 서비스 확인
    print("📊 서비스 상태:")
    print("-" * 80)
    
    results["ollama"] = check_ollama()
    results["postgres"] = await check_postgres()
    results["milvus"] = check_milvus()
    results["neo4j"] = await check_neo4j()
    results["redis"] = check_redis()
    
    print("-" * 80)
    
    # 요약
    healthy_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print()
    print("="*80)
    if healthy_count == total_count:
        print(f"✅ 모든 서비스 정상 ({healthy_count}/{total_count})")
    else:
        print(f"⚠️ 일부 서비스 오류 ({healthy_count}/{total_count})")
        print()
        print("실패한 서비스:")
        for service, status in results.items():
            if not status:
                print(f"  - {service}")
    print("="*80)
    print()


if __name__ == "__main__":
    asyncio.run(main())
