#!/usr/bin/env python3
"""
전체 데이터베이스 초기화 스크립트
PostgreSQL → Milvus → Neo4j 순차 초기화
"""
import sys
from pathlib import Path
import time

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LANGCHAIN_APP_DIR = PROJECT_ROOT / "langchain_app"
sys.path.insert(0, str(LANGCHAIN_APP_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import asyncio
from database.session import AsyncSessionLocal, async_engine
from database.base import Base
from sqlalchemy import select, text
from database.models.maple_dictionary import MapleDictionary
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_postgres_connection():
    """PostgreSQL 연결 확인"""
    print("\n1️⃣ PostgreSQL 연결 확인...")
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(text("SELECT 1"))
            result.scalar()
        print("   ✅ PostgreSQL 연결 성공")
        return True
    except Exception as e:
        print(f"   ❌ PostgreSQL 연결 실패: {e}")
        return False


async def create_tables():
    """테이블 생성"""
    print("\n2️⃣ 테이블 생성...")
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("   ✅ 테이블 생성 완료")
        return True
    except Exception as e:
        print(f"   ❌ 테이블 생성 실패: {e}")
        return False


async def load_sql_file():
    """SQL 파일로 데이터 로드"""
    print("\n3️⃣ 초기 데이터 로드 (SQL)...")
    
    sql_file = PROJECT_ROOT / "init" / "01_maple_data.sql"
    
    if not sql_file.exists():
        print(f"   ⚠️ SQL 파일 없음: {sql_file}")
        print(f"   먼저 export_postgres_data.py를 실행하세요")
        return False
    
    try:
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # SQL 실행
        async with async_engine.begin() as conn:
            # 파일을 statement 단위로 분할 실행
            statements = [s.strip() for s in sql_content.split(';') if s.strip()]
            for stmt in statements:
                if stmt and not stmt.startswith('--'):
                    await conn.execute(text(stmt))
        
        # 결과 확인
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(MapleDictionary))
            count = len(result.scalars().all())
        
        print(f"   ✅ 데이터 로드 완료: {count}개 엔티티")
        return True
        
    except Exception as e:
        print(f"   ❌ 데이터 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


async def sync_to_milvus():
    """PostgreSQL → Milvus 동기화"""
    print("\n4️⃣ Milvus 동기화...")
    
    try:
        # sync_to_milvus.py 스크립트 호출
        import subprocess
        result = subprocess.run(
            [sys.executable, str(LANGCHAIN_APP_DIR / "scripts" / "sync_to_milvus.py")],
            cwd=str(LANGCHAIN_APP_DIR),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("   ✅ Milvus 동기화 완료")
            return True
        else:
            print(f"   ❌ Milvus 동기화 실패")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"   ⚠️ Milvus 동기화 실패 (무시하고 진행): {e}")
        return False


async def sync_to_neo4j():
    """PostgreSQL → Neo4j 동기화"""
    print("\n5️⃣ Neo4j 동기화...")
    
    try:
        # sync_to_neo4j.py 스크립트 호출
        import subprocess
        result = subprocess.run(
            [sys.executable, str(LANGCHAIN_APP_DIR / "scripts" / "sync_to_neo4j.py")],
            cwd=str(LANGCHAIN_APP_DIR),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("   ✅ Neo4j 동기화 완료")
            return True
        else:
            print(f"   ❌ Neo4j 동기화 실패")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"   ⚠️ Neo4j 동기화 실패 (무시하고 진행): {e}")
        return False


def check_ollama_model():
    """Ollama 모델 확인 & 자동 pull"""
    print("\n6️⃣ Ollama 모델 확인...")
    
    from config.settings import settings
    import requests
    
    try:
        # Ollama 서버 확인
        response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
        
        if response.status_code != 200:
            print(f"   ❌ Ollama 서버 응답 없음: {settings.OLLAMA_BASE_URL}")
            return False
        
        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]
        
        target_model = settings.OLLAMA_MODEL
        
        # 모델 존재 확인
        if any(target_model in name for name in model_names):
            print(f"   ✅ 모델 이미 존재: {target_model}")
            return True
        
        # 모델 없으면 pull
        print(f"   ⚠️ 모델 없음, Pull 시작: {target_model}")
        print(f"   (시간이 걸릴 수 있습니다...)")
        
        pull_response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/pull",
            json={"name": target_model},
            stream=True,
            timeout=3600  # 1시간
        )
        
        # 진행상황 출력
        for line in pull_response.iter_lines():
            if line:
                import json
                try:
                    data = json.loads(line)
                    if "status" in data:
                        print(f"      {data['status']}")
                except:
                    pass
        
        print(f"   ✅ 모델 Pull 완료: {target_model}")
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Ollama 서버 연결 실패: {settings.OLLAMA_BASE_URL}")
        print(f"   Ollama를 먼저 실행하세요: ollama serve")
        return False
    except Exception as e:
        print(f"   ❌ Ollama 모델 확인 실패: {e}")
        return False


async def main():
    """전체 초기화 실행"""
    print("="*80)
    print("🚀 AI Platform 초기화 스크립트")
    print("="*80)
    
    # Step 1: PostgreSQL 연결 확인
    if not await check_postgres_connection():
        print("\n❌ PostgreSQL 연결 실패, 중단합니다")
        print("   Docker가 실행 중인지 확인하세요: docker-compose up -d")
        return
    
    # Step 2: 테이블 생성
    if not await create_tables():
        print("\n❌ 테이블 생성 실패, 중단합니다")
        return
    
    # Step 3: 데이터 로드
    if not await load_sql_file():
        print("\n⚠️ 데이터 로드 실패, Milvus/Neo4j 동기화는 건너뜁니다")
        # 계속 진행 (빈 DB로 시작 가능)
    else:
        # Step 4: Milvus 동기화
        await sync_to_milvus()
        
        # Step 5: Neo4j 동기화
        await sync_to_neo4j()
    
    # Step 6: Ollama 모델 확인
    check_ollama_model()
    
    # 완료
    print("\n" + "="*80)
    print("✅ 초기화 완료!")
    print("="*80)
    print("\n다음 단계:")
    print("1. FastAPI 서버 시작: cd langchain_app && python api/main.py")
    print("2. 테스트: python scripts/test_answer_generator.py '도적 전직 어디?'")
    print()


if __name__ == "__main__":
    asyncio.run(main())
