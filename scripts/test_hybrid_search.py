#!/usr/bin/env python3
"""
Hybrid Search 테스트 스크립트 (Async, Milvus 없이도 작동)
"""
import sys
import asyncio
from pathlib import Path

# 경로 설정
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LANGCHAIN_APP_DIR = PROJECT_ROOT / "langchain_app"
sys.path.insert(0, str(LANGCHAIN_APP_DIR))

# .env 로드
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

print("✅ 1. 환경 설정 완료")

# Import
from database.session import AsyncSessionLocal
from src.retrievers.hybrid_searcher import HybridSearcher

print("✅ 2. 모듈 import 완료")


async def test_hybrid(query: str, category: str = None):
    """Hybrid Search 테스트 (Async)"""
    async with AsyncSessionLocal() as db:
        print("✅ 3. DB 연결 완료\n")
        try:
            # Hybrid Searcher 생성 (Milvus 실패해도 PostgreSQL로 동작)
            searcher = HybridSearcher(db, use_milvus=True, verbose=True)
            print("✅ 4. Hybrid Searcher 생성 완료\n")
            
            # 검색
            print("="*80)
            print(f"❓ 질문: '{query}'")
            if category:
                print(f"📁 카테고리: {category}")
            print("="*80)
            
            results = await searcher.search(query, category=category, limit=10)
        
            # 결과 출력
            print("\n" + "="*80)
            print(f"📊 검색 결과: {len(results)}개")
            print("="*80 + "\n")
        
            for idx, result in enumerate(results, 1):
                data = result.get("data", {})
                score = result.get("score", 0)
                match_type = result.get("match_type", "unknown")
                sources = result.get("sources", [])
                
                print(f"{idx}. [{score:.1f}점] {data.get('canonical_name', 'Unknown')}")
                print(f"   📁 카테고리: {data.get('category', 'Unknown')}")
                print(f"   🏷️  동의어: {', '.join(data.get('synonyms', []))[:50]}")
                print(f"   📝 설명: {data.get('description', '없음')[:80]}...")
                print(f"   🎯 매칭: {match_type} (출처: {', '.join(sources)})")
                print()
            
                print("="*80)
                print("✅ 5. 검색 완료!\n")
            
        except Exception as e:
            print(f"\n❌ 에러 발생: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("📚 사용법:")
        print("  python test_hybrid_search.py <검색어> [카테고리]")
        print("\n예시:")
        print('  python test_hybrid_search.py "아이스진"')
        print('  python test_hybrid_search.py "도적 되려면"')
        print('  python test_hybrid_search.py "헤네시스" MAP')
        sys.exit(1)
    
    query = sys.argv[1]
    category = sys.argv[2] if len(sys.argv) > 2 else None
    
    asyncio.run(test_hybrid(query, category))
