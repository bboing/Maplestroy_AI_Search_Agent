#!/usr/bin/env python3
"""
Answer Generator 통합 테스트 (Async)
"""
import sys
import asyncio
from pathlib import Path

# 경로 설정
SCRIPT_DIR = Path(__file__).resolve().parent
print(SCRIPT_DIR)
PROJECT_ROOT = SCRIPT_DIR.parent
print(PROJECT_ROOT)
LANGCHAIN_APP_DIR = PROJECT_ROOT / "langchain_app"
sys.path.insert(0, str(LANGCHAIN_APP_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from database.session import AsyncSessionLocal
from src.retrievers.hybrid_searcher import HybridSearcher
from src.generators.answer_generator import AnswerGenerator
from src.agents.router_agent import RouterAgent


async def test_end_to_end(query: str):
    """전체 파이프라인 테스트 (Async)"""
    print("\n" + "="*80)
    print(f"🧪 End-to-End 테스트: '{query}'")
    print("="*80)
    
    async with AsyncSessionLocal() as db:
        try:
            # 1. Router로 Intent 분석
            print("\n[1단계] Intent 분석")
            router = RouterAgent()
            route_result = router.route(query)
            
            print(f"   🧭 Intent: {route_result['intent']}")
            print(f"   📁 Categories: {route_result['categories']}")
            print(f"   🔍 Strategy: {route_result['strategy']}")
            print(f"   🔑 Keywords: {route_result['keywords']}")
        
            # 2. Hybrid Search
            print("\n[2단계] Hybrid Search")
            searcher = HybridSearcher(db, verbose=True)
            search_results = await searcher.search(
                query=query,
                limit=5  # Router가 자동으로 category와 strategy 처리
            )
            print(f"\n   📊 검색 결과: {len(search_results)}개")
            for i, result in enumerate(search_results[:3], 1):
                data = result.get('data', {})
                canonical_name = data.get('canonical_name', 'Unknown')
                description = data.get('description', '')
                category = data.get('category', '')
                print(f"   {i}. [{result['score']:.1f}점] {canonical_name} ({category})")
                if description:
                    print(f"      {description[:60]}...")
            
            # 3. Answer Generation
            print("\n[3단계] Answer Generation")
            generator = AnswerGenerator(verbose=True)
            answer = await generator.generate(
                query=query,
                search_results=search_results,
                max_context_items=5
            )
            
            # 4. 결과 출력
            print("\n" + "="*80)
            print("✅ 최종 답변")
            print("="*80)
            print(f"\n💬 {answer['answer']}\n")
            print(f"📚 출처: {', '.join(answer['sources'][:3])}")
            print(f"🎯 신뢰도: {answer['confidence']:.1f}%")
            print("="*80 + "\n")
        
        except asyncio.TimeoutError:
            print("서버 응답이 너무 느림")
        
        except ConnectionError:
            print("DB 연결에 실패함")

        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()


async def run_test_suite():
    """테스트 스위트 (Async)"""
    test_queries = [
        "도적으로 전직하려면 어디로 가야 하나요?",
        "전사가 되기 위해서는 누구를 만나야 하나요?",
        "아이스진은 어디서 구할 수 있어?",
        "스포아는 어디서 잡을 수 있나요?",
        "커닝시티에는 어떤 NPC가 있어?"
    ]
    
    print("\n" + "🔬"*40)
    print("Answer Generator 통합 테스트 스위트 (Async)")
    print("🔬"*40 + "\n")
    
    for query in test_queries:
        await test_end_to_end(query)
        print("\n" + "-"*80 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 단일 쿼리 테스트
        query = " ".join(sys.argv[1:])
        asyncio.run(test_end_to_end(query))
    else:
        # 전체 테스트 스위트
        asyncio.run(run_test_suite())
