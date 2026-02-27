#!/usr/bin/env python3
"""Kiwi 키워드 추출 테스트"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LANGCHAIN_APP_DIR = PROJECT_ROOT / "langchain_app"
sys.path.insert(0, str(LANGCHAIN_APP_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from database.session import SessionLocal
from src.utils.keyword_extractor import MapleKeywordExtractor


def main():
    print("\n" + "="*80)
    print("🔍 Kiwi 키워드 추출 테스트")
    print("="*80 + "\n")
    
    db = SessionLocal()
    
    try:
        # Extractor 초기화
        extractor = MapleKeywordExtractor(db)
        
        # 테스트 쿼리
        test_queries = [
            "전사로 전직하려면 어디로 가야하나요?",
            "도적이 되고 싶으면 어디로 가야 하나요?",
            "아진 사려면 어디로?",
            "스포아잡으려면어디가야해?",
            "다크로드 위치 알려줘",
            "헤네시스에서 엘리니아 가는법",
        ]
        
        for query in test_queries:
            keywords = extractor.extract(query)
            
            print(f"📝 질문: {query}")
            print(f"   → 키워드: {keywords}")
            print()
        
        print("="*80)
        print("✅ 테스트 완료!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    main()
