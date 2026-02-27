#!/usr/bin/env python3
"""Streamlit RAG 서비스 테스트"""

import sys
from pathlib import Path

# 프로젝트 루트 설정
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "streamlit_app"))
sys.path.insert(0, str(project_root / "langchain_app"))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from services.maple_rag_service import MapleRAGService
import os

def test_query():
    """커닝시티 엔피시 질문 테스트"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
    
    print(f"GROQ_API_KEY: {groq_api_key[:20]}..." if groq_api_key else "GROQ_API_KEY: None")
    print(f"GROQ_MODEL_NAME: {groq_model}")
    print("\n" + "="*80)
    
    service = MapleRAGService(
        groq_api_key=groq_api_key,
        groq_model_name=groq_model
    )
    
    query = "커닝시티 엔피시 알려줘"
    print(f"질문: {query}")
    print("="*80 + "\n")
    
    try:
        result = service.query(query, max_results=5)
        
        print(f"✅ 답변: {result['answer']}")
        print(f"\n📊 Confidence: {result['confidence']}")
        print(f"\n📚 Sources ({len(result['sources'])}개):")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source}")
        
        print(f"\n🔍 검색 결과 ({len(result['search_results'])}개):")
        for i, res in enumerate(result['search_results'][:3], 1):
            print(f"  {i}. {res.get('canonical_name', 'N/A')} - {res.get('category', 'N/A')}")
            
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_query()
