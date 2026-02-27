"""
Maple RAG Service
기존 langchain_app의 RAG 로직을 Streamlit용으로 래핑
"""
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any
from langchain_groq import ChatGroq

# langchain_app 경로 추가
LANGCHAIN_DIR = Path(__file__).parent.parent.parent / "langchain_app"
sys.path.insert(0, str(LANGCHAIN_DIR))

from database.session import get_async_db
from src.retrievers.hybrid_searcher_fin import HybridSearcher
from src.generators.answer_generator import AnswerGenerator


class MapleRAGService:
    """
    Streamlit용 RAG 서비스
    
    기존 langchain_app의 비동기 로직을 동기적으로 래핑
    """
    
    def __init__(self):
        """
        RouterAgent와 AnswerGenerator가 자동으로 Ollama/Groq를 선택합니다.
        환경변수(OLLAMA_BASE_URL, GROQ_API_KEY)를 통해 제어됩니다.
        """
        pass
    
    def query(self, question: str, max_results: int = 5) -> Dict[str, Any]:
        """
        질문에 답변 (동기)
        
        Args:
            question: 사용자 질문
            max_results: 최대 검색 결과 수
            
        Returns:
            {
                "answer": str,
                "sources": List[str],
                "confidence": float,
                "search_results": List[Dict]
            }
        """
        # 비동기 로직을 동기로 실행
        try:
            # 이미 실행 중인 이벤트 루프가 있으면 사용
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Streamlit 등에서 이미 이벤트 루프가 실행 중인 경우
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.run(self._async_query(question, max_results))
            else:
                return loop.run_until_complete(self._async_query(question, max_results))
        except RuntimeError:
            # 이벤트 루프가 없으면 새로 생성
            return asyncio.run(self._async_query(question, max_results))
    
    async def _async_query(self, question: str, max_results: int) -> Dict[str, Any]:
        """
        실제 비동기 쿼리 로직
        """
        # DB 세션 생성
        async for session in get_async_db():
            # 1. Hybrid Search
            # Router Agent가 자동으로 Ollama/Groq 선택 (health check)
            searcher = HybridSearcher(
                db=session,
                use_milvus=True,
                use_neo4j=True,
                use_router=True,  # ✅ Router Agent 활성화 (Ollama/Groq 자동 전환)
                verbose=False
            )
            
            search_results = await searcher.search(
                query=question,
                limit=max_results
            )
            
            # 2. Answer Generation (LLM 자동 선택)
            generator = AnswerGenerator(
                llm=None,  # ✅ None → 자동으로 Ollama/Groq health check
                verbose=False
            )
            
            answer_result = await generator.generate(
                query=question,
                search_results=search_results,
                max_context_items=max_results
            )
            
            # 3. 결과 반환
            router_info = getattr(searcher, "last_router_result", {})
            return {
                "answer": answer_result["answer"],
                "sources": answer_result["sources"],
                "confidence": answer_result["confidence"],
                "search_results": search_results,
                "entities": router_info.get("entities", []),
                "sentences": router_info.get("sentences", []),
            }
