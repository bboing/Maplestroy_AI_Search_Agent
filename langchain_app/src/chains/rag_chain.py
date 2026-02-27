"""
RAG Chain - 검색(Retriever) + 생성(Generator) 파이프라인 오케스트레이터

역할:
  API 레이어(main.py)와 컴포넌트(Searcher, Generator) 사이를 연결한다.
  main.py는 체인만 호출하면 되고, 내부 조합 로직은 여기서 관리한다.

파이프라인:
  query → HybridSearcherFin(검색) → AnswerGenerator(생성) → result
"""
from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from src.retrievers.hybrid_searcher_fin import HybridSearcher as HybridSearcherFin
from src.generators.answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)


class MapleRAGChain:
    """
    메이플스토리 RAG 파이프라인

    컴포넌트:
    - HybridSearcherFin : hop 기반 하이브리드 검색 (PG + Milvus + Neo4j)
    - AnswerGenerator   : 검색 결과 → LLM 자연어 답변 생성

    사용법:
        chain = MapleRAGChain(db=db)
        result = await chain.ainvoke(query="도적 전직 어디?")
    """

    def __init__(
        self,
        db: AsyncSession,
        use_milvus: bool = True,
        use_neo4j: bool = True,
        verbose: bool = False,
    ):
        self.searcher = HybridSearcherFin(
            db=db,
            use_milvus=use_milvus,
            use_neo4j=use_neo4j,
            use_router=True,
            verbose=verbose,
        )
        self.generator = AnswerGenerator(verbose=verbose)

    async def ainvoke(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
        use_plan_execution: bool = True,
        max_context_items: int = 5,
    ) -> Dict[str, Any]:
        """
        RAG 파이프라인 실행 (async)

        Args:
            query             : 사용자 질문
            category          : 검색 카테고리 필터 (선택)
            limit             : 검색 결과 최대 개수
            use_plan_execution: 플랜 기반 실행 여부
            max_context_items : 답변 생성에 사용할 컨텍스트 수

        Returns:
            {
                "answer"        : str,
                "sources"       : List[str],
                "confidence"    : float,
                "search_results": List[dict]   # 상위 5개
            }
        """
        # 1. 하이브리드 검색
        search_results: List[Dict[str, Any]] = await self.searcher.search(
            query=query,
            category=category,
            limit=limit,
            use_plan_execution=use_plan_execution,
        )

        # 2. 답변 생성
        generated = await self.generator.generate(
            query=query,
            search_results=search_results,
            max_context_items=max_context_items,
        )

        return {
            "answer": generated["answer"],
            "sources": generated["sources"],
            "confidence": generated["confidence"],
            "search_results": search_results[:5],
        }
