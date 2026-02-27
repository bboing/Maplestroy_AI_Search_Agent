"""
Hybrid Search - Option 3: Intent 기반 조건부 병렬 (Async)

전략:
1. RouterAgent로 Intent만 분류 (Plan X)
2. PostgreSQL + Milvus 항상 병렬
3. 관계 Intent면 Neo4j도 병렬
4. RRF 병합

장점: 정확도 높음, 비동기 최적 활용
단점: Intent 분류 필요
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import asyncio
import logging
import re

from database.models.maple_dictionary import MapleDictionary
from src.retrievers.db_searcher import MapleDBSearcher
from src.retrievers.milvus_retriever import MilvusRetriever
from src.retrievers.neo4j_searcher import Neo4jSearcher
from src.agents.router_agent import RouterAgent
from src.utils.keyword_extractor import MapleKeywordExtractor

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    Intent 기반 하이브리드 검색 + Plan Execution
    
    전략:
    1. Router Agent로 Intent 분석 & Plan 수립
       - Query의 의도 파악
       - Multi-step 검색 전략 생성
    
    2. Plan 실행 (NEW!)
       - SQL_DB → PostgreSQL (키워드 검색)
       - GRAPH_DB → Neo4j (관계 추적) [준비중]
       - VECTOR_DB → Milvus (의미 검색)
    
    3. 결과 병합 & 랭킹
       - 여러 Step의 결과를 통합
       - 점수 기반 정렬
    """
    
    def __init__(
        self, 
        db: AsyncSession,
        use_milvus: bool = True,
        use_neo4j: bool = True,
        use_router: bool = True,
        verbose: bool = False
    ):
        self.db = db
        self.use_milvus = use_milvus
        self.use_neo4j = use_neo4j
        self.use_router = use_router
        self.verbose = verbose
        
        # PostgreSQL Searcher
        self.pg_searcher = MapleDBSearcher(db)
        
        # Milvus Searcher (옵션)
        self.milvus_searcher = None
        if use_milvus:
            try:
                self.milvus_searcher = MilvusRetriever()
                logger.info("✅ Milvus 검색 활성화")
            except Exception as e:
                logger.warning(f"⚠️ Milvus 연결 실패, PostgreSQL만 사용: {e}")
                self.use_milvus = False
        
        # Neo4j Searcher (옵션)
        self.neo4j_searcher = None
        if use_neo4j:
            try:
                self.neo4j_searcher = Neo4jSearcher()
                logger.info("✅ Neo4j 검색 활성화")
            except Exception as e:
                logger.warning(f"⚠️ Neo4j 연결 실패: {e}")
                self.use_neo4j = False
        
        # Router Agent (옵션)
        self.router = None
        if use_router:
            try:
                self.router = RouterAgent(verbose=False)
                logger.info("✅ Router Agent 활성화")
            except Exception as e:
                logger.warning(f"⚠️ Router Agent 초기화 실패: {e}")
                self.use_router = False
        
        # Keyword Extractor (Kiwi 형태소 분석 + 동의어 치환)
        try:
            self.keyword_extractor = MapleKeywordExtractor(db)
            logger.info("✅ Kiwi Keyword Extractor 활성화")
        except Exception as e:
            logger.warning(f"⚠️ Kiwi 초기화 실패, 기존 방식 사용: {e}")
            self.keyword_extractor = None
    
    # 관계 필요한 Intent 정의
    RELATION_INTENTS = {
        "npc_location", "item_drop", "monster_location",
        "item_purchase", "class_change", "map_connection"
    }
    
    async def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Option 3: Intent 기반 조건부 병렬
        
        전략:
        1. Intent 분류 (가볍게)
        2. PostgreSQL + Milvus 항상 병렬
        3. 관계 Intent면 Neo4j도 병렬
        4. RRF 병합
        
        Args:
            query: 검색 쿼리
            category: 카테고리 필터 (미사용)
            limit: 최대 결과 개수
            
        Returns:
            검색 결과 리스트 (RRF 점수 순)
        """
        if self.verbose:
            print(f"\n🔍 Option 3 Search: '{query}'")
        
        # Step 1: Intent 분류 (가볍게)
        intent = await self._classify_intent(query)
        
        if self.verbose:
            print(f"   🧭 Intent: {intent}")
        
        # Step 2: PostgreSQL + Milvus 항상 병렬
        pg_task = self._postgres_search(query, None, limit)
        milvus_task = self._milvus_semantic_search(query, limit) if self.use_milvus else asyncio.sleep(0)
        
        # Step 3: 관계 Intent면 Neo4j도 병렬
        if intent in self.RELATION_INTENTS and self.use_neo4j and self.neo4j_searcher:
            neo4j_task = self._neo4j_simple_search(query, limit)
            
            if self.verbose:
                print(f"   ⚡ PostgreSQL + Milvus + Neo4j 병렬 실행...")
            
            pg_results, milvus_results, neo4j_results = await asyncio.gather(
                pg_task, milvus_task, neo4j_task
            )
        else:
            if self.verbose:
                print(f"   ⚡ PostgreSQL + Milvus 병렬 실행...")
            
            pg_results, milvus_results = await asyncio.gather(pg_task, milvus_task)
            neo4j_results = []
        
        if not self.use_milvus:
            milvus_results = []
        
        if self.verbose:
            print(f"   PostgreSQL: {len(pg_results)}개")
            print(f"   Milvus: {len(milvus_results)}개")
            print(f"   Neo4j: {len(neo4j_results)}개")
        
        # sources 필드 추가
        for r in pg_results:
            if "sources" not in r:
                r["sources"] = ["PostgreSQL"]
        for r in milvus_results:
            if "sources" not in r:
                r["sources"] = ["Milvus"]
        for r in neo4j_results:
            if "sources" not in r:
                r["sources"] = ["Neo4j"]
        
        # Step 4: RRF 병합
        results_by_source = {
            "PostgreSQL": pg_results,
            "Milvus": milvus_results,
            "Neo4j": neo4j_results
        }
        
        final_results = self._apply_rrf(results_by_source)[:limit]
        
        if self.verbose:
            print(f"   📊 최종: {len(final_results)}개\n")
        
        return final_results
    
    async def _classify_intent(self, query: str) -> str:
        """
        가벼운 Intent 분류 (Plan 생성 X)
        
        Plan 대비 10배 빠름:
        - Plan: 500 토큰 생성
        - Intent: 10 토큰 생성
        """
        if not self.router:
            return "general"
        
        try:
            # 간단한 Intent 분류 프롬프트
            prompt = f"""다음 질문의 Intent를 1단어로만 답하세요:

class_change: 전직, 직업 변경
npc_location: NPC 어디 있는지
item_drop: 아이템 드랍, 누가 떨구는지
monster_location: 몬스터 어디 있는지
item_purchase: 아이템 구매, 누가 파는지
map_connection: 맵 이동, 가는 법
general: 일반 정보

질문: {query}
Intent:"""
            
            from langchain_core.messages import HumanMessage
            response = await self.router.llm.ainvoke([HumanMessage(content=prompt)])
            intent = response.content.strip().lower().replace("_", "_")
            
            # 유효한 Intent만 반환
            valid_intents = [
                "class_change", "npc_location", "item_drop",
                "monster_location", "item_purchase", "map_connection", "general"
            ]
            
            return intent if intent in valid_intents else "general"
            
        except Exception as e:
            logger.warning(f"Intent 분류 실패: {e}")
            return "general"
    
    async def _neo4j_simple_search(
        self,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Neo4j 간단 검색 (모든 관계 유형 시도)
        """
        if not self.neo4j_searcher:
            return []
        
        keywords = await self._extract_keywords(query)
        results = []
        
        for keyword in keywords[:2]:  # 상위 2개 키워드만
            try:
                # NPC 위치
                npc_loc = await self.neo4j_searcher.find_npc_location(keyword)
                for r in npc_loc:
                    results.append({
                        "score": 85,
                        "match_type": "neo4j_npc_location",
                        "sources": ["Neo4j"],
                        "data": {"canonical_name": r.get("map_name"), "id": r.get("map_id"), "category": "MAP"}
                    })
                
                # 몬스터 위치
                mon_loc = await self.neo4j_searcher.find_monster_locations(keyword)
                for r in mon_loc:
                    results.append({
                        "score": 85,
                        "match_type": "neo4j_monster_location",
                        "sources": ["Neo4j"],
                        "data": {"canonical_name": r.get("map_name"), "id": r.get("map_id"), "category": "MAP"}
                    })
                
                # 아이템 드랍
                droppers = await self.neo4j_searcher.find_item_droppers(keyword)
                for r in droppers:
                    results.append({
                        "score": 85,
                        "match_type": "neo4j_item_drop",
                        "sources": ["Neo4j"],
                        "data": {"canonical_name": r.get("monster_name"), "id": r.get("monster_id"), "category": "MONSTER"}
                    })
                
                # 아이템 판매 NPC
                sellers = await self.neo4j_searcher.find_item_sellers(keyword)
                for r in sellers:
                    results.append({
                        "score": 85,
                        "match_type": "neo4j_item_seller",
                        "sources": ["Neo4j"],
                        "data": {"canonical_name": r.get("npc_name"), "id": r.get("npc_id"), "category": "NPC"}
                    })
                
            except Exception as e:
                logger.warning(f"Neo4j 검색 실패 ({keyword}): {e}")
        
        return results[:limit]
    
    async def execute_plan(
        self,
        original_query: str,
        router_result: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Router의 Plan을 실제로 실행 (async/await 병렬 + 순차 하이브리드)
        + RRF (Reciprocal Rank Fusion) 적용
        
        전략:
        1. Plan을 배치로 그룹화
           - SQL_DB, VECTOR_DB는 독립적 → 병렬 실행
           - GRAPH_DB는 이전 결과 필요 → 새 배치 시작
        2. 각 배치를 asyncio.gather로 병렬 실행
        3. 배치 간에는 순차 실행 (의존성 보장)
        4. RRF로 다중 소스 결과 융합
        
        Args:
            original_query: 원본 질문
            router_result: Router가 생성한 Plan
            limit: 최대 결과 개수
            
        Returns:
            검색 결과 리스트 (RRF 점수 순)
        """
        plan = router_result.get("plan", [])
        
        if not plan:
            logger.warning("Plan이 비어있음, 기본 검색으로 폴백")
            return await self._postgres_search(original_query, None, limit)
        
        if self.verbose:
            print(f"\n   📋 Plan 실행 (병렬 최적화 + RRF):")
        
        # Plan을 배치로 그룹화
        batches = self._group_plan_into_batches(plan)
        
        if self.verbose:
            print(f"      배치: {len(batches)}개 (병렬 가능한 Step끼리 그룹화)")
        
        # 소스별 결과 수집 (RRF용)
        results_by_source = {
            "PostgreSQL": [],
            "Neo4j": [],
            "Milvus": []
        }
        previous_batch_results = []
        
        # 각 배치 실행
        for batch_idx, batch in enumerate(batches):
            if self.verbose:
                print(f"\n      === 배치 {batch_idx + 1}/{len(batches)} ({'병렬' if len(batch) > 1 else '순차'}) ===")
            
            # 배치 내 Step들을 병렬 실행
            batch_results = await self._execute_batch_parallel(
                batch, 
                original_query, 
                router_result, 
                previous_batch_results
            )
            
            # 소스별로 분류
            for result in batch_results:
                sources = result.get("sources", [])
                for source in sources:
                    if source in results_by_source:
                        results_by_source[source].append(result)
            
            # 이 배치 결과를 다음 배치에 전달
            previous_batch_results = batch_results
        
        # RRF 적용
        rrf_results = self._apply_rrf(results_by_source)
        
        if self.verbose:
            print(f"\n   ✅ Plan 실행 완료: 총 {len(rrf_results)}개 결과 (RRF 적용)")
        
        return rrf_results[:limit]
    
    def _group_plan_into_batches(self, plan: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Plan을 병렬 실행 가능한 배치로 그룹화
        
        규칙:
        1. SQL_DB, VECTOR_DB는 독립적 → 같은 배치에 포함 가능
        2. GRAPH_DB는 이전 결과 필요 → 새 배치 시작
        
        Args:
            plan: Step 리스트
            
        Returns:
            배치 리스트 (각 배치는 병렬 실행 가능한 Step들)
        """
        if not plan:
            return []
        
        batches = []
        current_batch = []
        
        for step in plan:
            tool = step.get("tool", "")
            
            # GRAPH_DB는 이전 결과에 의존 → 배치 분리
            if tool == "GRAPH_DB":
                # 현재 배치가 있으면 먼저 추가
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                # GRAPH_DB는 별도 배치
                batches.append([step])
            else:
                # SQL_DB, VECTOR_DB는 같은 배치에 추가
                current_batch.append(step)
        
        # 마지막 배치 추가
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    async def _execute_batch_parallel(
        self,
        batch: List[Dict[str, Any]],
        original_query: str,
        router_result: Dict[str, Any],
        previous_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        배치 내 Step들을 병렬 실행 (async/await)
        
        Args:
            batch: 병렬 실행할 Step 리스트
            original_query: 원본 질문
            router_result: Router 결과
            previous_results: 이전 배치 결과
            
        Returns:
            배치 실행 결과
        """
        batch_results = []
        
        # 단일 Step → 그냥 실행
        if len(batch) == 1:
            step = batch[0]
            step_num = step.get("step", 0)
            tool = step.get("tool", "")
            reason = step.get("reason", "")
            
            if self.verbose:
                print(f"      [{step_num}] {tool}: {reason}")
            
            results = await self._execute_single_step(step, original_query, router_result, previous_results)
            
            if self.verbose:
                print(f"         → {len(results)}개 발견")
            
            return results
        
        # 다중 Step → async 병렬 실행
        if self.verbose:
            for step in batch:
                print(f"      [{step.get('step', 0)}] {step.get('tool', '')}: {step.get('reason', '')}")
        
        # asyncio.gather로 병렬 실행
        tasks = [
            self._execute_single_step(step, original_query, router_result, previous_results)
            for step in batch
        ]
        
        try:
            # 모든 태스크 병렬 실행
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 수집
            for idx, (step, results) in enumerate(zip(batch, results_list)):
                if isinstance(results, Exception):
                    logger.error(f"Step {step.get('step', 0)} 실행 실패: {results}")
                else:
                    batch_results.extend(results)
                    
                    if self.verbose:
                        print(f"         [{step.get('step', 0)}] → {len(results)}개 발견")
        
        except Exception as e:
            logger.error(f"배치 병렬 실행 실패: {e}")
        
        return batch_results
    
    async def _execute_single_step(
        self,
        step: Dict[str, Any],
        original_query: str,
        router_result: Dict[str, Any],
        previous_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        단일 Step 실행 (async)
        
        Args:
            step: 실행할 Step
            original_query: 원본 질문
            router_result: Router 결과
            previous_results: 이전 결과
            
        Returns:
            Step 실행 결과
        """
        tool = step.get("tool", "")
        query = step.get("query", "")
        
        try:
            if tool == "SQL_DB":
                return await self._execute_sql_db_step(original_query, query, router_result)
                
            elif tool == "GRAPH_DB":
                # 이전 결과로 쿼리 조정
                adjusted_query = self._adjust_graph_query(query, previous_results)
                results = await self._execute_graph_db_step(original_query, adjusted_query, router_result)
                # PostgreSQL로 보충
                return await self._enrich_graph_results(results)
                
            elif tool == "VECTOR_DB":
                return await self._execute_vector_db_step(original_query, query, router_result)
            else:
                logger.warning(f"알 수 없는 Tool: {tool}")
                return []
                
        except Exception as e:
            logger.error(f"Step 실행 실패: {e}")
            return []
    
    def _adjust_graph_query(
        self,
        query: str,
        previous_results: List[Dict[str, Any]]
    ) -> str:
        """
        GRAPH_DB 쿼리를 이전 Step 결과로 조정
        
        Args:
            query: 원본 쿼리 (예: "다크로드 → 위치 → MAP")
            previous_results: 이전 Step의 검색 결과
            
        Returns:
            조정된 쿼리
        """
        if not previous_results:
            return query
        
        try:
            # 이전 Step에서 찾은 첫 번째 엔티티 이름 추출
            first_result = previous_results[0]
            data = first_result.get("data", {})
            canonical_name = data.get("canonical_name")
            
            if not canonical_name:
                return query
            
            # 쿼리에서 첫 번째 단어(엔티티 이름)를 실제 찾은 이름으로 치환
            # 예: "다크로드 → 위치 → MAP"
            parts = query.split("→")
            if len(parts) >= 2:
                # 첫 번째 부분을 실제 찾은 엔티티로 교체
                parts[0] = canonical_name
                adjusted_query = " → ".join(parts)
                
                if self.verbose and adjusted_query != query:
                    print(f"         쿼리 조정: {query} → {adjusted_query}")
                
                return adjusted_query
            
            return query
            
        except Exception as e:
            logger.warning(f"쿼리 조정 실패: {e}")
            return query
    
    async def _execute_sql_db_step(
        self,
        original_query: str,
        step_query: str,
        router_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        SQL_DB Step 실행 (PostgreSQL 검색, async)
        
        핵심: 질문에서 키워드를 추출하여 검색
        ⚠️ 카테고리 필터를 사용하지 않고 전체 검색 (Router의 오판단 방지)
        """
        # 키워드 추출 (조사 제거 + 불용어 제거)
        keywords = await self._extract_keywords(original_query)
        
        if self.verbose:
            print(f"         키워드: {keywords}")
        
        results = []
        
        # 각 키워드로 검색 (순차적으로, 카테고리 필터 없이 전체 검색)
        for keyword in keywords:
            try:
                keyword_results = await self.pg_searcher.search(
                    keyword,
                    category=None,  # ✅ 카테고리 필터 제거 - 전체 검색
                    limit=5
                )
                
                # sources 필드 추가!
                for result in keyword_results:
                    if "sources" not in result:
                        result["sources"] = ["PostgreSQL"]
                
                results.extend(keyword_results)
            except Exception as e:
                logger.warning(f"키워드 '{keyword}' 검색 실패: {e}")
                continue
        
        return results
    
    async def _execute_graph_db_step(
        self,
        original_query: str,
        step_query: str,
        router_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        GRAPH_DB Step 실행 (Neo4j 관계 검색, async)
        
        step_query 분석:
        - "NPC → MAP" → NPC 위치 찾기
        - "MONSTER → MAP" → 몬스터 출현 위치
        - "ITEM → NPC" → 아이템 판매 NPC
        - "ITEM → MONSTER" → 아이템 드랍 몬스터
        - "MAP → MAP" → 맵 연결
        """
        if not self.use_neo4j or not self.neo4j_searcher:
            logger.info("Neo4j 검색 비활성화")
            return []
        
        # step_query에서 관계 유형 추출
        step_query_lower = step_query.lower()
        
        # step_query에서 엔티티 이름 추출 (예: "다크로드 → 위치 → MAP")
        entity_name = None
        if "→" in step_query:
            parts = step_query.split("→")
            entity_name = parts[0].strip()
            
            # 카테고리 접두사 제거 (MAP, MONSTER, NPC, ITEM 등)
            category_prefixes = ["MAP ", "MONSTER ", "NPC ", "ITEM "]
            for prefix in category_prefixes:
                if entity_name.startswith(prefix):
                    entity_name = entity_name[len(prefix):].strip()
                    break
        
        # 키워드로 엔티티 이름 추출 (fallback)
        keywords = await self._extract_keywords(original_query)
        
        # step_query에서 추출한 엔티티가 있으면 우선 사용 (중복 제거)
        if entity_name:
            if entity_name not in keywords:
                keywords = [entity_name] + keywords
            else:
                # 이미 있으면 앞으로 이동
                keywords.remove(entity_name)
                keywords = [entity_name] + keywords
        
        if self.verbose:
            print(f"         GRAPH_DB step_query: {step_query}")
            print(f"         Keywords: {keywords}")
        
        results = []
        
        # NPC 관련 검색
        if "npc" in step_query_lower and "map" in step_query_lower:
            # Case 1: NPC 위치 검색 ("NPC가 어디에 있는지")
            if any(word in step_query_lower for word in ["위치", "어디", "있는지"]) and \
               any(word in original_query for word in ["어디", "위치"]):
                for keyword in keywords:
                    npc_results = await self.neo4j_searcher.find_npc_location(keyword)
                    results.extend(self._format_graph_results(npc_results, "graph_npc_location"))
            
            # Case 2: MAP → NPC ("맵에 어떤 NPC가 있는지")
            else:
                # PostgreSQL에서 MAP 검색 후 resident_npcs 활용
                for keyword in keywords:
                    if keyword not in ["MAP", "NPC", "MONSTER", "ITEM"] and len(keyword) >= 2:
                        try:
                            pg_results = await self.pg_searcher.search(keyword, category="MAP", limit=3)
                            # sources 필드 추가
                            for result in pg_results:
                                if "sources" not in result:
                                    result["sources"] = ["PostgreSQL"]
                            results.extend(pg_results)
                        except Exception as e:
                            logger.warning(f"MAP NPC 검색 실패 ({keyword}): {e}")
        
        # 몬스터 위치 검색
        elif "monster" in step_query_lower and "map" in step_query_lower:
            for keyword in keywords:
                monster_results = await self.neo4j_searcher.find_monster_locations(keyword)
                results.extend(self._format_graph_results(monster_results, "graph_monster_location"))
        
        # 아이템 판매 NPC 검색
        elif "item" in step_query_lower and "npc" in step_query_lower:
            # "판매" 또는 "sell" 키워드
            if any(word in step_query_lower for word in ["판매", "sell", "구매", "buy", "사"]):
                for keyword in keywords:
                    seller_results = await self.neo4j_searcher.find_item_sellers(keyword)
                    results.extend(self._format_graph_results(seller_results, "graph_item_seller"))
        
        # 아이템 드랍 몬스터 검색
        elif any(word in step_query_lower for word in ["드랍", "drop", "떨어", "나와", "나오"]):
            # "드랍", "몬스터" 키워드가 있으면 드랍 검색
            if "몬스터" in step_query_lower or "monster" in step_query_lower:
                for keyword in keywords:
                    dropper_results = await self.neo4j_searcher.find_item_droppers(keyword)
                    results.extend(self._format_graph_results(dropper_results, "graph_item_dropper"))
        
        # 맵 연결 검색
        elif "map" in step_query_lower and any(word in step_query_lower for word in ["경로", "connect", "이동", "가는"]):
            for keyword in keywords:
                map_results = await self.neo4j_searcher.find_map_connections(keyword)
                results.extend(self._format_graph_results(map_results, "graph_map_connection"))
        
        # ✅ MAP 검색 (resident_npcs, resident_monsters는 PostgreSQL에 있음)
        # "MAP → 커닝시티", "커닝시티에 어떤 NPC" 등
        else:
            # keywords에 맵 이름이 있으면 PostgreSQL 직접 검색
            for keyword in keywords:
                if keyword not in ["MAP", "NPC", "MONSTER", "ITEM"] and len(keyword) >= 2:
                    try:
                        # MAP 우선 검색
                        if "map" in step_query_lower or "맵" in step_query_lower or \
                           any(word in original_query for word in ["어떤", "있어", "주민"]):
                            pg_results = await self.pg_searcher.search(keyword, category="MAP", limit=3)
                            # sources 필드 추가
                            for result in pg_results:
                                if "sources" not in result:
                                    result["sources"] = ["PostgreSQL"]
                            results.extend(pg_results)
                        
                        # 결과 없으면 전체 카테고리 검색
                        if not results:
                            pg_results = await self.pg_searcher.search(keyword, category=None, limit=3)
                            # sources 필드 추가
                            for result in pg_results:
                                if "sources" not in result:
                                    result["sources"] = ["PostgreSQL"]
                            results.extend(pg_results)
                            
                    except Exception as e:
                        logger.warning(f"검색 실패 ({keyword}): {e}")
        
        return results
    
    def _format_graph_results(
        self,
        graph_results: List[Dict[str, Any]],
        match_type: str
    ) -> List[Dict[str, Any]]:
        """
        Neo4j 결과를 통합 포맷으로 변환
        """
        formatted_results = []
        
        for result in graph_results:
            # Neo4j 결과에서 이름 추출 (다양한 필드명 지원)
            name = (
                result.get("npc_name") or 
                result.get("map_name") or 
                result.get("monster_name") or 
                result.get("item_name") or 
                result.get("name", "Unknown")
            )
            
            # ID 추출
            entity_id = (
                result.get("npc_id") or 
                result.get("map_id") or 
                result.get("monster_id") or 
                result.get("item_id") or 
                result.get("id", "")
            )
            
            formatted_results.append({
                "score": 85.0,  # Graph 관계는 높은 신뢰도
                "match_type": match_type,
                "sources": ["Neo4j"],
                "data": {
                    "id": entity_id,
                    "canonical_name": name,
                    "category": "RELATION",  # 관계 검색 결과
                    "description": f"{result.get('relation_type', '')} 관계",
                    "relation_info": result  # 원본 관계 정보 보존
                }
            })
        
        return formatted_results
    
    async def _enrich_graph_results(self, graph_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Neo4j 검색 결과를 PostgreSQL detail_data로 보충 (async)
        
        Neo4j는 관계만 알려주고, 실제 상세 정보는 PostgreSQL에 있음
        예: Neo4j → "스포아"라는 이름만
            PostgreSQL → 스포아의 level, spawn_maps, drops 등 상세 정보
        
        Args:
            graph_results: Neo4j 검색 결과
            
        Returns:
            detail_data가 추가된 결과
        """
        enriched = []
        
        for result in graph_results:
            data = result.get("data", {})
            canonical_name = data.get("canonical_name")
            
            if not canonical_name or canonical_name == "Unknown":
                enriched.append(result)
                continue
            
            try:
                # PostgreSQL에서 전체 정보 조회 (async)
                stmt = select(MapleDictionary).where(
                    MapleDictionary.canonical_name == canonical_name
                )
                db_result = await self.db.execute(stmt)
                pg_entity = db_result.scalar_one_or_none()
                
                if pg_entity:
                    # PostgreSQL 데이터로 교체 (관계 정보는 유지)
                    relation_info = data.get("relation_info")
                    result["data"] = pg_entity.to_dict()
                    
                    # 관계 정보 추가
                    if relation_info:
                        result["data"]["relation_info"] = relation_info
                    
                    # Category 업데이트
                    result["data"]["category"] = str(pg_entity.category).split('.')[-1]
                    
                enriched.append(result)
                
            except Exception as e:
                logger.warning(f"Enrichment 실패 ({canonical_name}): {e}")
                enriched.append(result)
        
        return enriched
    
    async def _execute_vector_db_step(
        self,
        original_query: str,
        step_query: str,
        router_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        VECTOR_DB Step 실행 (Milvus 의미 검색, async)
        """
        if not self.use_milvus or not self.milvus_searcher:
            return []
        
        try:
            # 원본 질문으로 의미 검색
            results = await self.milvus_searcher.search(original_query, top_k=10)
            
            # 결과 포맷팅
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "score": result.get("score", 0) * 100,
                    "match_type": "vector_db",
                    "sources": ["Milvus"],  # 출처 추가!
                    "data": result
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"VECTOR_DB 검색 실패: {e}")
            return []
    
    async def _extract_keywords(self, query: str) -> List[str]:
        """
        질문에서 검색 키워드 추출 (Async)
        
        전략:
        1. Kiwi 형태소 분석 (우선, 정확도 높음)
        2. Fallback: 정규식 + 불용어 (Kiwi 실패 시)
        """
        # Kiwi 사용 가능하면 우선 사용
        if self.keyword_extractor:
            try:
                return await self.keyword_extractor.extract(query)
            except Exception as e:
                logger.warning(f"Kiwi 키워드 추출 실패, Fallback 사용: {e}")
        
        # Fallback: 기존 정규식 방식
        # 조사 (토큰 끝에서 제거)
        particles = ["이", "가", "을", "를", "은", "는", "도", "만", "의", "에", "서", "로", "와", "과"]
        
        # 접미사 형태 불용어 (단어 끝에 붙는 것들)
        suffix_stopwords = [
            "하려면", "해야", "가야", "가려면", "해야해", "가야해", "하면", "되면",
            "하고", "싶어", "싶다", "싶으면", "되고", "원해", "원하다",
            "하는", "있는", "없는", "가는", "오는", "나오는",
            "어요", "아요", "해요", "습니다", "ㅂ니다",
            "나요", "하나요", "할까요", "있나요", "없나요"
        ]
        
        # 완전 일치 불용어
        stopwords = {
            # 의문사
            "어디", "어디서", "어디로", "어디에", "어떻게", "어떤", "뭐", "무엇", "누구",
            "언제", "왜", "얼마", "몇", "무슨",
            # 동사/형용사 (단독)
            "하다", "있다", "없다", "가다", "오다", "되다", "보다", "주다", "받다", "잡다",
            "가야", "와야", "해야", "있어야",  # 추가!
            # 어미/조사
            "해", "요", "나", "나요", "하나요",
            # 일반어
            "것", "수", "때", "곳", "중", "등", "이런", "저런", "그런", "좀", "더"
        }
        
        # 한글, 영문, 숫자만 추출
        tokens = re.findall(r'[가-힣A-Za-z0-9]+', query)
        
        # 1. 조사 제거
        cleaned_tokens = []
        for token in tokens:
            # 3글자 이상이고 마지막 글자가 조사면 제거
            if len(token) >= 3 and token[-1] in particles:
                cleaned_tokens.append(token[:-1])  # 마지막 글자 제거
            else:
                cleaned_tokens.append(token)
        
        # 2. 불용어 제거 (완전 일치 + 접미사)
        keywords = []
        for token in cleaned_tokens:
            # 완전 일치 체크
            if token in stopwords:
                continue
            
            # 접미사 체크 (예: "전직하려면" → "하려면" 접미사 제거)
            is_stopword = False
            for suffix in suffix_stopwords:
                if len(token) > len(suffix) and token.endswith(suffix):
                    # 접미사 제거 후 남은 부분만 추출
                    core_word = token[:-len(suffix)]
                    if len(core_word) >= 2:  # 핵심 단어가 2글자 이상
                        keywords.append(core_word)
                    is_stopword = True
                    break
            
            # 불용어 아니고 2글자 이상이면 추가
            if not is_stopword and len(token) >= 2:
                keywords.append(token)
        
        # 3. 숫자만 있는 토큰 제외
        keywords = [k for k in keywords if not k.isdigit()]
        
        # 4. 중복 제거
        keywords = list(dict.fromkeys(keywords))
        
        # 5. 비어있으면 원본 쿼리 사용
        if not keywords:
            keywords = [query]
        
        return keywords
    
    def _apply_rrf(
        self,
        results_by_source: Dict[str, List[Dict[str, Any]]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        RRF (Reciprocal Rank Fusion) 적용
        
        여러 검색 소스의 결과를 랭크 기반으로 융합
        
        공식: RRF_score(d) = Σ 1 / (k + rank_i(d))
        
        Args:
            results_by_source: 소스별 결과
                {
                    "PostgreSQL": [...],
                    "Neo4j": [...],
                    "Milvus": [...]
                }
            k: RRF 상수 (기본 60)
            
        Returns:
            RRF 점수로 정렬된 결과
        """
        rrf_scores = {}  # entity_id -> RRF score
        entity_data = {}  # entity_id -> 실제 데이터
        
        # 각 소스별로 순위 기반 점수 계산
        for source, results in results_by_source.items():
            if not results:
                continue
            
            # 결과를 점수 순으로 정렬 (각 소스 내에서)
            sorted_results = sorted(
                results,
                key=lambda x: x.get("score", 0),
                reverse=True
            )
            
            # 순위 기반 RRF 점수 계산
            for rank, result in enumerate(sorted_results):
                data = result.get("data", {})
                entity_id = str(data.get("id", ""))
                
                if not entity_id:
                    continue
                
                # RRF 점수: 1 / (k + rank)
                rrf_score = 1.0 / (k + rank)
                
                # 누적
                if entity_id in rrf_scores:
                    rrf_scores[entity_id] += rrf_score
                else:
                    rrf_scores[entity_id] = rrf_score
                    entity_data[entity_id] = result
        
        # RRF 점수로 정렬
        sorted_entities = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 최종 결과 생성
        final_results = []
        for entity_id, rrf_score in sorted_entities:
            result = entity_data[entity_id]
            
            # RRF 점수를 0-100 스케일로 변환
            # (최대 RRF 점수를 100으로 정규화)
            max_rrf = sorted_entities[0][1] if sorted_entities else 1.0
            normalized_score = (rrf_score / max_rrf) * 100
            
            final_results.append({
                "score": normalized_score,
                "match_type": result.get("match_type", "rrf"),
                "data": result.get("data"),
                "sources": result.get("sources", []),
                "rrf_score": rrf_score  # 원본 RRF 점수도 보존
            })
        
        return final_results
    
    async def _postgres_search(
        self,
        query: str,
        category: Optional[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        """PostgreSQL 검색 (빠른 정확 매칭, async)"""
        try:
            results = await self.pg_searcher.search(query, category=category, limit=limit)
            return results
        except Exception as e:
            logger.error(f"PostgreSQL 검색 실패: {e}")
            return []
    
    async def _milvus_expansion_search(
        self,
        pg_results: List[Dict[str, Any]],
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Milvus 연관 확장 검색 (async)
        PostgreSQL에서 찾은 엔티티들의 연관 항목 검색
        """
        if not self.milvus_searcher:
            return []
        
        milvus_results = []
        seen_ids = set()
        
        # PostgreSQL에서 찾은 TOP 3 엔티티로 확장
        for pg_item in pg_results[:3]:
            data = pg_item.get("data", {})
            canonical_name = data.get("canonical_name", "")
            item_id = data.get("id")
            
            if item_id:
                seen_ids.add(str(item_id))
            
            if not canonical_name:
                continue
            
            try:
                # canonical_name으로 Milvus 검색
                results = await self.milvus_searcher.search(canonical_name, top_k=5)
                
                # 결과 추가
                for result in results:
                    result_id = result.get("id")
                    
                    if result_id and result_id not in seen_ids:
                        milvus_results.append({
                            "score": result.get("score", 0) * 50,  # 점수 조정
                            "match_type": "milvus_expansion",
                            "data": result,
                            "source_entity": canonical_name
                        })
                        seen_ids.add(result_id)
                        
                        if len(milvus_results) >= limit:
                            break
                
            except Exception as e:
                logger.warning(f"Milvus 확장 검색 실패 ({canonical_name}): {e}")
                continue
            
            if len(milvus_results) >= limit:
                break
        
        return milvus_results
    
    async def _milvus_semantic_search(
        self,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Milvus 의미 검색 (폴백, async)
        질문 전체를 의미적으로 검색
        """
        if not self.milvus_searcher:
            return []
        
        try:
            # Milvus Q&A 검색
            results = await self.milvus_searcher.search(query, top_k=limit)
            
            # 결과 포맷팅
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "score": result.get("score", 0) * 100,  # 점수 조정
                    "match_type": "milvus_semantic",
                    "sources": ["Milvus"],  # sources 필드 추가
                    "data": result
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Milvus 의미 검색 실패: {e}")
            return []
    
    def _merge_results(
        self,
        pg_results: List[Dict[str, Any]],
        milvus_results: List[Dict[str, Any]],
        mode: str = "expansion"
    ) -> List[Dict[str, Any]]:
        """
        PostgreSQL + Milvus 결과 병합 (RRF 적용)
        
        Args:
            mode: "expansion" (확장) 또는 "fallback" (폴백)
        """
        # RRF 적용
        results_by_source = {
            "PostgreSQL": pg_results,
            "Milvus": milvus_results,
            "Neo4j": []  # 이 메서드에서는 Neo4j 없음
        }
        
        return self._apply_rrf(results_by_source)


# 편의 함수
async def hybrid_search(
    db: AsyncSession,
    query: str,
    category: Optional[str] = None,
    limit: int = 10,
    use_milvus: bool = True
) -> List[Dict[str, Any]]:
    """
    간단한 하이브리드 검색 함수 (async)
    
    Usage:
        results = await hybrid_search(db, "아이스진 어디서 사나요?")
    """
    searcher = HybridSearcher(db, use_milvus=use_milvus)
    return await searcher.search(query, category=category, limit=limit)
