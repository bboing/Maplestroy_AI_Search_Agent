"""
Hybrid Search with HOP-based Routing (Async) - HOP STRATEGY
Router Agent → HOP 깊이 판단 → 선택적 DB 사용
+ HOP-1: Postgres + Milvus (직접 관계)
+ HOP-2+: Postgres + Milvus + Neo4j (체인 관계)
+ Entity/Sentence 분리 → DB별 최적 쿼리
+ Synonym Resolution → PostgreSQL 간접 매칭
+ Jina Reranker → RRF 후 노이즈 제거
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import asyncio
import logging
import re
import requests
import os

from database.models.maple_dictionary import MapleDictionary
from src.retrievers.db_searcher import MapleDBSearcher
from src.retrievers.milvus_retriever import MilvusRetriever
from src.retrievers.neo4j_searcher import Neo4jSearcher
from src.agents.router_agent_hop import RouterAgent
from src.utils.keyword_extractor import MapleKeywordExtractor
from config.settings import settings

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    ✅ HOP-based Hybrid Search (관계 깊이 기반 검색)
    
    핵심 아이디어:
    1. 관계 깊이(Hop)에 따라 DB 선택
       - hop=1: Postgres + Milvus (직접 관계)
       - hop=2+: Postgres + Milvus + Neo4j (체인 관계)
    
    2. Entity/Sentence 분리
       - Entity(명사) → Postgres (canonical_name + synonym)
       - Sentence(동사구) → Milvus (의미 검색)
    
    3. 무조건 병렬 실행
       - Postgres + Milvus 항상 병렬
       - hop >= 2면 Neo4j 추가
    
    검색 흐름:
    1. Router → hop, entities, sentences 추출
    2. Postgres(entities) + Milvus(sentences) 병렬 실행
    3. hop >= 2 → Neo4j 관계 검색 추가
    4. RRF 병합
    5. Reranker 재정렬
    6. 상위 N개 반환
    
    예시 (hop=1):
    - 질문: "물약 파는 사람 누구야?"
    - hop: 1 (ITEM-NPC 직접 관계)
    - entities: []
    - sentences: ["물약 파는 사람"]
    - 실행: Milvus만 → "미나" 발견
    
    예시 (hop=2):
    - 질문: "아이스진 얻으려면 어떻게?"
    - hop: 2 (ITEM-MONSTER-MAP 체인)
    - entities: ["아이스진"]
    - sentences: []
    - 실행: Postgres + Neo4j → "스포아" → "폐광" 발견
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
    
    async def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
        pg_threshold: int = 3,
        use_plan_execution: bool = True
    ) -> List[Dict[str, Any]]:
        """
        ✅ HOP 기반 하이브리드 검색
        
        전략:
        1. Router → hop, entities, sentences 추출
        2. Postgres (entities) + Milvus (sentences) 병렬 실행
        3. hop >= 2면 Neo4j 추가
        4. RRF 병합
        5. Reranker 재정렬
        
        Args:
            query: 검색 쿼리
            limit: 최대 결과 개수
            
        Returns:
            검색 결과 리스트 (점수 순 정렬)
        """
        print("hybrid_searcher_hop.search 호출 됨")
        if self.verbose:
            print(f"\n🔍 Hybrid Search (HOP): '{query}'")
        
        # Step 1: Router로 hop, entities, sentences 추출
        router_result = None
        hop = 1  # 기본값
        entities = []
        sentences = []
        
        if self.use_router and self.router:
            try:
                router_result = await self.router.route(query)
                hop = router_result.get("hop", 1)
                entities = router_result.get("entities", [])
                print(f"entities: {entities}")
                sentences = router_result.get("sentences", [])
                print(f"sentences: {sentences}")

                if self.verbose:
                    print(f"   🧭 Hop: {hop}")
                    print(f"   📌 Entities: {entities}")
                    print(f"   📝 Sentences: {sentences}")
                    
            except Exception as e:
                logger.warning(f"Router 실패, 자체 키워드 추출: {e}")
                # Fallback: 자체 키워드 추출
                raw_keywords = await self._extract_keywords(query)
                structured = self._reconstruct_ngrams(raw_keywords, query)
                entities = structured["entities"]
                sentences = structured["sentences"]
        
        # Step 2: Postgres + Milvus 병렬 실행 (무조건)
        results_by_source = {
            "PostgreSQL": [],
            "Milvus": [],
            "Neo4j": []
        }
        
        # 병렬 실행
        async def empty(): return []            # sentence 값 없으면 [] return / 코루틴 객체 반환해야 함. 리스트 객체면 awaitable이 아님. asyncti.gather는 약속된

        pg_task = self._search_postgres_with_synonym(entities, limit_per_entity=5) if entities else empty()
        milvus_task = self._search_milvus_sentences(sentences) if sentences and self.use_milvus else empty()

        pg_results, milvus_results = await asyncio.gather(pg_task, milvus_task)
        
        # sources 필드 추가
        if isinstance(pg_results, list):
            for result in pg_results:
                if "sources" not in result:
                    result["sources"] = ["PostgreSQL"]
            results_by_source["PostgreSQL"] = pg_results
        
        if isinstance(milvus_results, list):
            for result in milvus_results:
                if "sources" not in result:
                    result["sources"] = ["Milvus"]
            results_by_source["Milvus"] = milvus_results
        
        if self.verbose:
            print(f"   PostgreSQL: {len(pg_results) if isinstance(pg_results, list) else 0}개")
            print(f"   Milvus: {len(milvus_results) if isinstance(milvus_results, list) else 0}개")
        
        # Step 3: hop >= 2면 Neo4j 추가
        if hop >= 2 and self.use_neo4j and self.neo4j_searcher:
            if self.verbose:
                print(f"   🔗 Hop={hop} → Neo4j 관계 검색")
            
            neo4j_results = await self._search_neo4j_relations(query, entities, router_result)
            results_by_source["Neo4j"] = neo4j_results
            
            if self.verbose:
                print(f"   Neo4j: {len(neo4j_results)}개")
        
        # Step 4: RRF 병합
        rrf_results = self._apply_rrf(results_by_source)
        
        if self.verbose:
            print(f"   ✅ RRF 완료: {len(rrf_results)}개")
        
        # Step 5: Reranker (결과 > limit일 때)
        if len(rrf_results) > limit:
            rrf_results = await self._rerank_with_jina(query, rrf_results, top_n=limit)
            
            if self.verbose:
                print(f"   ✅ Reranker 완료: {len(rrf_results)}개")
        
        if self.verbose:
            print(f"   📊 최종: {len(rrf_results[:limit])}개\n")
        
        return rrf_results[:limit]
    
    
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
    
    def _reconstruct_ngrams(
        self,
        raw_keywords: List[str],
        original_query: str
    ) -> Dict[str, List[str]]:
        """
        ✅ N-gram 재구성: Entity vs Sentence 분류
        
        전략:
        1. 단일 명사 (NNG, NNP) → Entity (PostgreSQL 검색용)
        2. 연속된 단어 (동사 포함) → Sentence (Milvus 의미 검색용)
        
        예시:
        - Input: ['리스항구', '물약', '파는', '사람']
        - Output: {
            "entities": ['리스항구'],           # PG용
            "sentences": ['물약 파는 사람']     # Milvus용
          }
        
        Args:
            raw_keywords: LLM/Kiwi가 추출한 원본 키워드
            original_query: 사용자 원본 질문
            
        Returns:
            Entity와 Sentence로 분류된 딕셔너리
        """
        entities = []
        sentences = []
        
        # ✅ 휴리스틱 기반 분류 (Kiwi 없어도 작동)
        # 동사 패턴 리스트 (간접 표현)
        verb_patterns = [
            '파는', '사는', '팔', '살', '파', '사',
            '주는', '드는', '떨어', '나오',
            '있는', '가는', '하는', '되는',
            '할', '될', '갈', '올'
        ]
        
        # 연속된 키워드로 원문에 있는 구문 찾기 (N-gram)
        i = 0
        used_indices = set()  # 이미 사용한 키워드 인덱스
        
        while i < len(raw_keywords):
            if i in used_indices:
                i += 1
                continue
            
            # 동사 패턴이 있으면 Sentence 후보
            current_has_verb = any(verb in raw_keywords[i] for verb in verb_patterns)
            
            if current_has_verb or (i > 0 and any(verb in raw_keywords[i-1] for verb in verb_patterns)):
                # 2~4개 단어 조합해서 원문에 있는지 확인
                found_sentence = False
                for n in range(min(4, len(raw_keywords) - i), 1, -1):  # 4, 3, 2 순서
                    phrase = ' '.join(raw_keywords[i:i+n])
                    
                    # 원문에 이 구문이 있는지 확인
                    if phrase in original_query and n >= 2:  # 2개 이상만 Sentence
                        sentences.append(phrase)
                        for j in range(i, i+n):
                            used_indices.add(j)
                        i += n
                        found_sentence = True
                        break
                
                if not found_sentence:
                    # Sentence 못 만들면 Entity로
                    if not current_has_verb:  # 동사 단독은 버림
                        entities.append(raw_keywords[i])
                    used_indices.add(i)
                    i += 1
            else:
                # 동사 없으면 Entity
                entities.append(raw_keywords[i])
                used_indices.add(i)
                i += 1
        
        # Kiwi 있으면 추가 정제 (옵션)
        if self.keyword_extractor and hasattr(self.keyword_extractor, 'kiwi'):
            entities, sentences = self._refine_with_kiwi(
                entities, sentences, raw_keywords, original_query
            )
        
        # 중복 제거
        entities = list(dict.fromkeys(entities))
        sentences = list(dict.fromkeys(sentences))
        
        if self.verbose:
            print(f"         🔄 N-gram 재구성:")
            print(f"            Entities (PG): {entities}")
            print(f"            Sentences (Milvus): {sentences}")
        
        return {
            "entities": entities,
            "sentences": sentences
        }
    
    def _refine_with_kiwi(
        self,
        entities: List[str],
        sentences: List[str],
        raw_keywords: List[str],
        original_query: str
    ) -> tuple:
        """✅ NEW: Kiwi로 Entity/Sentence 분류 정제 (옵션)"""
        try:
            kiwi = self.keyword_extractor.kiwi
            
            # 원본 질문 형태소 분석
            tokens = kiwi.tokenize(original_query)
            token_dict = {token.form: token.tag for token in tokens[0][0]}  # {단어: 품사}
            
            # Entity 정제: 명사만 남기기
            refined_entities = []
            for entity in entities:
                pos_tag = token_dict.get(entity, 'UNKNOWN')
                if pos_tag in ['NNG', 'NNP', 'SL', 'SN']:  # 명사만
                    refined_entities.append(entity)
            
            # Sentence는 그대로 유지
            return refined_entities, sentences
            
        except Exception as e:
            if self.verbose:
                print(f"         ⚠️ Kiwi 정제 실패, 휴리스틱 결과 사용: {e}")
            return entities, sentences
    
    async def _find_synonyms(self, entity: str) -> List[str]:
        """
        ✅ PostgreSQL synonym 테이블에서 canonical_name 찾기
        
        예시:
        - Input: "물약"
        - Output: ["빨간 포션", "파란 포션", "하얀 포션"]
        
        Args:
            entity: 검색할 엔티티 이름
            
        Returns:
            동의어로 연결된 canonical_name 리스트
        """
        try:
            # 방법 1: description에서 해당 단어 포함하는 엔티티 찾기
            query = select(MapleDictionary).where(
                MapleDictionary.description.ilike(f"%{entity}%")
            ).limit(5)
            
            result = await self.db.execute(query)
            rows = result.scalars().all()
            
            canonical_names = []
            for row in rows:
                if row.canonical_name and row.canonical_name != entity:
                    canonical_names.append(row.canonical_name)
            
            return canonical_names
            
        except Exception as e:
            logger.warning(f"Synonym 검색 실패 ({entity}): {e}")
            return []
    
    async def _search_postgres_with_synonym(
        self,
        entities: List[str],
        limit_per_entity: int = 3
    ) -> List[Dict[str, Any]]:
        """
        ✅ Entity → PostgreSQL 검색 (canonical_name + synonym)
        
        전략:
        1. canonical_name 직접 매칭 시도
        2. 결과 없으면 synonym 테이블 검색
        3. synonym으로 찾은 canonical_name으로 재검색
        
        Args:
            entities: 검색할 Entity 리스트
            limit_per_entity: Entity당 최대 결과 개수
            
        Returns:
            PostgreSQL 검색 결과
        """
        results = []
        
        for entity in entities:
            # 1차: canonical_name 직접 검색
            direct_results = await self.pg_searcher.search(
                entity,
                category=None,
                limit=limit_per_entity
            )
            
            # sources 필드 추가
            for result in direct_results:
                if "sources" not in result:
                    result["sources"] = ["PostgreSQL"]
                result["match_type"] = "direct"  # 직접 매칭
            
            if len(direct_results) > 0:
                results.extend(direct_results)
                
                if self.verbose:
                    print(f"         📌 Entity '{entity}': {len(direct_results)}개 (직접)")
            else:
                # 2차: synonym 검색 (직접 매칭 실패 시)
                if self.verbose:
                    print(f"         🔍 Entity '{entity}': 직접 매칭 실패, synonym 검색...")
                
                synonyms = await self._find_synonyms(entity)
                
                for canonical in synonyms[:2]:  # 상위 2개만
                    synonym_results = await self.pg_searcher.search(
                        canonical,
                        category=None,
                        limit=2
                    )
                    
                    # sources 필드 추가
                    for result in synonym_results:
                        if "sources" not in result:
                            result["sources"] = ["PostgreSQL"]
                        result["match_type"] = "synonym"  # synonym 매칭
                        result["original_query"] = entity  # 원본 검색어 보존
                    
                    results.extend(synonym_results)
                
                if self.verbose and len(synonyms) > 0:
                    print(f"         📌 Entity '{entity}': {len(synonyms)}개 synonym → {len(results)}개 결과")
        
        return results
    
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
    
    async def _rerank_with_jina(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        ✅ Jina Reranker로 결과 재정렬
        
        RRF 후 노이즈 제거를 위해 LLM 기반 reranker 적용
        
        Args:
            query: 사용자 질문
            results: RRF로 병합된 결과
            top_n: 반환할 상위 개수
            
        Returns:
            Reranker 점수로 정렬된 상위 결과
        """
        if not results:
            return results
        
        try:
            # Reranker API URL (환경변수에서 가져오기)
            reranker_url = getattr(settings, 'RERANKER_API_URL', None) or \
                          os.getenv('RERANKER_API_URL', 'http://localhost:8001/rerank')
            
            # 결과를 텍스트로 변환
            texts = []
            for result in results:
                data = result.get("data", {})
                # print(f"여기서 오류 터짐 get data / results : {results}")
                text = f"{data.get('canonical_name', '')} - {data.get('description', '')}"
                texts.append(text)
            
            # Reranker API 호출
            payload = {
                "query": query,
                "texts": texts,
                "top_n": min(top_n, len(texts))
            }
            
            response = requests.post(reranker_url, json=payload, timeout=3)
            
            if response.status_code != 200:
                logger.warning(f"Reranker API 실패 (status {response.status_code}), RRF 결과 사용")
                return results
            
            reranked_data = response.json()
            
            # Reranker 결과에 따라 재정렬
            reranked_results = []
            for item in reranked_data.get("results", []):
                index = item.get("index")
                # print(f"여기서 오류 터짐 get index / results : {results}")
                score = item.get("score", 0)
                
                if index < len(results):
                    result = results[index].copy()
                    result["rerank_score"] = score
                    result["score"] = score * 100  # 0-100 스케일
                    reranked_results.append(result)
            
            return reranked_results
            
        except requests.exceptions.Timeout:
            logger.warning("Reranker API 타임아웃, RRF 결과 사용")
            return results
        except Exception as e:
            logger.warning(f"Reranker 실패: {e}, RRF 결과 사용")
            return results
    
    async def _search_milvus_sentences(
        self,
        sentences: List[str]
    ) -> List[Dict[str, Any]]:
        """
        ✅ Sentence → Milvus 의미 검색
        
        Args:
            sentences: 동사구 리스트 (예: ["물약 파는 사람"])
            
        Returns:
            Milvus 검색 결과
        """
        if not self.milvus_searcher:
            return []
        
        all_results = []
        for sentence in sentences:
            try:
                results = await self.milvus_searcher.search(sentence, top_k=5)
                
                # 결과 포맷팅
                for result in results:
                    all_results.append({
                        "score": result.get("score", 0) * 100,
                        "match_type": "vector_semantic",
                        "sources": ["Milvus"],
                        "data": result,
                        "search_query": sentence
                    })
                    
            except Exception as e:
                logger.warning(f"Milvus 검색 실패 ({sentence}): {e}")
        
        return all_results
    
    async def _search_neo4j_relations(
        self,
        query: str,
        entities: List[str],
        router_result: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        ✅ Neo4j 관계 검색 (hop >= 2)
        
        Args:
            query: 사용자 질문
            entities: Entity 리스트
            router_result: Router 결과 (relation 정보)
            
        Returns:
            Neo4j 검색 결과
        """
        if not self.neo4j_searcher:
            return []
        
        results = []
        relation = router_result.get("relation", "") if router_result else ""
        
        # Entity 기반으로 관계 검색
        for entity in entities:
            try:
                # 관계 유형에 따라 적절한 Neo4j 메서드 호출
                if "MONSTER-MAP" in relation or "어디" in query:
                    # 몬스터 위치
                    monster_results = await self.neo4j_searcher.find_monster_locations(entity)
                    results.extend(self._format_graph_results(monster_results, "graph_monster_location"))
                
                elif "NPC-MAP" in relation or "QUEST-NPC-MAP" in relation:
                    # NPC 위치
                    npc_results = await self.neo4j_searcher.find_npc_location(entity)
                    results.extend(self._format_graph_results(npc_results, "graph_npc_location"))
                
                elif "ITEM-MONSTER" in relation or "드랍" in query or "얻" in query:
                    # 아이템 드랍 몬스터
                    dropper_results = await self.neo4j_searcher.find_item_droppers(entity)
                    results.extend(self._format_graph_results(dropper_results, "graph_item_dropper"))
                
                elif "ITEM-NPC" in relation or "파는" in query or "구매" in query:
                    # 아이템 판매 NPC
                    seller_results = await self.neo4j_searcher.find_item_sellers(entity)
                    results.extend(self._format_graph_results(seller_results, "graph_item_seller"))
                
                elif "MAP-MAP" in relation or "가는" in query:
                    # 맵 연결
                    map_results = await self.neo4j_searcher.find_map_connections(entity)
                    results.extend(self._format_graph_results(map_results, "graph_map_connection"))
                    
            except Exception as e:
                logger.warning(f"Neo4j 관계 검색 실패 ({entity}): {e}")
        
        # PostgreSQL로 보강
        enriched_results = await self._enrich_graph_results(results)
        
        return enriched_results
    
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
