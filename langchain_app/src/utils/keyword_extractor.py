"""
Kiwi 형태소 분석 기반 키워드 추출 + 동의어 치환 (Async)
Router Fallback용
"""
from typing import List, Dict, Set
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from kiwipiepy import Kiwi
from database.models.maple_dictionary import MapleDictionary
import logging

logger = logging.getLogger(__name__)


class SynonymMapper:
    """
    동의어 → Canonical Name 매핑 (메모리 캐시)
    
    역할:
    - PostgreSQL에서 synonyms 로드
    - 메모리 dict로 빠른 검색
    - 동의어 → 정식명 치환
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.mapping: Dict[str, str] = {}  # 동의어 → canonical_name
        self.canonical_set: Set[str] = set()  # canonical_name 집합
        # ⚠️ __init__에서는 async 호출 불가 → 별도로 load_mappings() 호출 필요
    
    async def load_mappings(self):
        """DB에서 동의어 매핑 로드 (Async)"""
        try:
            stmt = select(MapleDictionary)
            result = await self.db.execute(stmt)
            entities = result.scalars().all()
            
            for entity in entities:
                canonical = entity.canonical_name
                self.canonical_set.add(canonical)
                
                # 자기 자신도 매핑 (정식명으로 검색 가능)
                self.mapping[canonical] = canonical
                
                # 동의어들 → 정식명 매핑
                if entity.synonyms:
                    for syn in entity.synonyms:
                        self.mapping[syn] = canonical
            
            logger.info(f"✅ 동의어 매핑 로드: {len(self.mapping)}개 (엔티티: {len(self.canonical_set)}개)")
            
        except Exception as e:
            logger.error(f"동의어 매핑 로드 실패: {e}")
    
    async def reload(self):
        """DB 업데이트 시 재로드 (Async)"""
        self.mapping.clear()
        self.canonical_set.clear()
        await self.load_mappings()
    
    def normalize(self, keywords: List[str]) -> List[str]:
        """
        동의어를 정식명으로 치환
        
        예: ['아진', '사'] → ['아이스진', '사']
        """
        normalized = []
        
        for keyword in keywords:
            # 동의어 사전에 있으면 정식명으로 치환
            canonical = self.mapping.get(keyword, keyword)
            normalized.append(canonical)
        
        return normalized
    
    def get_all_terms(self) -> List[str]:
        """모든 용어 반환 (Kiwi 사전 등록용)"""
        return list(self.mapping.keys())


class MapleKeywordExtractor:
    """
    Kiwi 형태소 분석 + 동의어 치환 기반 키워드 추출 (Async with Lazy Init)
    
    흐름:
    1. Kiwi 형태소 분석 (띄어쓰기 교정 자동)
    2. 명사만 추출 (NNG, NNP, SL)
    3. 동의어 → 정식명 치환
    4. 중복 제거
    """
    
    def __init__(self, db: AsyncSession):
        self.kiwi = Kiwi()
        self.mapper = SynonymMapper(db)
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Lazy initialization (첫 extract 호출 시 자동 초기화)"""
        if not self._initialized:
            await self.mapper.load_mappings()
            
            # 메이플 용어를 Kiwi 사전에 등록 (정확도 향상!)
            self._register_maple_terms()
            
            self._initialized = True
            logger.info("✅ MapleKeywordExtractor 초기화 완료")
    
    def _register_maple_terms(self):
        """메이플 용어를 Kiwi 사용자 사전에 등록"""
        try:
            terms = self.mapper.get_all_terms()
            
            # Kiwi에 일괄 등록 (고유명사로)
            for term in terms:
                if len(term) >= 2:  # 2글자 이상만
                    self.kiwi.add_user_word(term, "NNP")  # 고유명사
            
            logger.info(f"✅ Kiwi 사용자 사전 등록: {len(terms)}개 용어")
            
        except Exception as e:
            logger.warning(f"Kiwi 사전 등록 실패 (무시하고 진행): {e}")
    
    async def extract(self, query: str) -> List[str]:
        """
        질문에서 키워드 추출 (Async)
        
        Args:
            query: 사용자 질문
            
        Returns:
            정식명으로 치환된 키워드 리스트
        """
        try:
            # 0. 초기화 확인 (lazy init)
            await self._ensure_initialized()
            
            # 1. 형태소 분석
            tokens = self.kiwi.tokenize(query)
            
            # 2. 명사만 추출 (NNG: 일반명사, NNP: 고유명사, SL: 외국어)
            nouns = [
                t.form for t in tokens 
                if t.tag in ['NNG', 'NNP', 'SL']
                and len(t.form) >= 2  # 2글자 이상
            ]
            
            # 3. 동의어 → 정식명 치환
            normalized = self.mapper.normalize(nouns)
            
            # 4. 중복 제거 (순서 유지)
            keywords = list(dict.fromkeys(normalized))
            
            # 5. 비어있으면 원본 쿼리 사용
            if not keywords:
                keywords = [query]
            
            return keywords
            
        except Exception as e:
            logger.error(f"키워드 추출 실패: {e}")
            # 실패 시 원본 쿼리 반환
            return [query]
    
    async def reload_mappings(self):
        """DB 업데이트 시 동의어 매핑 재로드 (Async)"""
        await self.mapper.reload()
        self._register_maple_terms()
