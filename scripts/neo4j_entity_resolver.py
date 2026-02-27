"""
Neo4j Entity Resolver
엔티티 이름 → Neo4j 노드 자동 매핑
"""
from typing import Dict, Optional, Set
from database.session import SessionLocal
from database.models.maple_dictionary import MapleDictionary
import logging

logger = logging.getLogger(__name__)


class EntityResolver:
    """
    엔티티 이름 해결기
    
    역할:
    1. canonical_name → entity_id 매핑
    2. synonyms → canonical_name 매핑
    3. 관계 생성 시 노드 존재 여부 확인
    """
    
    def __init__(self):
        self.db = SessionLocal()
        
        # 캐시: 이름 → (id, category)
        self.name_to_entity: Dict[str, tuple] = {}
        
        # 캐시: 카테고리별 모든 이름
        self.entities_by_category: Dict[str, Set[str]] = {
            "MAP": set(),
            "NPC": set(),
            "ITEM": set(),
            "MONSTER": set()
        }
        
        self._build_index()
    
    def _build_index(self):
        """인덱스 구축"""
        logger.info("🔨 Entity Resolver 인덱스 구축 중...")
        
        entities = self.db.query(MapleDictionary).all()
        
        for entity in entities:
            # Category 처리
            if hasattr(entity.category, 'value'):
                category = entity.category.value
            else:
                category = str(entity.category).split('.')[-1]
            
            entity_id = str(entity.id)
            canonical_name = entity.canonical_name
            
            # 1. canonical_name 매핑
            self.name_to_entity[canonical_name] = (entity_id, category)
            self.entities_by_category[category].add(canonical_name)
            
            # 2. synonyms 매핑
            if entity.synonyms:
                for synonym in entity.synonyms:
                    self.name_to_entity[synonym] = (entity_id, category)
        
        total_names = sum(len(names) for names in self.entities_by_category.values())
        logger.info(f"✅ 인덱스 구축 완료: {len(entities)}개 엔티티, {total_names}개 이름")
    
    def resolve(
        self, 
        name: str, 
        expected_category: Optional[str] = None
    ) -> Optional[tuple]:
        """
        이름 → (entity_id, category) 해결
        
        Args:
            name: 찾을 엔티티 이름
            expected_category: 예상 카테고리 (검증용)
            
        Returns:
            (entity_id, category) 또는 None
        """
        result = self.name_to_entity.get(name)
        
        if result is None:
            return None
        
        entity_id, category = result
        
        # 카테고리 검증
        if expected_category and category != expected_category:
            logger.warning(
                f"카테고리 불일치: {name} (기대: {expected_category}, 실제: {category})"
            )
            return None
        
        return result
    
    def exists(self, name: str, category: Optional[str] = None) -> bool:
        """엔티티 존재 여부 확인"""
        if category:
            return name in self.entities_by_category.get(category, set())
        return name in self.name_to_entity
    
    def get_all_names(self, category: str) -> Set[str]:
        """특정 카테고리의 모든 이름"""
        return self.entities_by_category.get(category, set())
    
    def close(self):
        """DB 연결 종료"""
        self.db.close()
