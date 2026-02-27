"""
Neo4j Relationship Schema
각 엔티티 타입별 관계 정의를 선언적으로 관리
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class RelationType(str, Enum):
    """관계 타입"""
    # MAP 관계
    HAS_NPC = "HAS_NPC"
    HAS_MONSTER = "HAS_MONSTER"
    CONNECTS_TO = "CONNECTS_TO"
    
    # NPC 관계
    LOCATED_IN = "LOCATED_IN"
    SELLS = "SELLS"
    
    # MONSTER 관계
    SPAWNS_IN = "SPAWNS_IN"
    DROPS = "DROPS"


@dataclass
class RelationshipRule:
    """
    관계 생성 규칙
    
    예:
        RelationshipRule(
            source_field="resident_npcs",
            target_category="NPC",
            relation_type="HAS_NPC",
            is_array=True
        )
    """
    source_field: str              # detail_data에서 읽을 필드명
    target_category: str           # 연결할 노드의 카테고리
    relation_type: RelationType    # 관계 타입
    is_array: bool = False         # 배열인지 단일값인지
    nested_field: Optional[str] = None  # dict일 경우 추출할 하위 필드
    reverse: bool = False          # True면 target -> source 방향으로 관계 생성
    bidirectional: bool = False    # True면 양방향 관계 생성 (source->target AND target->source)
    
    def extract_targets(self, detail_data: Dict[str, Any]) -> List[str]:
        """
        detail_data에서 타겟 이름들을 추출
        
        Returns:
            타겟 이름 리스트 (항상 list 반환)
        """
        value = detail_data.get(self.source_field)
        
        if value is None:
            return []
        
        # 단일 값
        if not self.is_array:
            if isinstance(value, str):
                return [value]
            return []
        
        # 배열 값
        if not isinstance(value, list):
            return []
        
        # nested_field가 있으면 dict에서 추출
        if self.nested_field:
            targets = []
            for item in value:
                if isinstance(item, dict):
                    target = item.get(self.nested_field)
                    if target:
                        targets.append(target)
                elif isinstance(item, str):
                    targets.append(item)
            return targets
        
        # 단순 문자열 리스트
        return [item for item in value if isinstance(item, str)]


class RelationshipSchema:
    """
    카테고리별 관계 스키마 정의
    """
    
    # MAP 관계 규칙
    MAP_RULES = [
        RelationshipRule(
            source_field="resident_npcs",
            target_category="NPC",
            relation_type=RelationType.HAS_NPC,
            is_array=True
        ),
        RelationshipRule(
            source_field="resident_monsters",
            target_category="MONSTER",
            relation_type=RelationType.HAS_MONSTER,
            is_array=True
        ),
        RelationshipRule(
            source_field="adjacent_maps",
            target_category="MAP",
            relation_type=RelationType.CONNECTS_TO,
            is_array=True,
            nested_field="target_map",  # dict일 경우 target_map 추출
            bidirectional=True  # 맵 연결은 양방향 (A→B 생성 시 B→A도 생성)
        ),
    ]
    
    # NPC 관계 규칙
    NPC_RULES = [
        RelationshipRule(
            source_field="location",
            target_category="MAP",
            relation_type=RelationType.LOCATED_IN,
            is_array=False
        ),
        RelationshipRule(
            source_field="sells_items",
            target_category="ITEM",
            relation_type=RelationType.SELLS,
            is_array=True
        ),
    ]
    
    # MONSTER 관계 규칙
    MONSTER_RULES = [
        RelationshipRule(
            source_field="spawn_maps",
            target_category="MAP",
            relation_type=RelationType.SPAWNS_IN,
            is_array=True
        ),
        RelationshipRule(
            source_field="drops",
            target_category="ITEM",
            relation_type=RelationType.DROPS,
            is_array=True,
            nested_field="item_name"  # drops는 {item_name, drop_rate} dict
        ),
    ]
    
    # ITEM 관계 규칙
    ITEM_RULES = [
        RelationshipRule(
            source_field="dropped_by",
            target_category="MONSTER",
            relation_type=RelationType.DROPS,  # MONSTER -[DROPS]-> ITEM
            is_array=True,
            reverse=True  # 역방향: MONSTER가 source, ITEM이 target
        ),
    ]
    
    # 전체 스키마 매핑
    SCHEMA_MAP = {
        "MAP": MAP_RULES,
        "NPC": NPC_RULES,
        "MONSTER": MONSTER_RULES,
        "ITEM": ITEM_RULES,
    }
    
    @classmethod
    def get_rules(cls, category: str) -> List[RelationshipRule]:
        """카테고리별 관계 규칙 반환"""
        return cls.SCHEMA_MAP.get(category, [])
    
    @classmethod
    def extract_all_relationships(
        cls, 
        category: str, 
        detail_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        카테고리와 detail_data에서 모든 관계 추출
        
        Returns:
            [
                {
                    "target_name": "커닝시티",
                    "target_category": "MAP",
                    "relation_type": "LOCATED_IN",
                    "reverse": False
                },
                ...
            ]
        """
        rules = cls.get_rules(category)
        relationships = []
        
        for rule in rules:
            targets = rule.extract_targets(detail_data)
            
            for target_name in targets:
                relationships.append({
                    "target_name": target_name,
                    "target_category": rule.target_category,
                    "relation_type": rule.relation_type.value,
                    "reverse": rule.reverse,
                    "bidirectional": rule.bidirectional
                })
        
        return relationships


# 편의 함수
def get_entity_relationships(
    category: str, 
    detail_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    엔티티의 모든 관계 추출
    
    Usage:
        relationships = get_entity_relationships("NPC", detail_data)
        # [{"target_name": "커닝시티", "target_category": "MAP", ...}]
    """
    return RelationshipSchema.extract_all_relationships(category, detail_data)
