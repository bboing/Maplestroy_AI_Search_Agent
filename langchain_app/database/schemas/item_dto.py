from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Any
from enum import Enum

class ItemType(str, Enum):
    """아이템 종류"""
    WEAPON = "WEAPON"           # 무기
    ARMOR = "ARMOR"             # 방어구
    EQUIPMENT = "EQUIPMENT"     # 장비 (무기/방어구 통칭)
    ACCESSORY = "ACCESSORY"     # 악세서리
    CONSUMABLE = "CONSUMABLE"   # 소비 아이템
    ETC = "ETC"                 # 기타
    QUEST = "QUEST"             # 퀘스트 아이템
    CASH = "CASH"               # 캐시 아이템

class ItemGrade(str, Enum):
    """아이템 등급"""
    COMMON = "COMMON"           # 일반
    RARE = "RARE"               # 레어
    EPIC = "EPIC"               # 에픽
    UNIQUE = "UNIQUE"           # 유니크
    LEGENDARY = "LEGENDARY"     # 레전더리

class ItemMetadata(BaseModel):
    """
    메이플스토리 아이템 메타데이터 (JSON 구조와 정확히 일치)
    """
    category: Literal["ITEM"] = "ITEM"
    
    # 1. 기본 정보
    item_type: ItemType = Field(..., description="아이템 종류")
    description: Optional[str] = Field(None, description="아이템 설명")
    
    # 2. 획득 정보
    obtainable_from: List[str] = Field(default_factory=list, description="획득처 (몬스터, NPC, 퀘스트 등)")
    dropped_by: Optional[List[str]] = Field(None, description="드랍하는 몬스터 리스트")
    
    # 3. 가격 정보
    price: Optional[int] = Field(None, description="NPC 판매 가격")
    
    # 4. 착용 요구사항
    required_level: int = Field(0, description="착용 가능 레벨")
    required_job: List[str] = Field(default_factory=list, description="착용 가능 직업")
    required_stats: Optional[Dict[str, int]] = Field(None, description="요구 스탯")
    
    # 5. 스탯 정보 (무기/방어구의 경우)
    stats: Optional[Dict[str, int]] = Field(None, description="스탯 정보 (공격력, 방어력 등)")
    
    # 6. 효과 (소비 아이템의 경우)
    effects: List[Dict[str, Any]] = Field(default_factory=list, description="아이템 효과")
    
    # 7. 아이템 속성
    tradable: Optional[bool] = Field(None, description="거래 가능 여부")
    stackable: Optional[bool] = Field(None, description="중첩 가능 여부")
    max_stack: Optional[int] = Field(None, description="최대 중첩 개수")
    quest_item: Optional[bool] = Field(None, description="퀘스트 아이템 여부")
    consumable: Optional[bool] = Field(None, description="소비 아이템 여부")
    
    def extract_graph_relations(self, item_name: str) -> List[Dict[str, Any]]:
        """
        Neo4j 연동을 위한 관계 데이터 추출
        """
        relations = []
        
        # 아이템-몬스터/NPC 획득 관계
        for source in self.obtainable_from:
            # 몬스터 판별 (간단한 휴리스틱)
            if any(keyword in source for keyword in ["버섯", "슬라임", "좀비", "드래곤", "보스"]):
                relations.append({
                    "from_node": {"label": "Monster", "name": source},
                    "to_node": {"label": "Item", "name": item_name},
                    "relation": "DROPS"
                })
            else:
                # NPC로 추정
                relations.append({
                    "from_node": {"label": "NPC", "name": source},
                    "to_node": {"label": "Item", "name": item_name},
                    "relation": "SELLS"
                })
        
        return relations
