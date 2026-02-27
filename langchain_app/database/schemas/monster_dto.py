from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Any
from enum import Enum

class MonsterType(str, Enum):
    """몬스터 종류"""
    NORMAL = "NORMAL"           # 일반 몬스터
    BOSS = "BOSS"               # 보스
    MINI_BOSS = "MINI_BOSS"     # 미니 보스
    FIELD_BOSS = "FIELD_BOSS"   # 필드 보스
    ELITE = "ELITE"             # 엘리트
    

class MonsterElement(str, Enum):
    """몬스터 속성"""
    FIRE = "FIRE"               # 화
    ICE = "ICE"                 # 빙
    LIGHTNING = "LIGHTNING"     # 전
    POISON = "POISON"           # 독
    HOLY = "HOLY"               # 성
    DARK = "DARK"               # 암
    UNDEAD = "UNDEAD"           # 언데드
    NEUTRAL = "NEUTRAL"         # 무속성

class DropInfo(BaseModel):
    """드랍 아이템 정보 (JSON 구조와 일치)"""
    item_name: str = Field(..., description="아이템 이름")
    drop_rate: float = Field(..., description="드랍률 (0.0~1.0)")
    quantity_min: Optional[int] = Field(None, description="최소 개수")
    quantity_max: Optional[int] = Field(None, description="최대 개수")

class MonsterMetadata(BaseModel):
    """
    메이플스토리 몬스터 메타데이터 (JSON 구조와 정확히 일치)
    """
    category: Literal["MONSTER"] = "MONSTER"
    
    # 1. 기본 정보
    monster_type: MonsterType = Field(default=MonsterType.NORMAL, description="몬스터 종류")
    level: int = Field(..., description="몬스터 레벨")
    description: Optional[str] = Field(None, description="몬스터 설명")
    
    # 2. 전투 정보
    hp: Optional[int] = Field(None, description="체력")
    mp: Optional[int] = Field(None, description="마나")
    attack: Optional[int] = Field(None, description="공격력")
    defense: Optional[int] = Field(None, description="방어력")
    
    # 3. 속성 정보
    element: Optional[str] = Field(None, description="속성 (문자열)")
    weaknesses: List[str] = Field(default_factory=list, description="약점 속성 목록")
    
    # 4. 경험치 및 보상
    exp: Optional[int] = Field(None, description="경험치")
    meso_drop_range: Optional[str] = Field(None, description="메소 드랍 범위 (예: '10-25')")
    drops: List[DropInfo] = Field(default_factory=list, description="드랍 아이템 목록")
    
    # 5. 위치 정보
    region: Optional[str] = Field(None, description="소속 지역")
    spawn_maps: List[str] = Field(default_factory=list, description="출몰 지역")
    
    # 6. 스킬 및 특징
    boss: Optional[bool] = Field(None, description="보스 여부")
    abilities: List[str] = Field(default_factory=list, description="특수 능력")
    
    def extract_graph_relations(self, monster_name: str) -> List[Dict[str, Any]]:
        """
        Neo4j 연동을 위한 관계 데이터 추출
        """
        relations = []
        
        # 몬스터-맵 스폰 관계
        for map_name in self.spawn_maps:
            relations.append({
                "from_node": {"label": "Monster", "name": monster_name},
                "to_node": {"label": "Map", "name": map_name},
                "relation": "SPAWNS_IN"
            })
        
        # 몬스터-아이템 드랍 관계
        for drop in self.drops:
            properties = {"drop_rate": drop.drop_rate}
            if drop.quantity_min is not None:
                properties["quantity_min"] = drop.quantity_min
            if drop.quantity_max is not None:
                properties["quantity_max"] = drop.quantity_max
                
            relations.append({
                "from_node": {"label": "Monster", "name": monster_name},
                "to_node": {"label": "Item", "name": drop.item_name},
                "relation": "DROPS",
                "properties": properties
            })
        
        return relations
