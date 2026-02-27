from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Any
from enum import Enum

class MapType(str, Enum):
    """맵의 성격 구분"""
    TOWN = "TOWN"                # 마을
    TOWN_INSIDE = "TOWN_INSIDE"  # 마을 내 건물 
    FIELD = "FIELD"              # 일반 사냥터
    DUNGEON = "DUNGEON"          # 던전
    BOSS_ARENA = "BOSS_ARENA"    # 보스 전용 맵
    HIDDEN_STREET = "HIDDEN_STREET" # 히든 스트리트
    TRANSIT = "TRANSIT"          # 정거장/배 내부

class PortalDirection(str, Enum):
    """포탈의 위치 및 방향"""
    TOP_LEFT = "TOP_LEFT"; MID_LEFT = "MID_LEFT"; BOTTOM_LEFT = "BOTTOM_LEFT"
    TOP_RIGHT = "TOP_RIGHT"; MID_RIGHT = "MID_RIGHT"; BOTTOM_RIGHT = "BOTTOM_RIGHT"
    TOP_CENTER = "TOP_CENTER"; MID_CENTER = "MID_CENTER"; BOTTOM_CENTER = "BOTTOM_CENTER"
    HIDDEN = "HIDDEN"
    

class PortalType(str, Enum):
    """포탈 및 이동 수단의 종류"""
    REGULAR = "REGULAR"          # 일반 포탈
    DIMENSION_GATE = "DIMENSION_GATE" # 디멘션 게이트
    TAXI = "TAXI"                # 모범 택시
    SHIP = "SHIP"                # 배/비행선
    STATION = "STATION"          # 정거장
    HIDDEN_STREET = "HIDDEN_STREET" # 히든 스트리트 진입
    BUILDING_ENTRANCE = "BUILDING_ENTRANCE"  # 건물 입구
    NPC_TRANSPORT = "NPC_TRANSPORT"  # NPC 이동 서비스
    HIDDEN = "HIDDEN"  # 숨겨진 포탈

class AdjacentMapInfo(BaseModel):
    """일반 인접 맵 상세 정보"""
    target_map: str = Field(..., description="연결된 맵 명칭")
    direction: PortalDirection = Field(..., description="포탈의 구체적 방향")
    portal_name: Optional[str] = Field(None, description="포탈 이름")
    location_context: Optional[str] = Field(None, description="포탈 위치 설명")
    portal_type: Optional[PortalType] = Field(None, description="포탈 타입")
    description: Optional[str] = Field(None, description="위치 상세 설명")

class SpecialPortalInfo(BaseModel):
    """특수 이동 수단 정보"""
    # target_map 또는 target_region 중 하나는 있어야 함
    target_map: Optional[str] = Field(None, description="목적지 맵 (건물 입구 등)")
    target_region: Optional[str] = Field(None, description="목적지 대륙 또는 마을")
    portal_type: PortalType = Field(..., description="이동 수단 종류")
    portal_name: Optional[str] = Field(None, description="포탈 이름")
    location_context: Optional[str] = Field(None, description="포탈 위치 설명")
    description: Optional[str] = Field(None, description="포탈 상세 설명")
    cost: Optional[int] = Field(None, description="이동 비용")
    required_item: Optional[str] = Field(None, description="필요 아이템")

class MapMetadata(BaseModel):
    """
    메이플스토리 통합 맵 메타데이터 DTO (Optional 강화)
    """
    category: Literal["MAP"] = "MAP"
    map_type: MapType = Field(default=MapType.FIELD)
    
    # 1. 기본 정보 (region만 필수로 두고 나머지는 선택)
    region: str = Field(..., description="소속 대륙") 
    bgm: Optional[str] = Field(None, description="배경음악 제목")
    # description을 Optional로 변경하여 JSON에 없어도 OK
    description: Optional[str] = Field(None, description="맵 공식 설명")

    # 2. 지리 및 이동
    # default_factory=list 덕분에 JSON에 없으면 빈 리스트로 자동 생성됨
    adjacent_maps: List[AdjacentMapInfo] = Field(default_factory=list)
    special_portals: List[SpecialPortalInfo] = Field(default_factory=list)

    # 3. 관계용 식별자 리스트
    resident_npcs: List[str] = Field(default_factory=list)
    resident_monsters: List[str] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)

    # 4. 진입 및 환경 조건 (기본값을 설정해서 Optional하게 만듦)
    min_level: int = Field(0)
    recommended_level_range: Optional[List[int]] = Field(None, description="추천 레벨 범위 [min, max]")
    required_quest: Optional[str] = Field(None)
    star_force_limit: int = Field(0)
    arcane_power_limit: int = Field(0)
    is_safe_zone: bool = Field(False)
    respawn_time: Optional[int] = Field(None, description="몬스터 리스폰 시간 (초)")

    def extract_graph_relations(self, map_name: str) -> List[Dict[str, Any]]:
        """
        Neo4j 연동을 위한 관계 데이터 추출 (몬스터 포함)
        """
        relations = []
        
        # 맵-대륙 관계
        relations.append({
            "from_node": {"label": "Map", "name": map_name},
            "to_node": {"label": "Region", "name": self.region},
            "relation": "BELONGS_TO"
        })
        
        # 일반 포탈 연결 관계
        for adj in self.adjacent_maps:
            relations.append({
                "from_node": {"label": "Map", "name": map_name},
                "to_node": {"label": "Map", "name": adj.target_map},
                "relation": "CONNECTED_TO",
                "properties": {"direction": adj.direction.value}
            })

        # 특수 이동 수단 관계
        for sp in self.special_portals:
            target = sp.target_map or sp.target_region  # 둘 중 하나 사용
            if target:
                relations.append({
                    "from_node": {"label": "Map", "name": map_name},
                    "to_node": {"label": "Destination", "name": target},
                    "relation": "HAS_TRANSPORT",
                    "properties": {"type": sp.portal_type.value, "cost": sp.cost}
                })
            
        # NPC 거주 관계
        for npc in self.resident_npcs:
            relations.append({
                "from_node": {"label": "NPC", "name": npc},
                "to_node": {"label": "Map", "name": map_name},
                "relation": "RESIDES_IN"
            })

        # 몬스터 스폰 관계 (신규 추가)
        for monster in self.resident_monsters:
            relations.append({
                "from_node": {"label": "Monster", "name": monster},
                "to_node": {"label": "Map", "name": map_name},
                "relation": "SPAWNS_IN"
            })
            
        return relations