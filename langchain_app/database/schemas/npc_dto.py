"""
NPC DTO (Data Transfer Object)
API 요청/응답 스키마 + MapleDictionary용 메타데이터 스키마
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal, Any
from datetime import datetime


class NPCCreate(BaseModel):
    """NPC 생성 요청"""
    npc_name: str = Field(..., min_length=1, max_length=100)
    city: str = Field(..., min_length=1, max_length=100)
    instruction: str = Field(..., min_length=10)
    description: Optional[str] = None
    keywords: Optional[str] = None
    extra_data: Optional[Dict] = None
    sample_conversations: Optional[List[Dict[str, str]]] = None


class NPCUpdate(BaseModel):
    """NPC 업데이트 요청"""
    instruction: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[str] = None
    extra_data: Optional[Dict] = None
    sample_conversations: Optional[List[Dict[str, str]]] = None
    is_active: Optional[str] = None


class NPCResponse(BaseModel):
    """NPC 응답"""
    id: str
    npc_name: str
    city: str
    instruction: str
    description: Optional[str]
    keywords: Optional[str]
    extra_data: Optional[Dict]
    sample_conversations: Optional[List[Dict[str, str]]]
    is_active: str
    created_at: Optional[datetime]
    
    class Config:
        from_attributes = True  # Pydantic v2 (orm_mode 대체)


class NPCChatRequest(BaseModel):
    """NPC 대화 요청"""
    npc_name: str = Field(..., description="NPC 이름")
    city: Optional[str] = Field(None, description="도시 (동명이인 방지)")
    message: str = Field(..., min_length=1, description="사용자 메시지")
    session_id: Optional[str] = Field(None, description="대화 세션 ID")
    use_rag: bool = Field(True, description="RAG 사용 여부")


class NPCChatResponse(BaseModel):
    """NPC 대화 응답"""
    npc_name: str
    city: str
    message: str
    response: str
    session_id: Optional[str]
    rag_used: bool
    retrieved_context: Optional[List[Dict]] = None
    latency_ms: Optional[int] = None


class NPCListResponse(BaseModel):
    """NPC 목록 응답"""
    npcs: List[NPCResponse]
    total: int
    cities: List[str]


class NPCMetadata(BaseModel):
    """
    메이플스토리 NPC 메타데이터 (MapleDictionary.detail_data용)
    JSON 구조와 정확히 일치하도록 설계
    """
    category: Literal["NPC"] = "NPC"
    
    # 1. 기본 정보 (JSON 실제 필드)
    npc_type: Optional[str] = Field(None, description="NPC 유형 (전직관, 상인 등)")
    location: Optional[str] = Field(None, description="위치 설명")
    region: Optional[str] = Field(None, description="소속 지역")
    description: Optional[str] = Field(None, description="NPC 설명")
    
    # 2. 기능 정보
    services: List[str] = Field(default_factory=list, description="제공 서비스 목록")
    related_quests: List[str] = Field(default_factory=list, description="관련 퀘스트 목록")
    
    # 3. 아이템 거래 정보
    sells_items: List[str] = Field(default_factory=list, description="판매 아이템 목록")
    buys_items: List[str] = Field(default_factory=list, description="구매 아이템 목록")
    
    # 4. 관계 정보
    related_npcs: List[str] = Field(default_factory=list, description="관련된 다른 NPC들")
    
    # 5. 대화 정보
    dialogue: Optional[str] = Field(None, description="대화 내용")
    personality: Optional[str] = Field(None, description="성격 특성")
    
    # 6. 상태 플래그
    is_merchant: Optional[bool] = Field(None, description="상인 여부")
    is_quest_giver: Optional[bool] = Field(None, description="퀘스트 제공 여부")
    
    def extract_graph_relations(self, npc_name: str) -> List[Dict[str, Any]]:
        """
        Neo4j 연동을 위한 관계 데이터 추출
        
        Returns:
            List of relationship dicts for Neo4j
        """
        relations = []
        
        # NPC-퀘스트 관계
        for quest in self.related_quests:
            relations.append({
                "from_node": {"label": "NPC", "name": npc_name},
                "to_node": {"label": "Quest", "name": quest},
                "relation": "GIVES_QUEST"
            })
        
        # NPC-아이템 관계 (판매)
        for item in self.sells_items:
            relations.append({
                "from_node": {"label": "NPC", "name": npc_name},
                "to_node": {"label": "Item", "name": item},
                "relation": "SELLS"
            })
        
        # NPC-아이템 관계 (구매)
        for item in self.buys_items:
            relations.append({
                "from_node": {"label": "NPC", "name": npc_name},
                "to_node": {"label": "Item", "name": item},
                "relation": "BUYS"
            })
        
        # NPC 간 관계
        for related_npc in self.related_npcs:
            relations.append({
                "from_node": {"label": "NPC", "name": npc_name},
                "to_node": {"label": "NPC", "name": related_npc},
                "relation": "RELATED_TO"
            })
        
        return relations
