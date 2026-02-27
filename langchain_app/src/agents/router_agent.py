"""
Router Agent - Query Intent 분석 및 검색 전략 결정
"""
from typing import Dict, Any, List, Optional
from enum import Enum
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import settings
import json
import logging
import requests
import os

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Query 의도 분류"""
    # NPC 관련
    CLASS_CHANGE = "class_change"      # 전직
    NPC_LOCATION = "npc_location"      # NPC 위치
    NPC_SERVICE = "npc_service"        # NPC 서비스
    
    # MAP 관련
    HUNTING_GROUND = "hunting_ground"  # 사냥터
    MAP_LOCATION = "map_location"      # 맵 위치
    MAP_FEATURE = "map_feature"        # 맵 특징
    
    # ITEM 관련
    ITEM_PURCHASE = "item_purchase"    # 아이템 구매
    ITEM_DROP = "item_drop"            # 아이템 드랍
    ITEM_INFO = "item_info"            # 아이템 정보
    
    # MONSTER 관련
    MONSTER_LOCATION = "monster_location"  # 몬스터 위치
    MONSTER_INFO = "monster_info"          # 몬스터 정보
    
    # 관계 관련
    QUEST_RELATION = "quest_relation"  # 퀘스트 연관
    ITEM_RELATION = "item_relation"    # 아이템 연관
    
    # 일반
    GENERAL = "general"                # 일반 질문


class SearchStrategy(str, Enum):
    """검색 전략"""
    SIMPLE = "simple"        # PostgreSQL 직접 검색
    SEMANTIC = "semantic"    # Milvus 의미 검색
    RELATION = "relation"    # Neo4j 관계 검색
    HYBRID = "hybrid"        # 복합 검색


class RouterAgent:
    """
    Query Intent 분석 및 검색 전략 결정
    
    역할:
    1. Query의 의도(Intent) 파악
    2. 검색할 Category 결정
    3. 검색 전략 결정
    4. 핵심 키워드 추출
    5. Multi-step 검색 계획 수립 (NEW!)
    """
    
    # 기존 간단한 Router 프롬프트 (하위 호환)
    ROUTER_SYSTEM_PROMPT = """당신은 메이플스토리 검색 시스템의 Router입니다.
사용자의 질문을 분석하여 의도(Intent)와 검색 전략을 결정합니다.

## 주요 Intent 분류

### NPC 관련
- class_change: 전직, 직업 변경 (예: "도적 전직 어디서?", "궁수로 전직하려면?")
- npc_location: NPC 위치 찾기 (예: "다크로드 어디?", "페이슨 위치")
- npc_service: NPC가 제공하는 서비스 (예: "창고 어디?", "상점 찾기")

### MAP 관련
- hunting_ground: 사냥터 찾기 (예: "도적 사냥터 추천", "20레벨 사냥터")
- map_location: 맵 위치 (예: "헤네시스 어떻게 가?", "리스항구 가는 법")
- map_feature: 맵 특징 (예: "엘리니아에 뭐있어?", "커닝시티 특징")

### ITEM 관련
- item_purchase: 아이템 구매 (예: "아이스진 어디서 사?", "물약 파는 곳")
- item_drop: 아이템 드랍 (예: "아이스진 떨구는 몹", "어디서 나와?")
- item_info: 아이템 정보 (예: "아이스진 능력치", "가격은?")

### MONSTER 관련
- monster_location: 몬스터 위치 (예: "스포아 어디?", "주황버섯 사냥터")
- monster_info: 몬스터 정보 (예: "스포아 레벨", "체력은?")
- monster_drop: 몬스터 드랍 정보(예: "스포아 잡으면 뭐 나와요?", "주황버섯이 떨구는 아이템")

## Category 우선순위

Intent에 따른 Category:
- class_change → NPC (전직관)
- hunting_ground → MAP, MONSTER (사냥터)
- item_purchase → ITEM, NPC (상점)
- item_drop → ITEM, MONSTER (드랍)
- npc_location → NPC, MAP (NPC 위치)
- monster_location → MONSTER, MAP (몬스터 위치)

## 검색 전략

- SIMPLE: 정확한 이름/위치 검색 (PostgreSQL)
- SEMANTIC: 의미 기반 추천/검색 (Milvus)
- RELATION: 관계 기반 검색 (Neo4j)
- HYBRID: 복합 검색

## 응답 형식 (JSON)

{
  "intent": "class_change",
  "categories": ["NPC"],
  "strategy": "SIMPLE",
  "keywords": ["도적", "전직"],
  "reasoning": "도적으로 전직하기 위한 NPC를 찾는 질문"
}"""
    
    # 새로운 전략 분석가 프롬프트
    STRATEGY_PLANNER_PROMPT = """너는 메이플스토리 전문 상담 시스템의 '전략 분석가'야.
너에게는 3가지 지식 저장소(Tool)가 있어. 유저의 질문을 해결하기 위한 최적의 '공략 로직'을 짜줘.

[도구 명세]

1. **SQL_DB** (PostgreSQL - 정적 데이터 조회)
   - 용도: 아이템/NPC/맵/몬스터의 고정된 스펙 조회
   - 강점: 빠르고 정확한 검색 (0.1초), 가격/수치/이름/설명
   - 예시: "아이스진 가격", "다크로드 위치", "스포아 레벨"
   - 쿼리 형식: 엔티티 이름 or 속성명

2. **GRAPH_DB** (Neo4j - 관계 추적)
   - 용도: 엔티티 간의 연결 관계 추적
   - 강점: 복잡한 관계 탐색, 경로 찾기, 연관 정보
   - 관계 유형:
     * NPC → MAP (위치)
     * MONSTER → MAP (출현 지역)
     * MONSTER → ITEM (몬스터 드랍)
     * ITEM → MONSTER (드랍 몬스터)
     * NPC → ITEM (판매)
     * MAP → MAP (이동 경로)
   - 예시: "다크로드가 있는 맵", "스포아가 떨구는 아이템", "헤네시스에서 엘리니아 가는 법", "아이스진 나오는 몬스터"
   - 쿼리 형식: "엔티티A → 관계 → 엔티티B"

3. **VECTOR_DB** (Milvus - 의미 검색)
   - 용도: 의미/맥락이 비슷한 정보 추천
   - 강점: 태그 기반 검색, 추천, 분위기/컨셉 매칭
   - 예시: "도적 사냥터 추천", "초보자 사냥터", "돈 잘 버는 아이템"
   - 쿼리 형식: 자연어 질문 or 컨셉

[전략 수립 원칙]

1. **단순 → 복잡 순서**로 검색 (빠른 도구부터)
   - 이름 검색 → SQL_DB 먼저
   - 관계 추적 → GRAPH_DB
   - 추천/비슷한 것 → VECTOR_DB

2. **필요한 정보만** 검색
   - "다크로드 어디?" → SQL_DB만 (위치 정보 있음)
   - "도적 전직 어디서?" → SQL_DB(전직 NPC) + GRAPH_DB(NPC→MAP 위치)

3. **의도 파악이 핵심**
   - "도적 사냥터" → VECTOR_DB (추천 의도)
   - "도적 전직" → SQL_DB (특정 NPC 찾기)
   - "스포아 잡으려면?" → SQL_DB(스포아 정보) + GRAPH_DB(스포아→MAP)

[출력 규격]

반드시 아래 JSON 형식으로만 답해:

{
  "thought": "유저 질문 분석 (무엇을 원하는지, 어떤 정보가 필요한지)",
  "plan": [
    {
      "step": 1,
      "tool": "SQL_DB|GRAPH_DB|VECTOR_DB",
      "query": "검색할 내용",
      "reason": "이 도구를 이 순서에 쓰는 이유",
      "expected": "이 단계에서 얻을 정보"
    }
  ]
}

[예시]

질문: "도적 전직 어디서?"
{
  "thought": "도적으로 전직하려면 전직 담당 NPC를 찾고, 그 NPC의 위치를 알아야 함",
  "plan": [
    {
      "step": 1,
      "tool": "SQL_DB",
      "query": "도적 전직 NPC",
      "reason": "먼저 전직을 담당하는 NPC 이름과 기본 정보 조회",
      "expected": "다크로드(NPC)"
    },
    {
      "step": 2,
      "tool": "GRAPH_DB",
      "query": "다크로드 → 위치 → MAP",
      "reason": "NPC가 어느 맵에 있는지 관계 추적",
      "expected": "여섯갈래길(MAP)"
    }
  ]
}

질문: "도적 사냥터 추천"
{
  "thought": "도적 직업에 적합한 사냥터를 추천해야 함. 레벨대/특성 고려 필요",
  "plan": [
    {
      "step": 1,
      "tool": "VECTOR_DB",
      "query": "도적 직업 적합한 사냥터",
      "reason": "의미 기반으로 도적 특성에 맞는 맵/몬스터 추천",
      "expected": "추천 사냥터 리스트(MAP, MONSTER)"
    }
  ]
}

질문: "아이스진 어디서 구해?"
{
  "thought": "아이스진을 구하는 방법 - 구매 or 드랍. 두 경로 모두 확인",
  "plan": [
    {
      "step": 1,
      "tool": "SQL_DB",
      "query": "아이스진",
      "reason": "아이템 기본 정보 조회",
      "expected": "아이스진 스펙, 가격"
    },
    {
      "step": 2,
      "tool": "GRAPH_DB",
      "query": "아이스진 → 판매 NPC",
      "reason": "어느 NPC가 파는지 확인",
      "expected": "판매 NPC 리스트"
    },
    {
      "step": 3,
      "tool": "GRAPH_DB",
      "query": "아이스진 → 드랍 몬스터",
      "reason": "어느 몬스터가 떨구는지 확인",
      "expected": "드랍 몬스터 리스트"
    }
  ]
}

이제 유저 질문에 대한 최적의 검색 전략을 JSON으로 답해줘."""

    def __init__(
        self,
        llm: Optional[ChatOllama] = None,
        use_strategy_planner: bool = True,
        verbose: bool = False
    ):
        # LLM이 명시적으로 전달되면 그대로 사용
        if llm:
            self.llm = llm
        else:
            # Health check로 Ollama/Groq 자동 선택
            self.llm = self._initialize_llm()
        
        self.use_strategy_planner = use_strategy_planner
        self.verbose = verbose
    
    def _initialize_llm(self):
        """Ollama health check 후 Groq fallback"""
        # 1. Ollama health check
        try:
            ollama_url = settings.OLLAMA_BASE_URL
            response = requests.get(f"{ollama_url}/api/tags", timeout=2)
            
            if response.status_code == 200:
                logger.info(f"✅ Ollama 연결 성공: {ollama_url}")
                return ChatOllama(
                    model=settings.OLLAMA_MODEL,
                    base_url=ollama_url,
                    temperature=0.0
                )
        except Exception as e:
            logger.warning(f"⚠️ Ollama 연결 실패: {e}")
        
        # 2. Groq fallback
        try:
            groq_api_key = getattr(settings, 'GROQ_API_KEY', None) or os.getenv('GROQ_API_KEY')
            groq_model = getattr(settings, 'GROQ_MODEL_NAME', None) or os.getenv('GROQ_MODEL_NAME', 'llama-3.3-70b-versatile')
            
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not found in settings or environment")
            
            logger.info(f"✅ Groq fallback 활성화: {groq_model}")
            return ChatGroq(
                model=groq_model,
                api_key=groq_api_key,
                temperature=0.0
            )
        except Exception as e:
            logger.error(f"❌ router_agent.py: Groq fallback도 실패: {e}")
            raise RuntimeError("router_agent.py: Ollama와 Groq 모두 사용 불가능합니다.")
    
    def _switch_to_groq(self):
        """Runtime에 Ollama 실패 시 Groq으로 전환"""
        try:
            groq_api_key = getattr(settings, 'GROQ_API_KEY', None) or os.getenv('GROQ_API_KEY')
            groq_model = getattr(settings, 'GROQ_MODEL_NAME', None) or os.getenv('GROQ_MODEL_NAME', 'llama-3.3-70b-versatile')
            
            if not groq_api_key:
                logger.error("GROQ_API_KEY not found")
                return
            
            logger.info(f"🔄 Groq으로 전환: {groq_model}")
            self.llm = ChatGroq(
                model=groq_model,
                api_key=groq_api_key,
                temperature=0.0
            )
        except Exception as e:
            logger.error(f"❌ Groq 전환 실패: {e}")
    
    def plan_search_strategy(self, query: str) -> Dict[str, Any]:
        """
        전략 분석가 모드: Multi-step 검색 계획 수립
        
        Args:
            query: 사용자 질문
            
        Returns:
            {
                "thought": str,  # 분석 내용
                "plan": [        # 검색 계획 (순서대로)
                    {
                        "step": int,
                        "tool": "SQL_DB|GRAPH_DB|VECTOR_DB",
                        "query": str,
                        "reason": str,
                        "expected": str
                    }
                ]
            }
        """
        if self.verbose:
            print(f"\n🧠 Strategy Planner: 전략 수립 중... '{query}'")
        
        try:
            # LLM으로 검색 전략 수립
            messages = [
                SystemMessage(content=self.STRATEGY_PLANNER_PROMPT),
                HumanMessage(content=f"유저 질문: {query}")
            ]
            
            response = self.llm.invoke(messages)
            
            # JSON 파싱
            result = self._parse_plan_response(response.content)
            
            if self.verbose:
                print(f"   💭 Thought: {result['thought']}")
                print(f"   📋 Plan: {len(result['plan'])} steps")
                for step in result['plan']:
                    print(f"      Step {step['step']}: {step['tool']} - {step['query']}")
            
            return result
            
        except Exception as e:
            logger.warning(f"Strategy Planner 실패: {e}")
            raise
    
    def route(self, query: str) -> Dict[str, Any]:
        """
        Query를 분석하여 검색 전략 결정
        
        Args:
            query: 사용자 질문
            
        Returns:
            기존 형식 (하위 호환):
            {
                "intent": QueryIntent,
                "categories": List[str],
                "strategy": SearchStrategy,
                "keywords": List[str],
                "reasoning": str
            }
            
            또는 새로운 형식 (use_strategy_planner=True):
            {
                "thought": str,
                "plan": List[Dict],
                "intent": QueryIntent,  # plan에서 추론
                "categories": List[str],  # plan에서 추론
            }
        """
        if self.verbose:
            print(f"\n🧭 Router: 분석 중... '{query}'")
        
        # 전략 수립 모드 시도
        if self.use_strategy_planner:
            try:
                plan_result = self.plan_search_strategy(query)
                # Plan을 기존 형식으로도 변환 (하위 호환)
                converted = self._convert_plan_to_route(plan_result, query)
                return {**plan_result, **converted}
            except Exception as e:
                logger.warning(f"Strategy Planner 실패, 기본 Router 시도: {e}")
                # Ollama Runtime 에러 시 Groq으로 전환
                if "not found" in str(e) or "404" in str(e) or "Connection" in str(e):
                    self._switch_to_groq()
        
        # 기존 Router 모드
        try:
            # LLM으로 Intent 분석
            messages = [
                SystemMessage(content=self.ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=f"질문: {query}\n\nJSON 형식으로 분석 결과를 응답해주세요.")
            ]
            
            response = self.llm.invoke(messages)
            
            # JSON 파싱
            result = self._parse_response(response.content)
            
            if self.verbose:
                print(f"   Intent: {result['intent']}")
                print(f"   Categories: {result['categories']}")
                print(f"   Strategy: {result['strategy']}")
                print(f"   Keywords: {result['keywords']}")
                print(f"   Reasoning: {result['reasoning']}")
            
            return result
            
        except Exception as e:
            logger.warning(f"Router LLM 실패, Fallback 사용: {e}")
            # Ollama Runtime 에러 시 Groq으로 전환 후 재시도
            if "not found" in str(e) or "404" in str(e) or "Connection" in str(e):
                self._switch_to_groq()
                # Groq으로 재시도
                try:
                    messages = [
                        SystemMessage(content=self.ROUTER_SYSTEM_PROMPT),
                        HumanMessage(content=f"질문: {query}\n\nJSON 형식으로 분석 결과를 응답해주세요.")
                    ]
                    response = self.llm.invoke(messages)
                    result = self._parse_response(response.content)
                    return result
                except:
                    pass  # Groq도 실패하면 fallback 사용
            
            # Fallback: 키워드 기반 분류
            return self._fallback_classification(query)
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
        """LLM 응답을 파싱 (기존 Router 형식)"""
        try:
            # JSON 블록 추출
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()
            
            # JSON 파싱
            data = json.loads(content)
            
            return {
                "intent": data.get("intent", QueryIntent.GENERAL),
                "categories": data.get("categories", []),
                "strategy": data.get("strategy", SearchStrategy.SEMANTIC),
                "keywords": data.get("keywords", []),
                "reasoning": data.get("reasoning", "")
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 실패: {e}, content={content}")
            # Fallback: 키워드 기반 간단한 분류
            return self._fallback_classification(content)
    
    def _parse_plan_response(self, content: str) -> Dict[str, Any]:
        """LLM 응답을 파싱 (전략 수립 형식)"""
        try:
            # JSON 블록 추출
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()
            
            # ✅ LLM이 {{를 쓰는 문제 해결
            content = content.replace("{{", "{").replace("}}", "}")
            
            # 디버깅: 파싱 시도 전 내용 출력
            if self.verbose:
                print(f"\n[DEBUG] 파싱할 내용:\n{content[:500]}\n")
            
            # JSON 파싱
            data = json.loads(content)
            
            return {
                "thought": data.get("thought", ""),
                "plan": data.get("plan", [])
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Plan JSON 파싱 실패: {e}")
            logger.error(f"LLM 응답 내용:\n{content}")
            raise
    
    def _convert_plan_to_route(self, plan_result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Plan 결과를 기존 Route 형식으로 변환 (하위 호환)
        
        Args:
            plan_result: Plan 결과
            query: 원본 사용자 질문
        
        Tool → Category 매핑:
        - SQL_DB → 첫 번째 query에서 추론
        - GRAPH_DB → 관계에서 추론
        - VECTOR_DB → 추천 의도
        """
        plan = plan_result.get("plan", [])
        
        if not plan:
            return {
                "intent": QueryIntent.GENERAL,
                "categories": [],
                "strategy": SearchStrategy.SEMANTIC,
                "keywords": [],
                "reasoning": plan_result.get("thought", "")
            }
        
        # 첫 번째 step의 tool로 strategy 결정
        first_tool = plan[0].get("tool", "SQL_DB")
        
        strategy_map = {
            "SQL_DB": SearchStrategy.SIMPLE,
            "GRAPH_DB": SearchStrategy.RELATION,
            "VECTOR_DB": SearchStrategy.SEMANTIC
        }
        
        strategy = strategy_map.get(first_tool, SearchStrategy.HYBRID)
        
        # Plan에서 category 추론
        categories = []
        for step in plan:
            query_lower = step.get("query", "").lower()
            if any(word in query_lower for word in ["npc", "엔피시", "상인"]):
                if "NPC" not in categories:
                    categories.append("NPC")
            if any(word in query_lower for word in ["map", "맵", "사냥터", "지역"]):
                if "MAP" not in categories:
                    categories.append("MAP")
            if any(word in query_lower for word in ["monster", "몬스터", "몹"]):
                if "MONSTER" not in categories:
                    categories.append("MONSTER")
            if any(word in query_lower for word in ["item", "아이템"]):
                if "ITEM" not in categories:
                    categories.append("ITEM")
        
        # ✅ 원본 질문 기반 Category 보정 (LLM이 놓친 경우 대비)
        original_lower = query.lower()
        
        # 아이템 관련
        if any(word in original_lower for word in ["아이템", "구하", "구매", "사", "파는", "드랍", "떨구", "나와"]):
            if "ITEM" not in categories:
                categories.insert(0, "ITEM")
        
        # 몬스터 관련
        if any(word in original_lower for word in ["몬스터", "몹", "잡"]):
            if "MONSTER" not in categories:
                idx = 0 if "ITEM" not in categories else 1
                categories.insert(idx, "MONSTER")
        
        # NPC 관련
        if any(word in original_lower for word in ["npc", "엔피시", "상인", "전직"]):
            if "NPC" not in categories:
                categories.append("NPC")
        
        # MAP 관련
        if any(word in original_lower for word in ["맵", "사냥터", "지역", "어디"]):
            if "MAP" not in categories:
                categories.append("MAP")
        
        # Intent 추론 (thought + 원본 질문 기반)
        thought_lower = plan_result.get("thought", "").lower()
        intent = QueryIntent.GENERAL
        
        # 전직
        if "전직" in thought_lower or "전직" in original_lower:
            intent = QueryIntent.CLASS_CHANGE
        # 사냥터
        elif ("사냥터" in thought_lower or "추천" in thought_lower) or \
             ("사냥터" in original_lower or ("추천" in original_lower and "레벨" in original_lower)):
            intent = QueryIntent.HUNTING_GROUND
        # 아이템 구매
        elif ("구매" in thought_lower or "파는" in thought_lower) or \
             any(word in original_lower for word in ["구하", "구매", "사", "파는"]):
            intent = QueryIntent.ITEM_PURCHASE
        # 아이템 드랍
        elif ("드랍" in thought_lower or "떨구" in thought_lower) or \
             any(word in original_lower for word in ["드랍", "떨구", "나와"]):
            intent = QueryIntent.ITEM_DROP
        # 몬스터 위치
        elif any(word in original_lower for word in ["잡", "몬스터"]) and "어디" in original_lower:
            intent = QueryIntent.MONSTER_LOCATION
        # NPC 위치
        elif ("위치" in thought_lower and "npc" in thought_lower) or \
             (any(word in original_lower for word in ["npc", "엔피시"]) and "어디" in original_lower):
            intent = QueryIntent.NPC_LOCATION
        # 맵 위치
        elif "어디" in original_lower and ("맵" in original_lower or "가는" in original_lower):
            intent = QueryIntent.MAP_LOCATION
        
        return {
            "intent": intent,
            "categories": categories,
            "strategy": strategy,
            "keywords": [step.get("query", "") for step in plan],
            "reasoning": plan_result.get("thought", "")
        }
    
    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        """
        LLM 실패 시 규칙 기반 전략 분석가
        
        더 이상 단순 키워드 매칭이 아닌, 
        질문을 분석하고 SQL/GRAPH/VECTOR 도구 선택 전략 수립
        """
        query_lower = query.lower()
        
        # === 전직 관련 ===
        if any(word in query_lower for word in ["전직", "직업", "배우", "가르쳐"]):
            # 전직 NPC 찾기 → NPC 위치 추적
            return {
                "thought": "전직 담당 NPC를 찾고, 그 NPC가 어디에 있는지 위치를 알아야 함",
                "plan": [
                    {
                        "step": 1,
                        "tool": "SQL_DB",
                        "query": f"{query} 전직 NPC",
                        "reason": "전직을 담당하는 NPC 이름 조회",
                        "expected": "전직 NPC 정보"
                    },
                    {
                        "step": 2,
                        "tool": "GRAPH_DB",
                        "query": "NPC → MAP 위치 관계",
                        "reason": "NPC가 어느 맵에 있는지 추적",
                        "expected": "맵 위치 정보"
                    }
                ],
                "intent": QueryIntent.CLASS_CHANGE,
                "categories": ["NPC"],
                "strategy": SearchStrategy.RELATION,
                "keywords": [query],
                "reasoning": "전직: SQL로 NPC 찾고 → GRAPH로 위치 추적"
            }
        
        # === 사냥터 추천 ===
        elif any(word in query_lower for word in ["사냥터", "사냥", "레벨업", "추천"]):
            return {
                "thought": "직업/레벨에 맞는 사냥터를 추천해야 함. 맵 특성과 몬스터 고려",
                "plan": [
                    {
                        "step": 1,
                        "tool": "VECTOR_DB",
                        "query": f"{query} 적합한 맵과 몬스터",
                        "reason": "의미 기반으로 적합한 사냥터 추천",
                        "expected": "추천 맵/몬스터 리스트"
                    }
                ],
                "intent": QueryIntent.HUNTING_GROUND,
                "categories": ["MAP", "MONSTER"],
                "strategy": SearchStrategy.SEMANTIC,
                "keywords": [query],
                "reasoning": "사냥터 추천: VECTOR로 의미 기반 추천"
            }
        
        # === 아이템 구입/획득 (구하다, 사다, 파는) ===
        elif any(word in query_lower for word in ["구하", "구매", "사다", "사", "파는", "상점", "어디서"]):
            return {
                "thought": "아이템을 구하는 방법 - 구매 경로와 드랍 경로 모두 확인",
                "plan": [
                    {
                        "step": 1,
                        "tool": "SQL_DB",
                        "query": f"{query} 아이템 정보",
                        "reason": "아이템 기본 정보 조회",
                        "expected": "아이템 스펙, 가격"
                    },
                    {
                        "step": 2,
                        "tool": "GRAPH_DB",
                        "query": "ITEM → NPC 판매 관계",
                        "reason": "어느 NPC가 파는지 확인",
                        "expected": "판매 NPC"
                    },
                    {
                        "step": 3,
                        "tool": "GRAPH_DB",
                        "query": "ITEM → MONSTER 드랍 관계",
                        "reason": "어느 몬스터가 떨구는지 확인",
                        "expected": "드랍 몬스터"
                    }
                ],
                "intent": QueryIntent.ITEM_PURCHASE,
                "categories": ["ITEM", "NPC", "MONSTER"],
                "strategy": SearchStrategy.RELATION,
                "keywords": [query],
                "reasoning": "아이템 획득: SQL로 정보 → GRAPH로 구매/드랍 경로"
            }
        
        # === 아이템 드랍 ===
        elif any(word in query_lower for word in ["드랍", "떨구", "떨어", "나와"]):
            return {
                "thought": "아이템을 드랍하는 몬스터를 찾고, 그 몬스터 위치 추적",
                "plan": [
                    {
                        "step": 1,
                        "tool": "GRAPH_DB",
                        "query": f"{query} ITEM → MONSTER 드랍",
                        "reason": "드랍하는 몬스터 찾기",
                        "expected": "몬스터 리스트"
                    },
                    {
                        "step": 2,
                        "tool": "GRAPH_DB",
                        "query": "MONSTER → MAP 위치",
                        "reason": "몬스터가 있는 맵 찾기",
                        "expected": "사냥터 정보"
                    }
                ],
                "intent": QueryIntent.ITEM_DROP,
                "categories": ["ITEM", "MONSTER", "MAP"],
                "strategy": SearchStrategy.RELATION,
                "keywords": [query],
                "reasoning": "드랍: GRAPH로 몬스터 찾고 → 위치 추적"
            }
        
        # === 몬스터 위치 (잡으려면) ===
        elif any(word in query_lower for word in ["잡", "몬스터", "몹"]):
            return {
                "thought": "몬스터 정보를 조회하고, 어느 맵에 출현하는지 확인",
                "plan": [
                    {
                        "step": 1,
                        "tool": "SQL_DB",
                        "query": f"{query} 몬스터 정보",
                        "reason": "몬스터 기본 스펙 조회",
                        "expected": "몬스터 레벨, HP, 공격력"
                    },
                    {
                        "step": 2,
                        "tool": "GRAPH_DB",
                        "query": "MONSTER → MAP 출현 지역",
                        "reason": "몬스터가 나타나는 맵 추적",
                        "expected": "출현 맵 리스트"
                    }
                ],
                "intent": QueryIntent.MONSTER_LOCATION,
                "categories": ["MONSTER", "MAP"],
                "strategy": SearchStrategy.RELATION,
                "keywords": [query],
                "reasoning": "몬스터 위치: SQL로 정보 → GRAPH로 출현 맵"
            }
        
        # === NPC/맵 위치 (어디, 위치) ===
        elif any(word in query_lower for word in ["어디", "위치", "있어"]):
            return {
                "thought": "엔티티 이름으로 위치 정보 조회",
                "plan": [
                    {
                        "step": 1,
                        "tool": "SQL_DB",
                        "query": f"{query}",
                        "reason": "엔티티 기본 정보 및 위치 조회",
                        "expected": "NPC 또는 맵 정보"
                    }
                ],
                "intent": QueryIntent.NPC_LOCATION,
                "categories": ["NPC", "MAP"],
                "strategy": SearchStrategy.SIMPLE,
                "keywords": [query],
                "reasoning": "위치 질문: SQL로 직접 조회"
            }
        
        # === 맵 이동 (가는 법) ===
        elif any(word in query_lower for word in ["가는", "이동", "가려면"]):
            return {
                "thought": "출발지에서 목적지까지의 이동 경로 찾기",
                "plan": [
                    {
                        "step": 1,
                        "tool": "GRAPH_DB",
                        "query": f"{query} MAP → MAP 경로",
                        "reason": "맵 간 이동 경로 추적",
                        "expected": "이동 경로"
                    }
                ],
                "intent": QueryIntent.MAP_LOCATION,
                "categories": ["MAP"],
                "strategy": SearchStrategy.RELATION,
                "keywords": [query],
                "reasoning": "이동 경로: GRAPH로 경로 탐색"
            }
        
        # === 일반 질문 (의미 검색) ===
        else:
            return {
                "thought": "명확한 의도를 파악하기 어려움. 의미 기반 검색으로 관련 정보 탐색",
                "plan": [
                    {
                        "step": 1,
                        "tool": "VECTOR_DB",
                        "query": f"{query}",
                        "reason": "의미적으로 유사한 정보 검색",
                        "expected": "관련 정보"
                    }
                ],
                "intent": QueryIntent.GENERAL,
                "categories": [],
                "strategy": SearchStrategy.SEMANTIC,
                "keywords": [query],
                "reasoning": "일반 질문: VECTOR로 의미 검색"
            }


# 편의 함수
def route_query(query: str, verbose: bool = True) -> Dict[str, Any]:
    """간단한 Router 실행"""
    router = RouterAgent(verbose=verbose)
    return router.route(query)
