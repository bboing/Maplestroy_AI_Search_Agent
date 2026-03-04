"""
Answer Generator - 검색 결과를 자연어 답변으로 생성
"""
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from src.models.llm import create_llm, switch_to_groq
from .schema_guide import SCHEMA_GUIDE
import logging

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    검색 결과 → 자연어 답변 생성
    
    역할:
    1. 검색 결과 Context 정리
    2. LLM Prompt 생성
    3. 자연어 답변 생성
    """
    
    SYSTEM_PROMPT = f"""당신은 메이플스토리 전문 가이드입니다.

[역할]
- 검색 결과를 바탕으로 사용자 질문에 정확하게 답변합니다.
- 검색된 정보만 사용하며, 없는 정보는 지어내지 않습니다.
- **반드시 한국어로만 답변합니다. 일본어, 영어 등 다른 언어 사용 금지.**

{SCHEMA_GUIDE}

[답변 규칙]
1. **검색 결과에 있는 정보는 충분히 제공**
   - MAP(위치) 질문: 장소명 + 지역 + 가는 방법(있으면)
   - NPC 질문: 이름 + 위치 + 역할(있으면)
   - ITEM 질문: 획득 방법 + 확률(있으면)
   - MONSTER 질문: 이름 + 특징(있으면)
   
2. **없는 정보는 솔직하게**
   - 검색 결과에 질문의 직접적인 답이 없으면 "해당 정보는 데이터에 없습니다"라고 답변
   - 검색 결과에서 유추하거나 추측하여 답변하지 말 것
   - NPC 목록만 있고 특정 NPC 역할이 없으면 "확인되지 않습니다"라고 답변

3. **정확성**
   - NPC 위치는 "위치:" 필드 값 그대로 사용
   - 드랍 확률은 백분율로 표시
   - 검색 결과에 명시되지 않은 내용은 절대 추측하지 말기

[답변 스타일]
- 자연스럽고 도움이 되는 답변 (2-3문장)
- 핵심 정보 우선, 보조 정보 추가
- 친절하지만 간결하게

[금지 사항]
- 없는 정보를 "없다"고 언급하지 말 것
- 질문과 전혀 관련 없는 정보 추가하지 말 것"""

    def __init__(
        self,
        llm=None,
        verbose: bool = False,
    ):
        self.llm = llm if llm else create_llm(temperature=0.3)
        self.verbose = verbose

    def _switch_to_groq(self):
        """Runtime에 Ollama 실패 시 Groq으로 전환"""
        result = switch_to_groq(temperature=0.3)
        if result:
            self.llm = result
    
    async def generate(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        max_context_items: int = 5
    ) -> Dict[str, Any]:
        """
        검색 결과로 답변 생성 (async)
        
        Args:
            query: 사용자 질문
            search_results: Hybrid Searcher 결과
            max_context_items: Context에 포함할 최대 항목 수
            
        Returns:
            {
                "answer": str,
                "sources": List[str],
                "confidence": float
            }
        """
        if self.verbose:
            print(f"\n💬 Answer Generator: '{query}'")
        
        # 1. Context 구축
        context = self._build_context(search_results, max_context_items)
        
        if not context:
            return {
                "answer": "죄송합니다. 관련 정보를 찾을 수 없습니다.",
                "sources": [],
                "confidence": 0.0
            }
        
        # 2. Prompt 생성
        prompt = self._create_prompt(query, context)
        
        if self.verbose:
            print(f"   📝 Context: {len(context)}개 항목")
        
        # 3. LLM 답변 생성 (async)
        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            
            # ainvoke로 비동기 호출
            response = await self.llm.ainvoke(messages)
            answer = response.content.strip()
            
            # 4. Source 정리
            sources = self._extract_sources(search_results[:max_context_items])
            
            # 5. 신뢰도 계산 (평균 점수)
            confidence = self._calculate_confidence(search_results[:max_context_items])

            if confidence < 60.0:
                return {
                    "answer": "죄송합니다. 정확한 답변을 찾지 못했습니다.(정확도 60% 이하) 데이터 추가 전까지 기대해 주세요!",
                    "sources": sources,
                    "confidence": confidence
                }
            
            if self.verbose:
                print(f"   ✅ 답변 생성 완료 (신뢰도: {confidence:.1f}%)")
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            # Ollama Runtime 에러 시 Groq으로 전환 후 재시도
            if "not found" in str(e) or "404" in str(e) or "Connection" in str(e):
                self._switch_to_groq()
                try:
                    messages = [
                        SystemMessage(content=self.SYSTEM_PROMPT),
                        HumanMessage(content=prompt)
                    ]
                    response = await self.llm.ainvoke(messages)
                    answer = response.content.strip()
                    sources = self._extract_sources(search_results[:max_context_items])
                    confidence = self._calculate_confidence(search_results[:max_context_items])
                    
                    return {
                        "answer": answer,
                        "sources": sources,
                        "confidence": confidence
                    }
                except Exception as retry_error:
                    logger.error(f"Groq 재시도도 실패: {retry_error}")
            
            return {
                "answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def _build_context(
        self,
        search_results: List[Dict[str, Any]],
        max_items: int
    ) -> List[Dict[str, Any]]:
        """
        검색 결과를 Context로 정리
        
        정리 전략:
        1. 상위 N개만 선택
        2. 중복 제거
        3. 핵심 정보만 추출
        """
        context = []
        seen_ids = set()
        
        for result in search_results[:max_items]:
            data = result.get("data", {})
            entity_id = data.get("id")
            
            # 중복 제거
            if entity_id and entity_id in seen_ids:
                continue
            
            if entity_id:
                seen_ids.add(entity_id)
            
            # Context 항목 생성
            category = data.get("category", "Unknown")
            detail_data = data.get("detail_data", {})
            
            context_item = {
                "name": data.get("canonical_name", "Unknown"),
                "category": category,
                "description": data.get("description", ""),
                "score": result.get("score", 0),
                "match_type": result.get("match_type", "unknown"),
                "sources": result.get("sources", [])
            }
            
            # 카테고리별 상세 정보 추가
            if category == "MAP" and detail_data:
                context_item["region"] = detail_data.get("region")
                context_item["bgm"] = detail_data.get("bgm", [])
                context_item["adjacent_maps"] = detail_data.get("adjacent_maps", [])
                context_item["special_portals"] = detail_data.get("special_portals", [])
                context_item["resident_npcs"] = detail_data.get("resident_npcs", [])
                context_item["resident_monsters"] = detail_data.get("resident_monsters", [])
            elif category == "NPC" and detail_data:
                context_item["location"] = detail_data.get("location")
                context_item["region"] = detail_data.get("region")
                context_item["services"] = detail_data.get("services", [])
            elif category == "MONSTER" and detail_data:
                context_item["level"] = detail_data.get("level")
                context_item["hp"] = detail_data.get("hp")
                context_item["mp"] = detail_data.get("mp")
                context_item["exp"] = detail_data.get("exp")
                context_item["region"] = detail_data.get("region")
                context_item["spawn_maps"] = detail_data.get("spawn_maps", [])
                context_item["drops"] = detail_data.get("drops", [])
            elif category == "ITEM" and detail_data:
                context_item["obtainable_from"] = detail_data.get("obtainable_from", [])
                context_item["dropped_by"] = detail_data.get("dropped_by", [])
            
            # 관계 정보 추가 (Neo4j 결과)
            if "relation_info" in data:
                context_item["relation"] = data["relation_info"]
            
            context.append(context_item)
        
        return context
    
    def _create_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        LLM Prompt 생성
        """
        # Context를 읽기 쉽게 포맷팅
        context_parts = []
        for idx, item in enumerate(context):
            name = item['name']
            category = item['category']
            score = item['score']
            description = item['description'][:200] if item['description'] else ""
            
            part = f"[{idx+1}] {name} ({category}) - {score:.0f}점\n"
            if description:
                part += f"설명: {description}\n"
            
            # 카테고리별 추가 정보
            if category == "MAP":
                if item.get('region'):
                    part += f"지역: {item['region']}\n"
                if item.get('adjacent_maps'):
                    adjacent_str = ", ".join([m.get('target_map', '') for m in item['adjacent_maps'][:3] if m.get('target_map')])
                    if adjacent_str:
                        part += f"연결된 맵: {adjacent_str}\n"
                if item.get('resident_npcs'):
                    # NPC 목록 전체 표시 (최대 10개)
                    npc_list = ', '.join(item['resident_npcs'][:10])
                    npc_count = len(item['resident_npcs'])
                    if npc_count > 10:
                        part += f"거주 NPC ({npc_count}개 중 10개): {npc_list}\n"
                    else:
                        part += f"거주 NPC: {npc_list}\n"
            elif category == "NPC":
                if item.get('location'):
                    part += f"위치: {item['location']}\n"
                if item.get('region'):
                    part += f"지역: {item['region']}\n"
            elif category == "MONSTER":
                if item.get('level'):
                    part += f"레벨: {item['level']}\n"
                if item.get('spawn_maps'):
                    part += f"출현 위치: {', '.join(item['spawn_maps'][:3])}\n"
                if item.get('drops'):
                    # 드랍 아이템 정보 (아이템명 + 확률)
                    drops_list = []
                    for drop in item['drops'][:5]:  # 최대 5개
                        item_name = drop.get('item_name', '')
                        drop_rate = drop.get('drop_rate', 0)
                        # 확률을 백분율로 변환 (0.001 → 0.1%)
                        drop_rate_percent = drop_rate * 100
                        if item_name:
                            drops_list.append(f"{item_name} ({drop_rate_percent:.2f}%)")
                    if drops_list:
                        part += f"드랍 아이템: {', '.join(drops_list)}\n"
            elif category == "ITEM":
                if item.get('obtainable_from'):
                    part += f"구매처: {', '.join(item['obtainable_from'][:3])}\n"
                if item.get('dropped_by'):
                    part += f"드랍 몬스터: {', '.join(item['dropped_by'][:3])}\n"
            
            # Neo4j 관계 정보
            if 'relation' in item:
                part += f"관계: {item['relation']}\n"
            
            context_parts.append(part.strip())
        
        context_text = "\n\n".join(context_parts)
        
        prompt = f"""[검색 결과]

{context_text}

[사용자 질문]
{query}

[답변 지침]
위 검색 결과를 바탕으로 사용자 질문에 도움이 되는 답변을 제공하세요.

**답변 원칙:**
1. **있는 정보는 충분히 제공**: 위치, 이름, 레벨, 드랍률 등 관련 정보를 포함하세요
2. **없는 정보는 솔직하게**: 검색 결과에 직접적인 답이 없으면 "해당 정보는 데이터에 없습니다"라고 답변하세요
3. **추측 금지**: 검색 결과에서 유추하거나 연관 지어 답변하지 마세요. NPC 목록이 있어도 특정 NPC의 역할이 명시되지 않으면 답변하지 마세요
4. **정확하게**: 검색 결과의 필드 값을 정확히 읽어서 사용하세요

**좋은 답변 예시:**
- 질문: "도적 전직 어디?" → "커닝시티 뒷골목의 다크로드를 찾아가면 됩니다."
- 질문: "페리온 NPC?" → "페리온에는 주먹펴고 일어서, 이얀, 만지, 리버, 소피아 등이 있습니다."
- 질문: "스포아 드랍?" → "스포아는 아이스진을 0.10% 확률로 드랍합니다."
- 질문: "리스항구 방어구상점 NPC?" → "해당 정보는 데이터에 없습니다."

**나쁜 답변 예시:**
- "재즈바 지하로 가야 합니다." (너무 짧음, NPC 이름이나 추가 정보 없음)
- "리스항구 방어구상점에는 페이슨이 있습니다." (NPC 목록에서 추측한 잘못된 답변)"""
        
        return prompt
    
    def _extract_sources(self, search_results: List[Dict[str, Any]]) -> List[str]:
        """
        검색 결과에서 출처 추출
        - Neo4j 또는 Milvus가 최종 답변에 기여한 경우 PostgreSQL 제거
          (PostgreSQL은 entity 해소 역할만 담당)
        """
        all_sources = set()
        for result in search_results:
            # Milvus 폴백 결과는 소스 계산 제외 (실제 답변 기여 아님)
            if result.get("is_milvus_fallback"):
                continue
            for source in result.get("sources", []):
                all_sources.add(source)

        if "Neo4j" in all_sources or "Milvus" in all_sources:
            all_sources.discard("PostgreSQL")

        # 순서 보장: PostgreSQL → Milvus → Neo4j
        return [s for s in ["PostgreSQL", "Milvus", "Neo4j"] if s in all_sources]
    
    def _calculate_confidence(self, search_results: List[Dict[str, Any]]) -> float:
        """
        신뢰도 계산 (최고 점수 기준)
        - PostgreSQL direct match가 있으면 reranker 점수 무시하고 80% 고정
        - 그 외에는 최고 reranker 점수 사용
        """
        if not search_results:
            return 0.0

        for result in search_results:
            if "PostgreSQL" in result.get("sources", []) and result.get("match_type") == "direct":
                return 80.0

        return min(max(r.get("score", 0) for r in search_results), 100.0)
