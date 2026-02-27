"""
Query Transformer for Milvus
사용자 질문 → Milvus 최적화 쿼리 변환 (HyDE + Feature Expansion)
"""
from typing import List, Optional
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class QueryTransformer:
    """
    사용자 질문을 Milvus 검색에 최적화된 형태로 변환
    
    전략 1: Feature-Based Expansion (특징 확장)
    전략 2: HyDE (가상 답변 생성)
    """
    
    FEATURE_EXPANSION_PROMPT = """당신은 검색 최적화 전문가입니다.
사용자의 질문을 Milvus(벡터 데이터베이스)에서 검색하기 가장 적합한 **'설명형 문장'**으로 변환하세요.

**변환 규칙:**
1. 질문의 핵심 대상(Entity)과 유저의 의도(Intent)를 파악합니다.
2. 해당 대상이 가질법한 **상세 특징과 환경**을 묘사하는 문장으로 확장합니다.
3. 불필요한 수식어("알려줘", "부탁해")는 제거합니다.

**예시:**
- 입력: "도적에게 좋은 사냥터"
- 출력: "도적 직업이 사냥하기 좋은 효율적인 필드. 지형이 복잡하지 않고 몬스터의 밀집도가 높으며 경험치 효율이 좋은 장소."

- 입력: "초보자 추천 장비"
- 출력: "초보자가 착용하기 적합한 저렴한 방어구나 무기. 낮은 레벨에서 구매 가능하고 방어력이나 공격력이 적당한 장비."

**질문:** {question}
**핵심 엔티티:** {entities}

**출력 (검색용 설명 텍스트만, 한 문장으로):**"""

    HYDE_PROMPT = """당신은 메이플스토리 백과사전 작성자입니다.
사용자의 질문에 대해, 백과사전에 적혀 있을 법한 **가상의 정답 2-3문장**을 작성하세요.

**중요:** 
- 사실이 아니어도 괜찮습니다 (검색용이므로)
- 백과사전 스타일로 작성
- 구체적이고 상세하게

**예시:**
질문: "도적 사냥터 추천"
백과사전: "도적 직업은 커닝시티 근처의 필드에서 사냥하기 적합합니다. 이 지역의 몬스터들은 도적의 빠른 공격 속도에 취약하며, 경험치 효율이 좋습니다."

**질문:** {question}
**핵심 엔티티:** {entities}

**백과사전 답변:**"""

    def __init__(
        self,
        llm: Optional[ChatOllama] = None,
        strategy: str = "hybrid",  # "expansion", "hyde", "hybrid"
        verbose: bool = False
    ):
        """
        Args:
            llm: ChatOllama 인스턴스
            strategy: "expansion", "hyde", "hybrid"
            verbose: 로그 출력
        """
        self.llm = llm
        self.strategy = strategy
        self.verbose = verbose
        
        # LLM 없으면 기본 생성 (transformer 전용 경량 모델)
        if self.llm is None and strategy in ["hyde", "hybrid"]:
            self.llm = ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,  # ✅ 환경변수
                model=settings.OLLAMA_MODEL,  # ✅ 환경변수
                temperature=0.3  # 약간의 창의성
            )
    
    def transform(
        self,
        question: str,
        entities: Optional[List[str]] = None
    ) -> str:
        """
        질문을 Milvus 검색용으로 변환
        
        Args:
            question: 사용자 질문
            entities: Router에서 추출한 엔티티 (선택)
            
        Returns:
            변환된 검색 쿼리
        """
        if entities is None:
            entities = []
        
        entity_str = ", ".join(entities) if entities else "없음"
        
        if self.verbose:
            print(f"\n🔄 Query Transform ({self.strategy})")
            print(f"   원본: {question}")
            print(f"   엔티티: {entity_str}")
        
        try:
            if self.strategy == "expansion":
                result = self._feature_expansion(question, entity_str)
            elif self.strategy == "hyde":
                result = self._hyde_transform(question, entity_str)
            elif self.strategy == "hybrid":
                result = self._hybrid_transform(question, entity_str)
            else:
                # 기본: 그대로 사용
                result = question
            
            if self.verbose:
                print(f"   변환: {result[:100]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Query transformation failed: {e}")
            # 실패 시 원본 반환
            return question
    
    def _feature_expansion(self, question: str, entities: str) -> str:
        """특징 기반 확장"""
        if not self.llm:
            return question
        
        prompt = self.FEATURE_EXPANSION_PROMPT.format(
            question=question,
            entities=entities
        )
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    
    def _hyde_transform(self, question: str, entities: str) -> str:
        """HyDE: 가상 답변 생성"""
        if not self.llm:
            return question
        
        prompt = self.HYDE_PROMPT.format(
            question=question,
            entities=entities
        )
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    
    def _hybrid_transform(self, question: str, entities: str) -> str:
        """
        Hybrid: Feature Expansion + HyDE 조합
        
        1. Feature Expansion으로 특징 추출
        2. HyDE로 가상 답변 생성
        3. 둘을 합침
        """
        if not self.llm:
            return question
        
        # 짧은 프롬프트로 한 번에 처리 (효율적)
        hybrid_prompt = f"""메이플스토리 검색 시스템입니다.
사용자 질문을 백과사전 답변 스타일로 변환하세요.

**변환 규칙:**
1. 질문의 핵심을 파악
2. 백과사전에 적혀있을 법한 2-3문장 작성
3. 구체적 특징 포함 (위치, 레벨, 가격 등)

**질문:** {question}
**핵심 엔티티:** {entities}

**백과사전 스타일 텍스트:**"""
        
        response = self.llm.invoke([HumanMessage(content=hybrid_prompt)])
        return response.content.strip()


# 편의 함수
def transform_query(
    question: str,
    entities: Optional[List[str]] = None,
    strategy: str = "hybrid"
) -> str:
    """
    간단한 쿼리 변환 함수
    
    Usage:
        transformed = transform_query("도적 사냥터 추천", ["도적", "사냥터"])
    """
    transformer = QueryTransformer(strategy=strategy)
    return transformer.transform(question, entities)
