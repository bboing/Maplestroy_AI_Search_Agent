"""
Maple-Agent Streamlit App
Groq API 기반 하이브리드 RAG 데모
"""
import streamlit as st
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv(Path(__file__).parent.parent / ".env")

# 경로 추가
STREAMLIT_DIR = Path(__file__).parent
LANGCHAIN_DIR = STREAMLIT_DIR.parent / "langchain_app"

sys.path.insert(0, str(LANGCHAIN_DIR))
sys.path.insert(0, str(STREAMLIT_DIR))

from components.sidebar import render_sidebar
from components.chat_interface import render_chat_interface
from services.maple_rag_service import MapleRAGService


def main():
    """메인 앱"""
    # 페이지 설정
    st.set_page_config(
        page_title="Nexon Maple-Agent Demo",
        page_icon="🏹",
        layout="wide"
    )
    
    # 사이드바 (정보 표시용)
    render_sidebar()
    
    # 메인 타이틀
    st.title("🍁 Maple-Agent: 하이브리드 RAG 지식베이스")
    st.markdown("""
    이 데모는 **PostgreSQL, Milvus, Neo4j**를 결합하여 메이플스토리 지식을 답변하는 AI 에이전트입니다.
    
    - **PostgreSQL**: 정확한 키워드 매칭
    - **Milvus**: 의미 기반 벡터 검색
    - **Neo4j**: 그래프 관계 추론
    - **RRF**: Reciprocal Rank Fusion으로 최적 결과 융합
    - **LLM**: Ollama (local) 또는 Groq (cloud) 자동 선택
    """)
    
    # RAG 서비스 초기화 (환경변수에서 자동으로 LLM 설정)
    if "rag_service" not in st.session_state:
        with st.spinner("🔄 RAG 엔진 초기화 중..."):
            try:
                st.session_state.rag_service = MapleRAGService()
                st.success("✅ RAG 엔진 준비 완료!")
            except Exception as e:
                st.error(f"❌ 초기화 실패: {e}")
                st.stop()
    
    # 채팅 인터페이스
    render_chat_interface(st.session_state.rag_service)


if __name__ == "__main__":
    main()
