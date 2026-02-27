"""
Sidebar Component
설정 및 정보 표시
"""
import streamlit as st
import os


def render_sidebar():
    """
    사이드바 렌더링
    """
    st.sidebar.title("⚙️ 설정")
    
    # LLM 상태 표시
    groq_api_key = os.getenv("GROQ_API_KEY")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    st.sidebar.markdown("### 🤖 LLM 엔진")
    if groq_api_key:
        st.sidebar.success("✅ Groq API 사용 가능")
    else:
        st.sidebar.info("ℹ️ Groq API 미설정")
    
    st.sidebar.info(f"🦙 Ollama: `{ollama_url}`")
    st.sidebar.caption("(자동으로 Ollama → Groq fallback)")
    
    # 구분선
    st.sidebar.divider()
    
    # 고급 설정 (옵션)
    with st.sidebar.expander("⚙️ 고급 설정"):
        st.session_state.max_results = st.slider(
            "최대 검색 결과",
            min_value=3,
            max_value=10,
            value=5,
            help="검색할 최대 항목 수"
        )
    
    # 정보
    st.sidebar.divider()
    st.sidebar.markdown("""
    ### 📚 사용 가능한 데이터
    - 맵 (MAP): 14개
    - NPC: 9개
    - 몬스터 (MONSTER): 8개
    - 아이템 (ITEM): 3개
    
    ### 💡 예시 질문
    - "도적 전직 어디서?"
    - "스포아 어디서 잡아?"
    - "아이스진 어디서 구해?"
    """)
