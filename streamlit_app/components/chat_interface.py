"""
Chat Interface Component
"""
import streamlit as st
from components.source_display import display_sources


def render_chat_interface(rag_service):
    """
    채팅 인터페이스 렌더링
    
    Args:
        rag_service: MapleRAGService 인스턴스
    """
    # 메시지 히스토리 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 기존 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 어시스턴트 메시지면 출처 표시
            if message["role"] == "assistant" and "sources" in message:
                display_sources(
                    message["sources"],
                    message["search_results"],
                    entities=message.get("entities", []),
                    sentences=message.get("sentences", []),
                    query=message.get("query", ""),
                )
    
    # 사용자 입력
    if prompt := st.chat_input("무엇이 궁금하신가요? (예: 아이스진은 어디서 얻을 수 있어?)"):
        # 사용자 메시지 추가
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 어시스턴트 응답
        with st.chat_message("assistant"):
            with st.spinner("🔍 지식 베이스 탐색 중..."):
                try:
                    # RAG 엔진 호출
                    result = rag_service.query(
                        prompt,
                        max_results=st.session_state.get("max_results", 5)
                    )
                    
                    # 답변 표시
                    st.markdown(result["answer"])
                    
                    # 출처 표시
                    display_sources(
                        result["sources"],
                        result["search_results"],
                        entities=result.get("entities", []),
                        sentences=result.get("sentences", []),
                        query=prompt,
                    )

                    # 메시지 히스토리 저장
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "search_results": result["search_results"],
                        "confidence": result["confidence"],
                        "entities": result.get("entities", []),
                        "sentences": result.get("sentences", []),
                        "query": prompt,
                    })
                    
                except Exception as e:
                    st.error(f"❌ 오류 발생: {e}")
                    st.exception(e)
