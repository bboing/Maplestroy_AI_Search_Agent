"""
Source Display Component
검색 출처 및 근거 표시
"""
import streamlit as st


def display_sources(sources: list, search_results: list, entities: list = None, sentences: list = None, query: str = "", confidence: float = None):
    """
    검색 출처 및 결과 표시

    Args:
        sources: 사용된 데이터 소스 리스트 (["PostgreSQL", "Milvus", "Neo4j"])
        search_results: 검색 결과 상세
        entities: Router가 추출한 키워드 (명사)
        sentences: Router가 추출한 문장 (동사구)
        query: 원본 사용자 쿼리
        confidence: 최종 신뢰도 점수
    """
    with st.expander("🔍 답변 근거 (Retrieval Sources)", expanded=False):
        # 데이터 소스 표시
        st.markdown("#### 📊 사용된 데이터 소스")
        source_cols = st.columns((len(sources) + (1 if confidence is not None else 0)) or 1)

        for idx, source in enumerate(sources):
            with source_cols[idx]:
                if source == "PostgreSQL":
                    st.success(f"✅ {source}")
                elif source == "Milvus":
                    st.info(f"🔵 {source}")
                elif source == "Neo4j":
                    st.warning(f"🟡 {source}")
                else:
                    st.write(f"📁 {source}")

        if confidence is not None:
            with source_cols[len(sources)]:
                st.metric("최종 신뢰도", f"{confidence:.1f}%")

        cols = source_cols  # 하위 호환
        
        for idx, source in enumerate(sources):
            with cols[idx]:
                if source == "PostgreSQL":
                    st.success(f"✅ {source}")
                elif source == "Milvus":
                    st.info(f"🔵 {source}")
                elif source == "Neo4j":
                    st.warning(f"🟡 {source}")
                else:
                    st.write(f"📁 {source}")
        
        st.divider()

        # 🔎 쿼리 분석 정보
        st.markdown("#### 🔎 쿼리 분석")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**추출 키워드 (Entities)**")
            if entities:
                st.code(", ".join(entities))
            else:
                st.caption("없음")

        with col2:
            st.markdown("**검색 문장 (Sentences)**")
            if sentences:
                st.code(", ".join(sentences))
            else:
                st.caption(f"(없음 → 원문 사용: {query})" if query else "없음")

        st.divider()

        # canonical_name 목록
        if search_results:
            canonical_names = [
                r.get("data", {}).get("canonical_name", "")
                for r in search_results[:5]
                if r.get("data", {}).get("canonical_name")
            ]
            if canonical_names:
                st.markdown("**검색된 Canonical Names**")
                st.code(" | ".join(canonical_names))
            st.divider()

        # 검색 결과 상세
        if search_results:
            st.markdown("#### 📝 검색된 항목")
            
            for idx, result in enumerate(search_results[:5], 1):
                data = result.get("data", {})
                score = result.get("score", 0)
                match_type = result.get("match_type", "unknown")
                
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{idx}. {data.get('canonical_name', 'Unknown')}** ({data.get('category', 'N/A')})")
                        
                        # 설명 (최대 100자)
                        description = data.get('description', '')
                        if len(description) > 100:
                            description = description[:100] + "..."
                        st.caption(description)
                    
                    with col2:
                        st.metric("Reranker 점수", f"{score:.1f}", help=f"Match Type: {match_type}")
                
                if idx < len(search_results[:5]):
                    st.divider()
