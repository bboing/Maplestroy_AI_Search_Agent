# MapleStory Hybrid RAG Search System

> 메이플스토리 게임 데이터 기반 자연어 Q&A 시스템
> 검색 전략을 7단계로 반복 개선하며 MRR +14%, Latency -57% 달성

---

## 프로젝트 개요

메이플스토리 NPC·몬스터·아이템·맵 데이터를 기반으로 자연어 질문에 답하는 **Hybrid RAG 검색 시스템**.
단순 키워드 검색이 아닌 PostgreSQL + Milvus + Neo4j 3개 DB를 조합하고,
LLM 기반 라우팅 전략을 7단계에 걸쳐 반복 설계·평가·개선한 과정을 담았습니다.

---

## 아키텍처

### 데이터 적재 파이프라인

```
maple_data.json
      │
      ▼
① import_data.py     → PostgreSQL  (Pydantic DTO 검증 · 카테고리별 저장)
      │
      ├─② sync_to_milvus.py → Milvus   (QAGenerator · HuggingFace 임베딩 384차원)
      │
      └─③ sync_to_neo4j.py → Neo4j    (EntityResolver · 노드/관계 생성)
```

### 검색 파이프라인 (최종: hybrid_searcher_fin.py)

```
사용자 질문
     │
     ▼
Router Agent (LLM)
  → hop 깊이 결정 (1 or 2+)
  → entities (명사) / sentences (동사구) 추출
     │
     ├── PostgreSQL(entities) ──┐
     │                          ├── RRF 융합
     └── Milvus(sentences)    ──┘
               │
               ▼ (hop ≥ 2일 때만)
            Neo4j
          PG 결과의 canonical_name으로 관계 탐색
               │
               ▼
        Jina Reranker v2
               │
               ▼
        Answer Generator (LLM)
               │
               ▼
           최종 자연어 답변
```

### DB 역할 분담

| DB | 역할 | 예시 쿼리 |
|----|------|----------|
| **PostgreSQL** | 엔티티 저장 · 키워드 검색 · canonical_name + synonym 관리 | `"다크로드"` 정확 매칭 |
| **Milvus** | 의미 기반 벡터 검색 (HNSW/COSINE) | `"물약 파는 사람"` 유사 검색 |
| **Neo4j** | 그래프 관계 탐색 (multi-hop) | `ITEM → MONSTER → MAP` 체인 |

---

## 기술 스택

| 분야 | 기술 |
|------|------|
| API 서버 | FastAPI (async) |
| UI | Streamlit |
| LLM | Ollama (local, llama3.1:8b) / Groq (cloud, fallback) |
| 키워드 검색 | PostgreSQL + SQLAlchemy AsyncSession |
| 벡터 검색 | Milvus + paraphrase-multilingual-MiniLM-L12-v2 (384차원) |
| 그래프 DB | Neo4j (Cypher) |
| 한국어 분석 | Kiwi (형태소, LLM fallback) |
| Reranker | Jina Reranker v2 |
| 비동기 | asyncio.gather |
| 컨테이너 | Docker Compose |
| 모니터링 | Langfuse |

---

## 검색 전략 진화 (Plan 1 → 7)

### 공통 핵심 기술

- **RRF (Reciprocal Rank Fusion)**: 소스별 점수 스케일 차이를 순위 기반으로 정규화
  `RRF_score(d) = Σ 1 / (k + rank_i(d))` (k=60)
- **asyncio.gather**: 다중 DB 비동기 병렬 실행
- **Entity/Sentence 분리**: 명사 → PostgreSQL, 동사구 → Milvus
- **canonical_name 해소**: 동의어 입력 → PG 표제어 추출 → Neo4j 정확 탐색

### Plan별 전략 요약

| Plan | 전략 | MRR | nDCG@10 | Latency |
|------|------|-----|---------|---------|
| P1 | LLM Multi-step Plan 생성 + Batch 병렬 | 0.645 | 0.441 | 6,859ms |
| P2 | Plan 제거 · 임계값 기반 조건부 Neo4j | 0.199 | 0.139 | 15ms |
| P3 | Intent 분류 · 관계 Intent시 Neo4j 병렬 | 0.240 | 0.153 | 391ms |
| P4 | LLM 키워드 추출(최대 3개) · 3DB 완전 병렬 · 차등 RRF | 0.213 | 0.159 | 858ms |
| P5 | Entity/Sentence 분리 · Plan 내 DB별 최적 입력 | 0.670 | 0.466 | 7,459ms |
| P6 | HOP 기반 라우팅 · Jina Reranker 도입 | 0.736 | 0.515 | 3,094ms |
| **P7 (fin)** | **HOP + canonical_name → Neo4j 탐색 강화** | **0.736** | **0.517** | **2,930ms** |

### 핵심 발견

- **P2~P4**: LLM Plan을 제거하고 단순화했더니 P1보다 정확도 급락 → 설계 전략이 DB 수보다 중요
- **P4**: Neo4j를 항상 호출해도 정확도 하락 → "더 많은 DB ≠ 더 좋은 결과"
- **P6**: HOP 조건부 호출 + Reranker 조합으로 처음으로 정확도·속도 동시 개선
- **명시적 가중치(P4)보다 쿼리 입력 분리(P5~P7)가 더 효과적**: Entity/Sentence 분리 자체가 자연스러운 가중치 역할

---

## 정량적 성과 (P1 → P7 기준)

| 지표 | P1 (Baseline) | P7 (FIN) | 개선율 |
|------|--------------|---------|--------|
| MRR | 0.645 | 0.736 | **+14%** |
| nDCG@10 | 0.441 | 0.517 | **+17%** |
| Recall@10 | 0.420 | 0.510 | **+21%** |
| Precision@5 | 0.226 | 0.252 | **+12%** |
| Latency | 6,859ms | 2,930ms | **-57%** |

> 평가셋: `scripts/evaluate_search.py` · `training/data/output_data/evaluation_report.json`

---

## 파일 구조

```
taegyun_nexon_pj/
├── docker-compose.yml                      # 로컬 개발 (FastAPI + Ollama)
├── docker-compose.prod.yml                 # 프로덕션 (Streamlit + Groq)
├── requirements.txt
├── .env.example
│
├── langchain_app/                          # FastAPI 백엔드
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── api/
│   │   ├── main.py                         # FastAPI 엔트리포인트
│   │   └── reranker_jina.py                # Jina Reranker v2 API 호출
│   ├── config/
│   │   └── settings.py                     # 환경 변수 / DB 연결 설정
│   ├── database/
│   │   ├── base.py                         # SQLAlchemy Base
│   │   ├── session.py                      # AsyncSession 팩토리
│   │   ├── neo4j_connection.py             # Neo4j 드라이버 초기화
│   │   ├── models/
│   │   │   └── maple_dictionary.py         # 단일 테이블 (canonical_name + JSONB)
│   │   └── schemas/
│   │       ├── npc_dto.py
│   │       ├── monster_dto.py
│   │       ├── item_dto.py
│   │       └── map_dto.py
│   └── src/
│       ├── agents/
│       │   ├── router_agent_fin.py         # 최종 라우터 (HOP + Entity/Sentence) — P7
│       │   ├── router_agent_hop.py         # HOP 라우터 — P6
│       │   ├── router_agent_sep.py         # SEP 라우터 (Plan 포함) — P5
│       │   └── router_agent.py             # Plan 기반 원형 — P1
│       ├── retrievers/
│       │   ├── hybrid_searcher_fin.py      # 최종 검색기 (canonical_name → Neo4j) — P7
│       │   ├── hybrid_searcher_hop.py      # HOP 조건부 Neo4j + Reranker — P6
│       │   ├── hybrid_searcher_sep.py      # Entity/Sentence 분리 + Plan — P5
│       │   ├── hybrid_searcher_option4.py  # 차등 RRF 가중치 — P4
│       │   ├── hybrid_searcher_option3.py  # RELATION_INTENT 조건 병렬 — P3
│       │   ├── hybrid_searcher_option2.py  # threshold 기반 조건부 Neo4j — P2
│       │   ├── hybrid_searcher.py          # LLM Plan 기반 원형 — P1
│       │   ├── db_searcher.py              # PostgreSQL 키워드 검색
│       │   ├── milvus_retriever.py         # Milvus 벡터 검색
│       │   ├── neo4j_searcher.py           # Neo4j 그래프 탐색
│       │   └── document_processor.py      # 문서 전처리
│       ├── generators/
│       │   ├── answer_generator.py         # LLM 자연어 답변 생성
│       │   └── schema_guide.py             # DB 스키마 프롬프트 가이드
│       ├── models/
│       │   ├── llm.py                      # Ollama / Groq LLM 초기화
│       │   ├── embeddings.py               # HuggingFace 임베딩 (384차원)
│       │   └── langfuse_callback.py        # Langfuse 모니터링 콜백
│       ├── chains/
│       │   └── rag_chain.py                # LangChain RAG 체인
│       └── utils/
│           ├── keyword_extractor.py        # Kiwi 형태소 + 동의어 처리
│           ├── qa_generator.py             # 평가셋 자동 생성 (QA pair)
│           └── query_transformer.py        # 쿼리 변환 유틸
│
├── streamlit_app/                          # 포트폴리오 데모 UI
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py                              # Streamlit 메인
│   ├── components/
│   │   ├── chat_interface.py               # 채팅 UI 컴포넌트
│   │   ├── sidebar.py                      # 사이드바 (모델 설정)
│   │   └── source_display.py              # 검색 출처 표시
│   └── services/
│       └── maple_rag_service.py            # FastAPI 호출 서비스 레이어
│
├── scripts/                                # 데이터 적재 · 평가 · 디버깅
│   ├── import_data.py                      # PostgreSQL 데이터 적재
│   ├── sync_to_milvus.py                   # Milvus 임베딩 동기화
│   ├── sync_to_neo4j.py                    # Neo4j 관계 그래프 구축
│   ├── evaluate_search.py                  # 버전별 성능 평가 (MRR·nDCG·Latency)
│   ├── init_all_databases.py               # 3DB 일괄 초기화
│   ├── health_check.py                     # 서비스 상태 확인
│   ├── neo4j_entity_resolver.py            # canonical_name 해소 로직
│   ├── neo4j_relationship_schema.py        # Neo4j 스키마 정의
│   ├── export_postgres_data.py
│   ├── check_neo4j_data.py
│   ├── check_neo4j_relations.py
│   ├── test_hybrid_search.py
│   ├── test_answer_generator.py
│   ├── test_keyword_extractor.py
│   ├── setup.sh / setup-dev.sh / setup-prod.sh
│   └── requirements.txt
│
└── training/
    └── data/
        ├── input_data/
        │   └── maple_data.json             # 메이플 지식 베이스 (NPC·몬스터·아이템·맵)
        ├── test/
        │   └── search_test_queries.json    # 평가용 쿼리셋
        ├── train.jsonl                     # 파인튜닝 학습 데이터 / 사용하지 않음(자원이 한정적임)
        ├── valid.jsonl                     # 파인튜닝 검증 데이터 / 사용하지 않음
        └── output_data/
            └── evaluation_report.json      # 버전별 평가 결과 (MRR·nDCG·Latency)
```

---

## 실행 환경 요구사항

### OS별 사전 준비

| OS | 필요 사항 |
|----|---------|
| **macOS** | Docker Desktop, Ollama |
| **Linux** | Docker Engine (`sudo systemctl start docker`), Ollama |
| **Windows** | Docker Desktop + **WSL2** 활성화 필수, WSL2 터미널에서 실행 |

> **Windows 사용자 주의**: `setup.sh` 등 셸 스크립트는 WSL2(Ubuntu) 터미널에서만 실행 가능합니다.
> PowerShell이나 CMD에서는 동작하지 않습니다.
>
> WSL2 설치: `wsl --install` (PowerShell 관리자 권한)

### GPU 지원

Jina Reranker는 아래 우선순위로 GPU를 자동 감지합니다.

```
MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
```

GPU가 없는 환경에서도 CPU로 자동 fallback되어 정상 동작합니다.

---

## 실행 방법

### 환경 변수 설정

```bash
# .env 파일
BIZ_POSTGRES_HOST=localhost
BIZ_POSTGRES_PORT=5432
BIZ_POSTGRES_DB=maple_npc_db
BIZ_POSTGRES_USER=postgres
BIZ_POSTGRES_PASSWORD=your_password

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

MILVUS_HOST=localhost
MILVUS_PORT=19530

# Groq API (Streamlit 데모용)
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL_NAME=llama-3.3-70b-versatile
```

### 의존성 설치

```bash
python3 -m venv nexon_venv
source nexon_venv/bin/activate
pip install -r requirements.txt
```

### 인프라 실행

```bash
# 로컬 개발 (FastAPI + 호스트 Ollama)
docker-compose up -d

# 포트폴리오 데모 (Streamlit + Groq API)
docker-compose -f docker-compose.prod.yml up -d
```

### 데이터 초기화

```bash
python scripts/import_data.py       # PostgreSQL 적재
python scripts/sync_to_milvus.py    # Milvus 벡터 동기화
python scripts/sync_to_neo4j.py     # Neo4j 관계 그래프 구축
```

### 접속

- FastAPI 문서: `http://localhost:8000/docs`
- Streamlit 데모: `http://localhost:8501`

---

## 테스트 질문 예시

```
"도적 전직 어디서 해?"
"스포아 어디서 잡아?"
"아이스진 구하는 방법"
"커닝시티에 어떤 NPC 있어?"
"리스항구 가는 법"
"얼음바지 드랍하는 몬스터"
```

---

## 한계점 및 개선 방향

**한계점**
- 평가셋이 템플릿 기반 자체 제작 → 실제 유저 질의 패턴과 괴리 가능
- MAP/NPC/ITEM/MONSTER 4개 카테고리만 포함 (직업·스킬·보스 미포함)
- HOP 분석·Entity/Sentence 분리 모두 LLM 응답에 의존

**개선 방향**
- 실제 유저 질의 로그 기반 평가셋 재구축
- 게임 패치 감지 → 자동 DB 동기화 파이프라인
- LLM 기반 Cross-encoder Reranker 실험
- BGE-M3 등 도메인 특화 임베딩 모델 파인튜닝

---

## 라이선스

MIT License © 2026 Taegyun Kim

---

## 서드파티 라이선스

| 라이브러리 / 모델 | 라이선스 | 출처 |
|------------------|---------|------|
| [jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) | CC-BY-NC-4.0 | Jina AI — 비상업적 연구·평가 목적 사용 |
| [kiwipiepy](https://github.com/bab2min/kiwipiepy) | LGPL v3 | bab2min |
| [sentence-transformers](https://github.com/UKPLab/sentence-transformers) | Apache 2.0 | UKPLab |
| [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | Apache 2.0 | sentence-transformers |
| [LangChain](https://github.com/langchain-ai/langchain) | MIT | LangChain AI |
| [FastAPI](https://github.com/fastapi/fastapi) | MIT | Sebastián Ramírez |
| [Streamlit](https://github.com/streamlit/streamlit) | Apache 2.0 | Streamlit Inc. |
| [Milvus](https://github.com/milvus-io/milvus) | Apache 2.0 | Zilliz |
| [Neo4j Python Driver](https://github.com/neo4j/neo4j-python-driver) | Apache 2.0 | Neo4j Inc. |
| [Ollama](https://github.com/ollama/ollama) | MIT | Ollama |
| [Langfuse](https://github.com/langfuse/langfuse) | MIT | Langfuse |
| [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy) | MIT | SQLAlchemy |
| [PyMilvus](https://github.com/milvus-io/pymilvus) | Apache 2.0 | Zilliz |