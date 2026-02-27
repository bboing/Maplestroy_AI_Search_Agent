# 검색 시스템 테스트셋

## 파일 구조

```
search_test_queries.json  # 20개 테스트 질문
```

## 테스트 케이스 형식

```json
{
  "id": 1,
  "category": "class_change",
  "query": "도적 전직 어디서 하나요?",
  "ground_truth": ["다크로드", "여섯갈래길"],
  "relevance": {
    "다크로드": 3,
    "여섯갈래길": 3,
    "넬라": 1
  },
  "expected_intent": "class_change"
}
```

### 필드 설명

- **id**: 테스트 케이스 번호
- **category**: 질문 유형 (class_change, item_drop, npc_info 등)
- **query**: 사용자 질문
- **ground_truth**: 정답 엔티티 이름 리스트
- **relevance**: 엔티티별 관련도 (0-3)
  - 3: 매우 관련 (핵심 정답)
  - 2: 관련있음 (부가 정보)
  - 1: 약간 관련
  - 0: 무관
- **expected_intent**: 예상 Intent

## 평가 지표

### MRR (Mean Reciprocal Rank)
첫 번째 정답의 순위 역수 평균

```
질문 1: 정답 1위 → 1/1 = 1.0
질문 2: 정답 3위 → 1/3 = 0.33
MRR = (1.0 + 0.33) / 2 = 0.665
```

### nDCG@K (Normalized Discounted Cumulative Gain)
순위를 고려한 정확도 (높을수록 좋음)

### Precision@K
상위 K개 중 정답 비율

### Recall@K
전체 정답 중 상위 K개에 포함된 비율

## 사용법

### 1. 단일 시스템 평가

```bash
# Plan 모드 평가
python scripts/evaluate_search.py --plan --verbose

# 기본 Hybrid 평가
python scripts/evaluate_search.py --verbose
```

### 2. 시스템 비교

```bash
# Plan vs 기본 Hybrid 비교
python scripts/evaluate_search.py --mode compare
```

### 3. Docker에서 실행

```bash
docker exec ai-langchain-api python scripts/evaluate_search.py --plan
```

## 예상 결과

```
==========================================
평가 결과 요약
==========================================
MRR:          0.7500
nDCG@10:      0.8200
nDCG@5:       0.7800
Precision@5:  0.6800
Recall@10:    0.9200
==========================================
```

## 테스트 케이스 추가

새로운 질문 추가 시:

1. `search_test_queries.json`에 케이스 추가
2. `ground_truth` 정확히 지정 (DB에 있는 이름)
3. `relevance` 점수 부여 (3: 핵심, 2: 관련, 1: 약간)
4. 평가 스크립트 재실행

## 카테고리 목록

- `class_change`: 전직 관련
- `item_drop`: 아이템 드랍
- `npc_info`: NPC 정보
- `map_npc`: 맵의 NPC
- `monster_location`: 몬스터 위치
- `item_purchase`: 아이템 구매
- `npc_location`: NPC 위치
- `map_info`: 맵 정보
- `monster_info`: 몬스터 정보
- `hunting_ground`: 사냥터 추천
- `complex`: 복합 질문
