"""
LLM 답변 생성 시 참고할 데이터 스키마 가이드

각 카테고리(MAP, NPC, MONSTER, ITEM)별로 제공되는 상세 정보 필드를
LLM에게 명시하여 답변 품질을 향상시킵니다.
"""

SCHEMA_GUIDE = """
[데이터 스키마]
각 카테고리별 제공되는 상세 정보:

**MAP (맵)**:
- region: 소속 지역 (예: 빅토리아 아일랜드, 엘나스 산맥)
- adjacent_maps: 연결된 맵 리스트 (이동 가능한 인접 맵)
  * target_map: 연결된 맵 이름
  * direction: 포털 방향
- resident_npcs: 해당 맵에 있는 NPC 리스트
- resident_monsters: 해당 맵에 출현하는 몬스터 리스트

**NPC**:
- location: 정확한 위치 (예: 재즈바, 무기상점, 선착장)
- region: 소속 지역 (예: 커닝시티, 헤네시스)
- services: 제공 서비스 리스트 (예: 도적 전직, 무기 판매, 창고)
- npc_type: NPC 유형 (예: 전직관, 상인, 안내원)

**MONSTER (몬스터)**:
- level: 몬스터 레벨
- hp: 체력
- spawn_maps: 출현하는 맵 리스트
- drops: 드랍하는 아이템 리스트
  * item_name: 아이템 이름
  * drop_rate: 드랍 확률
- region: 주요 출현 지역

**ITEM (아이템)**:
- obtainable_from: 구매 가능한 NPC 리스트
- dropped_by: 드랍하는 몬스터 리스트
- price: 구매 가격 (메소)
- required_level: 착용 가능 레벨
- item_type: 아이템 종류 (예: WEAPON, ARMOR, CONSUMABLE)

[중요 지침]
1. **위 필드 정보가 제공되면 반드시 활용하세요**
   - location 필드가 있으면 → 구체적 위치 언급
   - adjacent_maps 필드가 있으면 → 이동 경로 안내
   - spawn_maps 필드가 있으면 → 출현 위치 명시
   - dropped_by 필드가 있으면 → 드랍 몬스터 안내

2. **없는 정보는 추측하지 마세요**
   - 제공되지 않은 필드는 "확인되지 않습니다"라고 명시

3. **관계 정보 우선 활용**
   - 검색 결과에 관계 정보가 있으면 우선적으로 사용
   - 예: NPC → MAP 관계, MONSTER → ITEM 드랍 관계
"""


# 카테고리별 필수 필드 (참고용)
CATEGORY_FIELDS = {
    "MAP": ["region", "adjacent_maps", "resident_npcs", "resident_monsters"],
    "NPC": ["location", "region", "services", "npc_type"],
    "MONSTER": ["level", "spawn_maps", "drops", "region"],
    "ITEM": ["obtainable_from", "dropped_by", "price", "required_level"]
}
