"""
Q&A 생성기
PostgreSQL 데이터 → 질문+답변 쌍 생성 (템플릿 기반, 할루시네이션 제거)
"""
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class QAGenerator:
    """
    템플릿 기반 Q&A 생성
    카테고리별로 다양한 질문 패턴 생성
    """
    
    def generate_qa_pairs(self, entity: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        엔티티에서 Q&A 쌍 생성
        
        Returns:
            [
                {"question": "...", "answer": "...", "qa_type": "..."},
                ...
            ]
        """
        category = entity.get('category')
        
        if category == 'ITEM':
            return self._generate_item_qa(entity)
        elif category == 'NPC':
            return self._generate_npc_qa(entity)
        elif category == 'MAP':
            return self._generate_map_qa(entity)
        elif category == 'MONSTER':
            return self._generate_monster_qa(entity)
        else:
            return []
    
    def _generate_item_qa(self, entity: Dict[str, Any]) -> List[Dict[str, str]]:
        """아이템 Q&A 생성"""
        name = entity['canonical_name']
        synonyms = entity.get('synonyms', [])
        detail = entity.get('detail_data', {})
        # detail_data.description 우선 사용 (더 자세함)
        description = detail.get('description') or entity.get('description', '')
        
        qa_pairs = []
        
        # Q1: 기본 정보
        qa_pairs.append({
            "question": f"{name}은 무엇인가요?",
            "answer": f"{name}은(는) {description}",
            "qa_type": "item_info"
        })
        
        # Q2: 동의어
        if synonyms:
            qa_pairs.append({
                "question": f"{name}의 다른 이름은?",
                "answer": f"{name}은(는) {', '.join(synonyms[1:])}(으)로도 불립니다.",
                "qa_type": "item_synonym"
            })
        
        # Q3: 구매처
        if 'obtainable_from' in detail and detail['obtainable_from']:
            sources = detail['obtainable_from']
            qa_pairs.append({
                "question": f"{name}은 어디서 구매할 수 있나요?",
                "answer": f"{name}은(는) {', '.join(sources)}에게서 구매할 수 있습니다.",
                "qa_type": "item_purchase"
            })
            
            # 변형 질문
            qa_pairs.append({
                "question": f"{name} 사려면 어디로 가야 하나요?",
                "answer": f"{name}을(를) 사려면 {sources[0]}(을)를 찾아가세요.",
                "qa_type": "item_purchase"
            })
        
        # Q4: 가격
        if 'price' in detail and detail['price']:
            qa_pairs.append({
                "question": f"{name} 가격은 얼마인가요?",
                "answer": f"{name}의 가격은 {detail['price']}메소입니다.",
                "qa_type": "item_price"
            })
        
        # Q5: 아이템 타입
        if 'item_type' in detail:
            type_desc = {
                'WEAPON': '무기',
                'ARMOR': '방어구',
                'EQUIPMENT': '장비',
                'CONSUMABLE': '소비 아이템',
                'CASH': '캐시 아이템'
            }.get(detail['item_type'], detail['item_type'])
            
            qa_pairs.append({
                "question": f"{name}은 어떤 종류의 아이템인가요?",
                "answer": f"{name}은(는) {type_desc} 아이템입니다.",
                "qa_type": "item_type"
            })
        
        # Q6: 필요 레벨
        required_level = detail.get('required_level', 0)
        if required_level and required_level > 0:
            qa_pairs.append({
                "question": f"{name}은 몇 레벨부터 착용할 수 있나요?",
                "answer": f"{name}은(는) 레벨 {required_level}부터 착용 가능합니다.",
                "qa_type": "item_requirement"
            })
        
        # Q7: 드랍 몬스터 (dropped_by)
        dropped_by = detail.get('dropped_by')
        if dropped_by:
            # 문자열 또는 리스트 처리
            if isinstance(dropped_by, str):
                monsters = [dropped_by]
            elif isinstance(dropped_by, list):
                monsters = dropped_by
            else:
                monsters = []
            
            if monsters:
                monster_list = ', '.join(monsters[:3])  # 최대 3개
                qa_pairs.append({
                    "question": f"{name}을 어디서 얻을 수 있나요?",
                    "answer": f"{name}은(는) {monster_list} 등이 드랍합니다.",
                    "qa_type": "item_drop_source"
                })
                
                # 변형 질문
                qa_pairs.append({
                    "question": f"{name} 어디서 나와?",
                    "answer": f"{name}은(는) {monster_list}을(를) 잡으면 얻을 수 있습니다.",
                    "qa_type": "item_drop_source"
                })
                
                # 각 몬스터별 개별 질문
                for monster_name in monsters[:3]:
                    qa_pairs.append({
                        "question": f"{monster_name}을 잡으면 {name}을 얻을 수 있나요?",
                        "answer": f"네, {monster_name}을(를) 잡으면 {name}을(를) 드랍합니다.",
                        "qa_type": "item_drop_confirmation"
                    })
        
        return qa_pairs
    
    def _generate_npc_qa(self, entity: Dict[str, Any]) -> List[Dict[str, str]]:
        """NPC Q&A 생성"""
        name = entity['canonical_name']
        detail = entity.get('detail_data', {})
        # detail_data.description 우선 사용 (더 자세함)
        description = detail.get('description') or entity.get('description', '')
        
        qa_pairs = []
        
        # Q1: 기본 정보
        qa_pairs.append({
            "question": f"{name}은 누구인가요?",
            "answer": f"{name}은(는) {description}",
            "qa_type": "npc_info"
        })
        
        # Q2: 위치
        location = detail.get('location')
        if location:  # None이 아닐 때만
            qa_pairs.append({
                "question": f"{name}은 어디에 있나요?",
                "answer": f"{name}은(는) {location}에 있습니다.",
                "qa_type": "npc_location"
            })
            
            qa_pairs.append({
                "question": f"{name}을 만나려면 어디로 가야 하나요?",
                "answer": f"{name}을(를) 만나려면 {location}(으)로 가세요.",
                "qa_type": "npc_location"
            })
        
        # Q3: 서비스
        if 'services' in detail and detail['services']:
            services = ', '.join(detail['services'])
            qa_pairs.append({
                "question": f"{name}은 무엇을 해주나요?",
                "answer": f"{name}은(는) {services} 서비스를 제공합니다.",
                "qa_type": "npc_service"
            })
            
            qa_pairs.append({
                "question": f"{name}에게서 뭘 할 수 있나요?",
                "answer": f"{name}에게서는 {services}을(를) 할 수 있습니다.",
                "qa_type": "npc_service"
            })
        
        # Q4: 판매 아이템
        if 'sells_items' in detail and detail['sells_items']:
            items = ', '.join(detail['sells_items'][:3])
            qa_pairs.append({
                "question": f"{name}은 무엇을 판매하나요?",
                "answer": f"{name}은(는) {items} 등을 판매합니다.",
                "qa_type": "npc_sells"
            })
        
        return qa_pairs
    
    def _generate_map_qa(self, entity: Dict[str, Any]) -> List[Dict[str, str]]:
        """맵 Q&A 생성"""
        name = entity['canonical_name']
        detail = entity.get('detail_data', {})
        # detail_data.description 우선 사용 (더 자세함)
        description = detail.get('description') or entity.get('description', '')
        
        qa_pairs = []
        
        # Q1: 기본 정보
        qa_pairs.append({
            "question": f"{name}은 어떤 곳인가요?",
            "answer": f"{name}은(는) {description}",
            "qa_type": "map_info"
        })
        
        # Q2: 지역
        region = detail.get('region')
        if region:  # None이 아닐 때만
            qa_pairs.append({
                "question": f"{name}은 어느 지역에 있나요?",
                "answer": f"{name}은(는) {region} 지역에 위치합니다.",
                "qa_type": "map_location"
            })
        
        # Q3: NPC
        if 'resident_npcs' in detail and detail['resident_npcs']:
            npcs = ', '.join(detail['resident_npcs'][:3])
            qa_pairs.append({
                "question": f"{name}에는 어떤 NPC가 있나요?",
                "answer": f"{name}에는 {npcs} 등이 있습니다.",
                "qa_type": "map_npcs"
            })
            
            qa_pairs.append({
                "question": f"{name}에서 뭘 할 수 있나요?",
                "answer": f"{name}에서는 {npcs} NPC를 만나 다양한 활동을 할 수 있습니다.",
                "qa_type": "map_activities"
            })
        
        # Q4: 몬스터
        if 'resident_monsters' in detail and detail['resident_monsters']:
            monsters = ', '.join(detail['resident_monsters'][:3])
            qa_pairs.append({
                "question": f"{name}에는 어떤 몬스터가 나오나요?",
                "answer": f"{name}에는 {monsters} 등의 몬스터가 출몰합니다.",
                "qa_type": "map_monsters"
            })
            
            qa_pairs.append({
                "question": f"{name}에서 사냥할 수 있나요?",
                "answer": f"네, {name}에서는 {monsters} 등을 사냥할 수 있습니다.",
                "qa_type": "map_hunting"
            })
        
        # Q5: 추천 레벨
        level_range = detail.get('recommended_level_range')
        if level_range and isinstance(level_range, list) and len(level_range) == 2:
            qa_pairs.append({
                "question": f"{name}은 몇 레벨에게 적합한가요?",
                "answer": f"{name}은(는) 레벨 {level_range[0]}~{level_range[1]} 구간에 적합합니다.",
                "qa_type": "map_level"
            })
        
        return qa_pairs
    
    def _generate_monster_qa(self, entity: Dict[str, Any]) -> List[Dict[str, str]]:
        """몬스터 Q&A 생성"""
        name = entity['canonical_name']
        detail = entity.get('detail_data', {})
        # detail_data.description 우선 사용 (더 자세함)
        description = detail.get('description') or entity.get('description', '')
        
        qa_pairs = []
        
        # Q1: 기본 정보
        qa_pairs.append({
            "question": f"{name}은 어떤 몬스터인가요?",
            "answer": f"{name}은(는) {description}",
            "qa_type": "monster_info"
        })
        
        # Q2: 레벨
        level = detail.get('level')
        if level is not None:  # 0일 수도 있으니 is not None 사용
            qa_pairs.append({
                "question": f"{name}은 몇 레벨 몬스터인가요?",
                "answer": f"{name}은(는) 레벨 {level} 몬스터입니다.",
                "qa_type": "monster_level"
            })
        
        # Q3: 체력
        hp = detail.get('hp')
        if hp:  # None이 아닐 때만
            qa_pairs.append({
                "question": f"{name}의 체력은 얼마인가요?",
                "answer": f"{name}의 체력은 {hp}입니다.",
                "qa_type": "monster_hp"
            })
        
        # Q4: 스폰 위치
        if 'spawn_maps' in detail and detail['spawn_maps']:
            maps = ', '.join(detail['spawn_maps'])
            qa_pairs.append({
                "question": f"{name}은 어디서 잡을 수 있나요?",
                "answer": f"{name}은(는) {maps}에서 잡을 수 있습니다.",
                "qa_type": "monster_location"
            })
            
            qa_pairs.append({
                "question": f"{name} 사냥터는 어디인가요?",
                "answer": f"{name} 사냥터는 {maps}입니다.",
                "qa_type": "monster_location"
            })
        
        # Q5: 드랍 아이템
        if 'drops' in detail and detail['drops']:
            drop_items = [d.get('item_name', '') for d in detail['drops'] if d.get('item_name')]
            if drop_items:
                items = ', '.join(drop_items[:3])
                qa_pairs.append({
                    "question": f"{name}은 무엇을 드랍하나요?",
                    "answer": f"{name}은(는) {items} 등을 드랍합니다.",
                    "qa_type": "monster_drops"
                })
        
        # Q6: 속성
        element = detail.get('element')
        if element:  # None이 아닐 때만
            element_kr = {
                'FIRE': '불',
                'ICE': '얼음',
                'LIGHTNING': '번개',
                'POISON': '독',
                'HOLY': '신성',
                'DARK': '어둠',
                'UNDEAD': '언데드'
            }.get(element, element)
            
            qa_pairs.append({
                "question": f"{name}의 속성은 무엇인가요?",
                "answer": f"{name}은(는) {element_kr} 속성 몬스터입니다.",
                "qa_type": "monster_element"
            })
        
        return qa_pairs
    
    def create_embedding_text(self, qa: Dict[str, str]) -> str:
        """
        Q&A를 임베딩용 텍스트로 변환
        
        전략: Question + Answer 합치기
        → 양방향 검색 가능 (Q로도, A로도)
        """
        question = qa['question']
        answer = qa['answer']
        
        # 포맷: "[Q] 질문 [A] 답변"
        return f"[Q] {question}\n[A] {answer}"
    
    def generate_batch(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        여러 엔티티에서 Q&A 배치 생성
        
        Returns:
            [
                {
                    "entity_id": "uuid",
                    "entity_name": "아이스진",
                    "entity_type": "ITEM",
                    "question": "...",
                    "answer": "...",
                    "qa_type": "...",
                    "embedding_text": "[Q] ... [A] ..."
                },
                ...
            ]
        """
        all_qa = []
        
        for entity in entities:
            entity_id = entity.get('id')
            entity_name = entity.get('canonical_name')
            entity_type = entity.get('category')
            
            # Q&A 생성
            qa_pairs = self.generate_qa_pairs(entity)
            
            for qa in qa_pairs:
                all_qa.append({
                    "entity_id": str(entity_id),
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "question": qa['question'],
                    "answer": qa['answer'],
                    "qa_type": qa['qa_type'],
                    "embedding_text": self.create_embedding_text(qa)
                })
        
        logger.info(f"✅ Q&A 생성 완료: {len(entities)}개 엔티티 → {len(all_qa)}개 Q&A")
        return all_qa


# 편의 함수
def generate_qa_from_entity(entity: Dict[str, Any]) -> List[Dict[str, str]]:
    """단일 엔티티 Q&A 생성"""
    generator = QAGenerator()
    return generator.generate_qa_pairs(entity)
