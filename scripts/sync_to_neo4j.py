#!/usr/bin/env python3
"""
PostgreSQL → Neo4j 데이터 동기화
엔티티 간 관계를 Neo4j에 구축
+ EntityResolver로 자동 매핑
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "langchain_app"))

# load .env to path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from database.session import SessionLocal
from database.models.maple_dictionary import MapleDictionary
from database.neo4j_connection import neo4j_conn
from neo4j_entity_resolver import EntityResolver
from neo4j_relationship_schema import RelationshipSchema
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jSyncer:
    """PostgreSQL → Neo4j 동기화"""
    
    def __init__(self):
        self.db = SessionLocal()
        self.neo4j = neo4j_conn
        self.resolver = EntityResolver()  # Entity Resolver
        self.failed_relations = []  # 실패한 관계 추적
    
    def sync_all(self, skip_clear: bool = False):
        """
        전체 동기화
        
        Args:
            skip_clear: True면 기존 데이터 삭제 건너뜀 (증분 추가만)
        """
        logger.info("=" * 80)
        logger.info("🔄 Neo4j 동기화 시작")
        if skip_clear:
            logger.info("⚠️  증분 모드: 기존 데이터 유지")
        logger.info("=" * 80)
        
        # 1. 기존 데이터 삭제 (옵션)
        if not skip_clear:
            self.clear_neo4j()
        
        # 2. 노드 생성 (MERGE 사용으로 중복 방지)
        self.create_nodes()
        
        # 3. 관계 생성 (MERGE 사용으로 중복 방지)
        self.create_relationships()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Neo4j 동기화 완료!")
        logger.info("=" * 80)
    
    def clear_neo4j(self):
        """Neo4j 데이터 초기화"""
        logger.info("\n🗑️  기존 데이터 삭제 중...")
        
        query = "MATCH (n) DETACH DELETE n"
        self.neo4j.execute_write(query)
        
        logger.info("✅ Neo4j 초기화 완료")
    
    def create_nodes(self):
        """노드 생성"""
        logger.info("\n📦 노드 생성 중...")
        
        entities = self.db.query(MapleDictionary).all()
        
        for entity in entities:
            self._create_node(entity)
        
        logger.info(f"✅ {len(entities)}개 노드 생성 완료")
    
    def _create_node(self, entity: MapleDictionary):
        """단일 노드 생성 (MERGE 사용으로 중복 방지)"""
        # Category Enum → 문자열 변환
        if hasattr(entity.category, 'value'):
            category = entity.category.value  # Enum.value
        else:
            category = str(entity.category).split('.')[-1]  # "CategoryEnum.MAP" → "MAP"
        
        # MERGE로 변경 (없으면 생성, 있으면 업데이트)
        query = f"""
        MERGE (n:{category} {{name: $name}})
        ON CREATE SET
            n.id = $id,
            n.category = $category,
            n.description = $description,
            n.created_at = timestamp()
        ON MATCH SET
            n.id = $id,
            n.description = $description,
            n.updated_at = timestamp()
        """
        
        parameters = {
            "id": str(entity.id),
            "name": entity.canonical_name,
            "category": category,
            "description": entity.description or ""
        }
        
        self.neo4j.execute_write(query, parameters)
    
    def create_relationships(self):
        """관계 생성"""
        logger.info("\n🔗 관계 생성 중...")
        
        entities = self.db.query(MapleDictionary).all()
        
        relationship_count = 0
        
        for entity in entities:
            relationship_count += self._create_entity_relationships(entity)
        
        logger.info(f"✅ {relationship_count}개 관계 생성 완료")
    
    def _create_entity_relationships(self, entity: MapleDictionary) -> int:
        """
        엔티티 관계 생성 (Schema 기반)
        """
        # Category 처리
        if hasattr(entity.category, 'value'):
            category = entity.category.value
        else:
            category = str(entity.category).split('.')[-1]
        
        detail_data = entity.detail_data or {}
        count = 0
        
        # Schema에서 관계 추출
        relationships = RelationshipSchema.extract_all_relationships(
            category, 
            detail_data
        )
        
        # 관계 생성
        for rel in relationships:
            target_name = rel["target_name"]
            target_category = rel["target_category"]
            relation_type = rel["relation_type"]
            reverse = rel.get("reverse", False)
            bidirectional = rel.get("bidirectional", False)

            # 노드 존재 확인
            if not self.resolver.exists(target_name, target_category):
                continue  # 노드 없으면 스킵

            # 역방향 관계 처리
            if reverse:
                # target -> source 방향
                created = self._create_relationship(
                    source_id=None,  # target이 source
                    source_category=target_category,
                    target_name=entity.canonical_name,
                    target_category=category,
                    relation_type=relation_type,
                    source_name=target_name  # 이름으로 찾기
                )
            else:
                # source -> target 방향 (기본)
                created = self._create_relationship(
                    source_id=str(entity.id),
                    source_category=category,
                    target_name=target_name,
                    target_category=target_category,
                    relation_type=relation_type
                )

                # 양방향이면 역방향도 추가 생성
                if created and bidirectional and self.resolver.exists(entity.canonical_name, category):
                    self._create_relationship(
                        source_id=None,
                        source_category=target_category,
                        target_name=entity.canonical_name,
                        target_category=category,
                        relation_type=relation_type,
                        source_name=target_name  # target → source (역방향)
                    )

            if created:
                count += 1
        
        return count
    
    def _create_relationship(
        self,
        source_id: str,
        source_category: str,
        target_name: str,
        target_category: str,
        relation_type: str,
        source_name: str = None
    ) -> bool:
        """
        범용 관계 생성
        
        Args:
            source_id: 출발 노드 ID (또는 None)
            source_category: 출발 노드 카테고리
            target_name: 도착 노드 이름
            target_category: 도착 노드 카테고리
            relation_type: 관계 타입
            source_name: 출발 노드 이름 (ID 대신 사용 시)
            
        Returns:
            성공 여부
        """
        # source_name이 있으면 이름으로 찾기 (역방향용)
        if source_name:
            query = f"""
            MATCH (source:{source_category} {{name: $source_name}})
            MATCH (target:{target_category} {{name: $target_name}})
            MERGE (source)-[:{relation_type}]->(target)
            """
            parameters = {
                "source_name": source_name,
                "target_name": target_name
            }
        else:
            # 기본: ID로 찾기
            query = f"""
            MATCH (source:{source_category} {{id: $source_id}})
            MATCH (target:{target_category} {{name: $target_name}})
            MERGE (source)-[:{relation_type}]->(target)
            """
            parameters = {
                "source_id": source_id,
                "target_name": target_name
            }
        
        try:
            self.neo4j.execute_write(query, parameters)
            return True
        except Exception as e:
            logger.debug(
                f"관계 생성 실패: {source_category}-[{relation_type}]->{target_category}({target_name})"
            )
            return False
    
    # ========================================
    # 아래는 레거시 메서드 (호환성 유지)
    # ========================================
    
    def _create_npc_location(self, npc_id: str, location: str) -> int:
        """NPC → MAP (LOCATED_IN)"""
        # 1. MAP 노드 존재 확인
        if not self.resolver.exists(location, "MAP"):
            self.failed_relations.append({
                "type": "NPC_LOCATION",
                "reason": f"MAP '{location}' 노드 없음"
            })
            return 0
        
        # 2. 관계 생성 (MERGE로 중복 방지)
        query = """
        MATCH (npc:NPC {id: $npc_id})
        MATCH (map:MAP {name: $location})
        MERGE (npc)-[:LOCATED_IN]->(map)
        """
        
        try:
            self.neo4j.execute_write(query, {
                "npc_id": str(npc_id),
                "location": location
            })
            return 1
        except Exception as e:
            logger.debug(f"NPC 위치 관계 생성 실패: {e}")
            self.failed_relations.append({
                "type": "NPC_LOCATION",
                "reason": str(e)
            })
            return 0
    
    def _create_npc_sells_item(self, npc_id: str, item_name: str) -> int:
        """NPC → ITEM (SELLS)"""
        # ITEM 노드 존재 확인
        if not self.resolver.exists(item_name, "ITEM"):
            return 0  # 조용히 스킵 (드랍 아이템 많아서 로그 방지)
        
        query = """
        MATCH (npc:NPC {id: $npc_id})
        MATCH (item:ITEM {name: $item_name})
        MERGE (npc)-[:SELLS]->(item)
        """
        
        try:
            self.neo4j.execute_write(query, {
                "npc_id": str(npc_id),
                "item_name": item_name
            })
            return 1
        except Exception as e:
            logger.debug(f"NPC 판매 관계 생성 실패: {item_name}")
            return 0
    
    def _create_monster_spawns_in_map(self, monster_id: str, map_name: str) -> int:
        """MONSTER → MAP (SPAWNS_IN)"""
        # MAP 노드 존재 확인
        if not self.resolver.exists(map_name, "MAP"):
            return 0
        
        query = """
        MATCH (monster:MONSTER {id: $monster_id})
        MATCH (map:MAP {name: $map_name})
        MERGE (monster)-[:SPAWNS_IN]->(map)
        """
        
        try:
            self.neo4j.execute_write(query, {
                "monster_id": str(monster_id),
                "map_name": map_name
            })
            return 1
        except Exception as e:
            logger.debug(f"몬스터 출현 관계 생성 실패: {map_name}")
            return 0
    
    def _create_monster_drops_item(self, monster_id: str, item_name: str) -> int:
        """MONSTER → ITEM (DROPS)"""
        # ITEM 노드 존재 확인
        if not self.resolver.exists(item_name, "ITEM"):
            return 0  # 조용히 스킵
        
        query = """
        MATCH (monster:MONSTER {id: $monster_id})
        MATCH (item:ITEM {name: $item_name})
        MERGE (monster)-[:DROPS]->(item)
        """
        
        try:
            self.neo4j.execute_write(query, {
                "monster_id": str(monster_id),
                "item_name": item_name
            })
            return 1
        except Exception as e:
            logger.debug(f"몬스터 드랍 관계 생성 실패: {item_name}")
            return 0
    
    def _create_map_connects_to_map(self, map_id: str, target_map: str) -> int:
        """MAP → MAP (CONNECTS_TO)"""
        # 대상 MAP 노드 존재 확인
        if not self.resolver.exists(target_map, "MAP"):
            return 0
        
        query = """
        MATCH (map1:MAP {id: $map_id})
        MATCH (map2:MAP {name: $target_map})
        MERGE (map1)-[:CONNECTS_TO]->(map2)
        """
        
        try:
            self.neo4j.execute_write(query, {
                "map_id": str(map_id),
                "target_map": target_map
            })
            return 1
        except Exception as e:
            logger.debug(f"맵 연결 관계 생성 실패: {target_map}")
            return 0
    
    def _create_map_has_npc(self, map_id: str, npc_name: str) -> int:
        """MAP → NPC (HAS_NPC)"""
        # NPC 노드 존재 확인
        if not self.resolver.exists(npc_name, "NPC"):
            return 0
        
        query = """
        MATCH (map:MAP {id: $map_id})
        MATCH (npc:NPC {name: $npc_name})
        MERGE (map)-[:HAS_NPC]->(npc)
        """
        
        try:
            self.neo4j.execute_write(query, {
                "map_id": str(map_id),
                "npc_name": npc_name
            })
            return 1
        except Exception as e:
            logger.debug(f"맵-NPC 관계 생성 실패: {npc_name}")
            return 0
    
    def _create_map_has_monster(self, map_id: str, monster_name: str) -> int:
        """MAP → MONSTER (HAS_MONSTER)"""
        # MONSTER 노드 존재 확인
        if not self.resolver.exists(monster_name, "MONSTER"):
            return 0
        
        query = """
        MATCH (map:MAP {id: $map_id})
        MATCH (monster:MONSTER {name: $monster_name})
        MERGE (map)-[:HAS_MONSTER]->(monster)
        """
        
        try:
            self.neo4j.execute_write(query, {
                "map_id": str(map_id),
                "monster_name": monster_name
            })
            return 1
        except Exception as e:
            logger.debug(f"맵-몬스터 관계 생성 실패: {monster_name}")
            return 0
    
    def close(self):
        """연결 종료"""
        # 실패한 관계 리포트
        if self.failed_relations:
            logger.warning(f"\n⚠️  관계 생성 실패: {len(self.failed_relations)}개")
            for i, fail in enumerate(self.failed_relations[:10], 1):
                logger.warning(f"   {i}. {fail['type']}: {fail['reason']}")
            if len(self.failed_relations) > 10:
                logger.warning(f"   ... 외 {len(self.failed_relations) - 10}개")
        
        self.db.close()
        self.resolver.close()
        self.neo4j.close()


def main():
    """
    사용법:
      python sync_to_neo4j.py              # 전체 초기화 후 동기화
      python sync_to_neo4j.py --no-clear   # 기존 데이터 유지하며 추가
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="PostgreSQL → Neo4j 동기화")
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="기존 Neo4j 데이터를 삭제하지 않고 유지 (증분 모드)"
    )
    args = parser.parse_args()
    
    syncer = Neo4jSyncer()
    
    try:
        syncer.sync_all(skip_clear=args.no_clear)
    except KeyboardInterrupt:
        logger.info("\n\n중단됨")
    except Exception as e:
        logger.error(f"\n❌ 오류 발생: {e}", exc_info=True)
    finally:
        syncer.close()


if __name__ == "__main__":
    main()
