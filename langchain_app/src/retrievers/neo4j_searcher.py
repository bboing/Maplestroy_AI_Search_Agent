"""
Neo4j Graph Searcher - 관계 기반 검색 (Async)
"""
from typing import List, Dict, Any, Optional
from database.neo4j_connection import async_neo4j_conn
import logging

logger = logging.getLogger(__name__)


class Neo4jSearcher:
    """
    Neo4j 관계 검색기 (Async)
    
    주요 기능:
    - NPC → MAP 위치 찾기
    - MONSTER → MAP 출현 지역 찾기
    - ITEM → NPC 판매처 찾기
    - ITEM → MONSTER 드랍처 찾기
    - MAP → MAP 이동 경로 찾기
    """
    
    def __init__(self):
        self.conn = async_neo4j_conn
    
    async def find_npc_location(self, npc_name: str) -> List[Dict[str, Any]]:
        """NPC가 위치한 맵 찾기"""
        query = """
        MATCH (npc:NPC)-[:LOCATED_IN]->(map:MAP)
        WHERE npc.name CONTAINS $npc_name
        RETURN npc.name as npc_name, map.name as map_name, 
               npc.id as npc_id, map.id as map_id
        """
        
        try:
            results = await self.conn.execute_query(query, {"npc_name": npc_name})
            return [
                {
                    "npc_name": record["npc_name"],
                    "map_name": record["map_name"],
                    "npc_id": record["npc_id"],
                    "map_id": record["map_id"],
                    "relation_type": "LOCATED_IN"
                }
                for record in results
            ]
        except Exception as e:
            logger.error(f"Neo4j NPC 위치 검색 실패: {e}")
            return []
    
    async def find_monster_locations(self, monster_name: str) -> List[Dict[str, Any]]:
        """몬스터가 출현하는 맵 찾기"""
        query = """
        MATCH (monster:MONSTER)-[:SPAWNS_IN]->(map:MAP)
        WHERE monster.name CONTAINS $monster_name
        RETURN monster.name as monster_name, map.name as map_name,
               monster.id as monster_id, map.id as map_id
        """
        
        try:
            results = await self.conn.execute_query(query, {"monster_name": monster_name})
            return [
                {
                    "monster_name": record["monster_name"],
                    "map_name": record["map_name"],
                    "monster_id": record["monster_id"],
                    "map_id": record["map_id"],
                    "relation_type": "SPAWNS_IN"
                }
                for record in results
            ]
        except Exception as e:
            logger.error(f"Neo4j 몬스터 위치 검색 실패: {e}")
            return []
    
    async def find_item_sellers(self, item_name: str) -> List[Dict[str, Any]]:
        """아이템을 판매하는 NPC 찾기"""
        query = """
        MATCH (npc:NPC)-[:SELLS]->(item:ITEM)
        WHERE item.name CONTAINS $item_name
        RETURN npc.name as npc_name, item.name as item_name,
               npc.id as npc_id, item.id as item_id
        """
        
        try:
            results = await self.conn.execute_query(query, {"item_name": item_name})
            return [
                {
                    "npc_name": record["npc_name"],
                    "item_name": record["item_name"],
                    "npc_id": record["npc_id"],
                    "item_id": record["item_id"],
                    "relation_type": "SELLS"
                }
                for record in results
            ]
        except Exception as e:
            logger.error(f"Neo4j 아이템 판매처 검색 실패: {e}")
            return []
    
    async def find_item_droppers(self, item_name: str) -> List[Dict[str, Any]]:
        """아이템을 드랍하는 몬스터 찾기"""
        query = """
        MATCH (monster:MONSTER)-[:DROPS]->(item:ITEM)
        WHERE item.name CONTAINS $item_name
        RETURN monster.name as monster_name, item.name as item_name,
               monster.id as monster_id, item.id as item_id
        """

        try:
            results = await self.conn.execute_query(query, {"item_name": item_name})
            return [
                {
                    "monster_name": record["monster_name"],
                    "item_name": record["item_name"],
                    "monster_id": record["monster_id"],
                    "item_id": record["item_id"],
                    "relation_type": "DROPS"
                }
                for record in results
            ]
        except Exception as e:
            logger.error(f"Neo4j 아이템 드랍처 검색 실패: {e}")
            return []

    async def find_item_droppers_with_location(self, item_name: str) -> List[Dict[str, Any]]:
        """아이템을 드랍하는 몬스터 + 그 몬스터의 서식지(맵)까지 2-hop 검색

        ITEM → MONSTER → MAP 체인 관계
        예: "얼음바지" → "스포아" → "리스항구 해변"
        """
        query = """
        MATCH (monster:MONSTER)-[:DROPS]->(item:ITEM)
        WHERE item.name CONTAINS $item_name
        OPTIONAL MATCH (monster)-[:SPAWNS_IN]->(map:MAP)
        RETURN monster.name as monster_name, item.name as item_name,
               map.name as map_name,
               monster.id as monster_id, item.id as item_id, map.id as map_id
        """

        try:
            results = await self.conn.execute_query(query, {"item_name": item_name})
            return [
                {
                    "monster_name": record["monster_name"],
                    "item_name": record["item_name"],
                    "map_name": record["map_name"],
                    "monster_id": record["monster_id"],
                    "item_id": record["item_id"],
                    "map_id": record["map_id"],
                    "relation_type": "DROPS+SPAWNS_IN"
                }
                for record in results
            ]
        except Exception as e:
            logger.error(f"Neo4j 아이템→몬스터→맵 2-hop 검색 실패: {e}")
            return []
    
    async def find_map_connections(self, map_name: str) -> List[Dict[str, Any]]:
        """맵의 연결된 맵들 찾기"""
        query = """
        MATCH (map1:MAP)-[:CONNECTS_TO]->(map2:MAP)
        WHERE map1.name CONTAINS $map_name
        RETURN map1.name as from_map, map2.name as to_map,
               map1.id as from_id, map2.id as to_id
        """
        
        try:
            results = await self.conn.execute_query(query, {"map_name": map_name})
            return [
                {
                    "from_map": record["from_map"],
                    "to_map": record["to_map"],
                    "from_id": record["from_id"],
                    "to_id": record["to_id"],
                    "relation_type": "CONNECTS_TO"
                }
                for record in results
            ]
        except Exception as e:
            logger.error(f"Neo4j 맵 연결 검색 실패: {e}")
            return []
    
    async def find_map_npcs(self, map_name: str) -> List[Dict[str, Any]]:
        """맵에 있는 NPC들 찾기"""
        query = """
        MATCH (map:MAP)-[:HAS_NPC]->(npc:NPC)
        WHERE map.name CONTAINS $map_name
        RETURN map.name as map_name, npc.name as npc_name,
               map.id as map_id, npc.id as npc_id
        """
        
        try:
            results = await self.conn.execute_query(query, {"map_name": map_name})
            return [
                {
                    "map_name": record["map_name"],
                    "npc_name": record["npc_name"],
                    "map_id": record["map_id"],
                    "npc_id": record["npc_id"],
                    "relation_type": "HAS_NPC"
                }
                for record in results
            ]
        except Exception as e:
            logger.error(f"Neo4j 맵 NPC 검색 실패: {e}")
            return []
    
    async def find_map_monsters(self, map_name: str) -> List[Dict[str, Any]]:
        """맵에 출현하는 몬스터들 찾기"""
        query = """
        MATCH (map:MAP)-[:HAS_MONSTER]->(monster:MONSTER)
        WHERE map.name CONTAINS $map_name
        RETURN map.name as map_name, monster.name as monster_name,
               map.id as map_id, monster.id as monster_id
        """
        
        try:
            results = await self.conn.execute_query(query, {"map_name": map_name})
            return [
                {
                    "map_name": record["map_name"],
                    "monster_name": record["monster_name"],
                    "map_id": record["map_id"],
                    "monster_id": record["monster_id"],
                    "relation_type": "HAS_MONSTER"
                }
                for record in results
            ]
        except Exception as e:
            logger.error(f"Neo4j 맵 몬스터 검색 실패: {e}")
            return []
    
    async def find_path_between_maps(
        self, 
        start_map: str, 
        end_map: str, 
        max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        """두 맵 사이의 최단 경로 찾기"""
        query = """
        MATCH path = shortestPath(
            (start:MAP)-[:CONNECTS_TO*..5]-(end:MAP)
        )
        WHERE start.name CONTAINS $start_map
          AND end.name CONTAINS $end_map
        RETURN [node in nodes(path) | node.name] as path_names,
               length(path) as distance
        LIMIT 1
        """
        
        try:
            results = await self.conn.execute_query(query, {
                "start_map": start_map,
                "end_map": end_map
            })
            
            if results:
                record = results[0]
                return [{
                    "path": record["path_names"],
                    "distance": record["distance"],
                    "relation_type": "PATH"
                }]
            return []
        except Exception as e:
            logger.error(f"Neo4j 경로 검색 실패: {e}")
            return []
    
    async def search_by_relation(
        self, 
        entity_name: str, 
        relation_type: str
    ) -> List[Dict[str, Any]]:
        """범용 관계 검색 (동적 쿼리)"""
        # 관계 타입에 따라 적절한 함수 호출
        relation_map = {
            "LOCATED_IN": self.find_npc_location,
            "SPAWNS_IN": self.find_monster_locations,
            "SELLS": self.find_item_sellers,
            "DROPS": self.find_item_droppers,
            "CONNECTS_TO": self.find_map_connections,
            "HAS_NPC": self.find_map_npcs,
            "HAS_MONSTER": self.find_map_monsters,
        }
        
        func = relation_map.get(relation_type.upper())
        if func:
            return await func(entity_name)
        
        logger.warning(f"지원하지 않는 관계 타입: {relation_type}")
        return []
