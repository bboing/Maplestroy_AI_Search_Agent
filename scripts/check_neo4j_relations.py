#!/usr/bin/env python3
"""
Neo4j 관계 확인 스크립트
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "langchain_app"))

# load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from database.neo4j_connection import neo4j_conn

def check_map_relations():
    """MAP → NPC, MAP → MONSTER 관계 확인"""
    
    print("=" * 80)
    print("🔍 Neo4j MAP 관계 확인")
    print("=" * 80)
    
    # MAP → NPC 관계
    print("\n1️⃣ MAP -[HAS_NPC]-> NPC 관계:")
    query1 = """
    MATCH (m:MAP)-[r:HAS_NPC]->(n:NPC)
    RETURN m.name AS map_name, n.name AS npc_name
    LIMIT 10
    """
    results1 = neo4j_conn.execute_query(query1)
    
    if results1:
        for record in results1:
            print(f"   {record['map_name']} → {record['npc_name']}")
        print(f"   ... 총 {len(results1)}개 (최대 10개 표시)")
    else:
        print("   ❌ HAS_NPC 관계 없음!")
    
    # MAP → MONSTER 관계
    print("\n2️⃣ MAP -[HAS_MONSTER]-> MONSTER 관계:")
    query2 = """
    MATCH (m:MAP)-[r:HAS_MONSTER]->(mon:MONSTER)
    RETURN m.name AS map_name, mon.name AS monster_name
    LIMIT 10
    """
    results2 = neo4j_conn.execute_query(query2)
    
    if results2:
        for record in results2:
            print(f"   {record['map_name']} → {record['monster_name']}")
        print(f"   ... 총 {len(results2)}개 (최대 10개 표시)")
    else:
        print("   ❌ HAS_MONSTER 관계 없음!")
    
    # 전체 관계 타입 통계
    print("\n3️⃣ 전체 관계 타입 통계:")
    query3 = """
    MATCH ()-[r]->()
    RETURN type(r) AS relation_type, count(*) AS count
    ORDER BY count DESC
    """
    results3 = neo4j_conn.execute_query(query3)
    
    if results3:
        for record in results3:
            print(f"   {record['relation_type']}: {record['count']}개")
    else:
        print("   ❌ 관계 없음!")
    
    print("\n" + "=" * 80)
    
    neo4j_conn.close()

if __name__ == "__main__":
    check_map_relations()
