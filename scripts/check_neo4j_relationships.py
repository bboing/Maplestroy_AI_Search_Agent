#!/usr/bin/env python3
"""Neo4j 관계 상세 확인"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "langchain_app"))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from database.neo4j_connection import Neo4jConnection

def check_relationships():
    neo4j = Neo4jConnection()
    
    try:
        # 1. 전체 관계 개수
        result = neo4j.execute_query("MATCH ()-[r]->() RETURN count(r) as total")
        total = result[0]["total"]
        print(f"\n🔗 전체 관계 개수: {total}")
        
        if total == 0:
            print("\n❌ 관계가 하나도 없습니다!")
            print("sync_to_neo4j.py의 관계 생성 로직에 문제가 있습니다.")
            return
        
        # 2. 관계 타입별 개수
        result = neo4j.execute_query("""
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(r) as count
            ORDER BY count DESC
        """)
        print("\n📊 관계 타입별:")
        for record in result:
            print(f"  - {record['rel_type']}: {record['count']}개")
        
        # 3. 관계 샘플 (처음 10개)
        result = neo4j.execute_query("""
            MATCH (a)-[r]->(b)
            RETURN a.name as from_node, type(r) as rel_type, b.name as to_node
            LIMIT 10
        """)
        print("\n🔍 관계 샘플 (처음 10개):")
        for record in result:
            print(f"  {record['from_node']} --[{record['rel_type']}]--> {record['to_node']}")
        
        # 4. 노틸러스 관련 관계
        result = neo4j.execute_query("""
            MATCH (n)-[r]-(m)
            WHERE n.name CONTAINS '노틸러스' OR m.name CONTAINS '노틸러스'
            RETURN n.name as node1, type(r) as rel_type, m.name as node2
        """)
        print(f"\n🎯 노틸러스 관련 관계 ({len(result)}개):")
        for record in result:
            print(f"  {record['node1']} <--[{record['rel_type']}]--> {record['node2']}")
            
    finally:
        neo4j.close()

if __name__ == "__main__":
    check_relationships()
