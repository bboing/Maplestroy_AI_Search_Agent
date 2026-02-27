#!/usr/bin/env python3
"""Neo4j 데이터 상태 확인"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "langchain_app"))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from database.neo4j_connection import Neo4jConnection

def check_data():
    """Neo4j 데이터 상태 확인"""
    neo4j = Neo4jConnection()
    
    try:
        # 전체 노드 개수
        result = neo4j.execute_query("MATCH (n) RETURN count(n) as total")
        total_nodes = result[0]["total"]
        print(f"\n📊 전체 노드 개수: {total_nodes}")
        
        # 카테고리별 노드 개수
        result = neo4j.execute_query("""
            MATCH (n)
            RETURN labels(n)[0] as category, count(n) as count
            ORDER BY count DESC
        """)
        print("\n📁 카테고리별 노드:")
        for record in result:
            print(f"  - {record['category']}: {record['count']}개")
        
        # MAP 노드 샘플
        result = neo4j.execute_query("""
            MATCH (m:MAP)
            RETURN m.name as name
            LIMIT 10
        """)
        print("\n🗺️ MAP 노드 샘플:")
        for record in result:
            print(f"  - {record['name']}")
        
        # 관계 개수
        result = neo4j.execute_query("MATCH ()-[r]->() RETURN count(r) as total")
        total_relations = result[0]["total"]
        print(f"\n🔗 전체 관계 개수: {total_relations}")
        
        # 관계 타입별 개수
        result = neo4j.execute_query("""
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(r) as count
            ORDER BY count DESC
        """)
        print("\n🔗 관계 타입별:")
        for record in result:
            print(f"  - {record['rel_type']}: {record['count']}개")
            
    finally:
        neo4j.close()

if __name__ == "__main__":
    check_data()
