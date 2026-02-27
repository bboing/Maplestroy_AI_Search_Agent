#!/usr/bin/env python3
"""Neo4j 전체 MAP 노드 확인"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "langchain_app"))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from database.neo4j_connection import Neo4jConnection

def check_all_maps():
    neo4j = Neo4jConnection()
    
    try:
        # 전체 MAP 노드
        result = neo4j.execute_query("""
            MATCH (m:MAP)
            RETURN m.name as name
            ORDER BY m.name
        """)
        
        print(f"\n📍 전체 MAP 노드 ({len(result)}개):")
        for i, record in enumerate(result, 1):
            print(f"  {i:2d}. {record['name']}")
            
    finally:
        neo4j.close()

if __name__ == "__main__":
    check_all_maps()
