#!/usr/bin/env python3
"""
PostgreSQL 데이터 Export → SQL 파일 생성
"""
import sys
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LANGCHAIN_APP_DIR = PROJECT_ROOT / "langchain_app"
sys.path.insert(0, str(LANGCHAIN_APP_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import asyncio
from database.session import AsyncSessionLocal
from database.models.maple_dictionary import MapleDictionary
from sqlalchemy import select
import json


async def export_data_to_sql():
    """PostgreSQL 데이터를 SQL INSERT 문으로 export"""
    
    output_file = PROJECT_ROOT / "init" / "01_maple_data.sql"
    output_file.parent.mkdir(exist_ok=True)
    
    print(f"📤 PostgreSQL 데이터 Export 시작...")
    print(f"출력 파일: {output_file}")
    
    async with AsyncSessionLocal() as db:
        # maple_dictionary 데이터 조회
        stmt = select(MapleDictionary)
        result = await db.execute(stmt)
        entities = result.scalars().all()
        
        print(f"✅ {len(entities)}개 엔티티 발견")
        
        # SQL 파일 생성
        with open(output_file, 'w', encoding='utf-8') as f:
            # 헤더
            f.write(f"-- Maple Dictionary Data Export\n")
            f.write(f"-- Generated at: {datetime.now().isoformat()}\n")
            f.write(f"-- Total entities: {len(entities)}\n\n")
            
            # 기존 데이터 삭제 (선택적)
            f.write("-- Clean up existing data\n")
            f.write("TRUNCATE TABLE maple_dictionary RESTART IDENTITY CASCADE;\n\n")
            
            # INSERT 문 생성
            f.write("-- Insert entities\n")
            for entity in entities:
                # synonyms 배열 처리
                synonyms_sql = "ARRAY[" + ", ".join([f"'{s}'" for s in (entity.synonyms or [])]) + "]"
                
                # detail_data JSONB 처리
                detail_json = json.dumps(entity.detail_data or {}, ensure_ascii=False)
                detail_json = detail_json.replace("'", "''")  # SQL escape
                
                # category enum 처리
                category_value = str(entity.category).split('.')[-1]  # CategoryEnum.MAP -> MAP
                
                sql = f"""INSERT INTO maple_dictionary (
    id, canonical_name, synonyms, category, description, detail_data, created_at, updated_at
) VALUES (
    '{entity.id}',
    '{entity.canonical_name.replace("'", "''")}',
    {synonyms_sql},
    '{category_value}',
    '{(entity.description or "").replace("'", "''")}',
    '{detail_json}'::jsonb,
    '{entity.created_at.isoformat()}',
    '{entity.updated_at.isoformat()}'
) ON CONFLICT (id) DO UPDATE SET
    canonical_name = EXCLUDED.canonical_name,
    synonyms = EXCLUDED.synonyms,
    category = EXCLUDED.category,
    description = EXCLUDED.description,
    detail_data = EXCLUDED.detail_data,
    updated_at = EXCLUDED.updated_at;

"""
                f.write(sql)
            
            # 시퀀스 업데이트 (Auto-increment 재설정)
            f.write("\n-- Update sequences\n")
            f.write("SELECT setval(pg_get_serial_sequence('maple_dictionary', 'id'), (SELECT MAX(id) FROM maple_dictionary));\n")
        
        print(f"✅ SQL 파일 생성 완료: {output_file}")
        print(f"   크기: {output_file.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    asyncio.run(export_data_to_sql())
