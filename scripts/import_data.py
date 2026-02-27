#!/usr/bin/env python3
"""
Database 데이터 Import 스크립트
- JSON 데이터를 읽어 MapleDictionary 테이블에 저장
- 카테고리별 metadata 적정성 검사 (Pydantic 검증)
"""
import sys
import json
import os
from pathlib import Path
from typing import Dict, Any

# Path 문제 해결: langchain_app 모듈 경로 추가
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LANGCHAIN_APP_DIR = PROJECT_ROOT / "langchain_app"

sys.path.insert(0, str(LANGCHAIN_APP_DIR))

# .env 파일 명시적으로 로드
from dotenv import load_dotenv
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ .env 파일 로드 완료: {env_path}")
else:
    print(f"⚠️  .env 파일을 찾을 수 없습니다: {env_path}")

# 이제 database 모듈 임포트 가능
from sqlalchemy.orm import Session
from database.session import SessionLocal, engine
from database.models.maple_dictionary import MapleDictionary, CategoryEnum
from database.base import Base

# 존재하는 DTO들만 임포트
from database.schemas.map_dto import MapMetadata
from database.schemas.npc_dto import NPCMetadata
from database.schemas.item_dto import ItemMetadata
from database.schemas.monster_dto import MonsterMetadata

# 1. 카테고리별 스키마 매핑 테이블 (Schema Registry)
SCHEMA_MAP: Dict[str, Any] = {
    CategoryEnum.MAP.value: MapMetadata,
    CategoryEnum.NPC.value: NPCMetadata,
    CategoryEnum.ITEM.value: ItemMetadata,
    CategoryEnum.MONSTER.value: MonsterMetadata,
    # TODO: 필요시 BOSS, QUEST, SKILL 등 추가
}

def get_schema_class(category: str):
    """카테고리에 맞는 Pydantic 모델 클래스 반환"""
    schema_class = SCHEMA_MAP.get(category)
    if not schema_class:
        raise ValueError(f"❌ '{category}' 카테고리에 정의된 스키마가 없습니다.")
    return schema_class

def import_maple_data(file_path: str):
    """
    JSON 파일에서 메이플 데이터를 읽어 MapleDictionary 테이블에 저장
    
    Args:
        file_path: JSON 파일 경로
    """
    # ✅ 테이블 생성 (없으면 생성)
    print("📋 테이블 생성 중...")
    Base.metadata.create_all(bind=engine)
    print(f"   ✅ 테이블: {list(Base.metadata.tables.keys())}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    db: Session = SessionLocal()
    try:
        success_count = 0
        error_count = 0
        
        for entry in json_data:
            try:
                # 2. 필수 기본 필드 추출
                name = entry.get("canonical_name")
                category = entry.get("category")
                description = entry.get("description", "")
                synonyms = entry.get("synonyms", [])
                
                if not name or not category:
                    print(f"⚠️  필수 필드 누락: name={name}, category={category}")
                    error_count += 1
                    continue
                
                # 3. 카테고리에 따른 동적 스키마 검증
                schema_class = get_schema_class(category)
                
                # JSON 데이터 중 detail_data에 해당하는 부분 추출
                # entry 전체에서 기본 필드를 제외한 나머지를 detail_data로 간주
                detail_data_payload = entry.get("detail_data", entry.get("metadata", {})) 
                
                # Pydantic 검증 실행
                validated_meta = schema_class(**detail_data_payload)

                # 4. DB 모델 생성 및 저장 (metadata → detail_data로 변경!)
                db_item = MapleDictionary(
                    canonical_name=name,
                    category=category,
                    description=description or getattr(validated_meta, 'description', ''),
                    synonyms=synonyms,
                    detail_data=validated_meta.model_dump() # 검증된 데이터만 JSONB로 저장
                )

                # Upsert 로직 (이름 중복 시 업데이트)
                existing = db.query(MapleDictionary).filter_by(canonical_name=name).first()
                if existing:
                    existing.category = db_item.category
                    existing.detail_data = db_item.detail_data
                    existing.description = db_item.description
                    existing.synonyms = db_item.synonyms
                    print(f"🔄 업데이트: {name} ({category})")
                else:
                    db.add(db_item)
                    print(f"✅ 추가: {name} ({category})")
                
                # 중요: 각 항목마다 flush하여 DB에 즉시 반영 (중복 체크 정확도 향상)
                db.flush()
                
                success_count += 1
                
            except Exception as e:
                print(f"❌ 항목 처리 실패 [{entry.get('canonical_name', 'unknown')}]: {e}")
                error_count += 1
                continue

        db.commit()
        print(f"\n📊 Import 완료!")
        print(f"  ✅ 성공: {success_count}개")
        print(f"  ❌ 실패: {error_count}개")

    except Exception as e:
        db.rollback()
        print(f"❌ 전체 임포트 실패: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    import sys
    
    # 커맨드라인 인자로 파일 경로 받기
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        # 기본값: training/data/input_data/maple_data.json
        data_file = str(PROJECT_ROOT / "training" / "data" / "input_data" / "maple_data.json")
    
    print(f"📂 데이터 파일: {data_file}")
    
    if not Path(data_file).exists():
        print(f"❌ 파일이 존재하지 않습니다: {data_file}")
        sys.exit(1)
    
    import_maple_data(data_file)
