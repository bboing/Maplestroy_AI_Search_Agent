#!/usr/bin/env python3
"""
PostgreSQL → Milvus 동기화
Q&A 형식으로 데이터 생성 및 임베딩
"""
import sys
from pathlib import Path
from uuid import uuid4

# 경로 설정
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LANGCHAIN_APP_DIR = PROJECT_ROOT / "langchain_app"
sys.path.insert(0, str(LANGCHAIN_APP_DIR))

# .env 로드
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

print("✅ 1. 환경 설정 완료")

# Import
from database.session import SessionLocal
from database.models.maple_dictionary import MapleDictionary
from src.utils.qa_generator import QAGenerator
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import settings
import os

print("✅ 2. 모듈 import 완료")


# 임베딩 모델 초기화
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def connect_milvus():
    """Milvus 연결"""
    try:
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT
        )
        print(f"✅ Milvus 연결: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    except Exception as e:
        print(f"❌ Milvus 연결 실패: {e}")
        raise


def create_qa_collection(collection_name: str = "maple_qa", drop_existing: bool = False):
    """Q&A 전용 컬렉션 생성"""
    
    if utility.has_collection(collection_name):
        if drop_existing:
            print(f"⚠️  기존 컬렉션 삭제: {collection_name}")
            utility.drop_collection(collection_name)
        else:
            print(f"✅ 기존 컬렉션 사용: {collection_name}")
            return Collection(collection_name)
    
    # 스키마 정의
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="entity_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="entity_name", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="entity_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="qa_type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # MiniLM 384차원
    ]
    
    schema = CollectionSchema(fields=fields, description="Maple Q&A with embeddings")
    collection = Collection(name=collection_name, schema=schema)
    
    # 인덱스 생성 (HNSW - 빠른 검색)
    index_params = {
        "metric_type": "COSINE",  # 코사인 유사도
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }
    
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"✅ 컬렉션 생성 완료: {collection_name}")
    
    return collection


def sync_to_milvus(batch_size: int = 100, drop_existing: bool = False):
    """PostgreSQL → Milvus 동기화"""
    
    print("\n" + "="*80)
    print("PostgreSQL → Milvus Q&A 동기화 시작")
    print("="*80 + "\n")
    
    # 1. DB 연결
    db = SessionLocal()
    connect_milvus()
    
    # 2. 임베딩 모델 초기화
    print("🤖 임베딩 모델 로딩 중... (처음이면 다운로드 시간 소요)")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 한국어 지원!
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ 3. 임베딩 모델 로드 완료")
    except Exception as e:
        print(f"❌ 임베딩 모델 로드 실패: {e}")
        print("\n해결 방법:")
        print("  pip install sentence-transformers")
        print("  pip install torch")
        db.close()
        return
    
    # 3. Q&A 생성기
    qa_generator = QAGenerator()
    
    # 4. PostgreSQL에서 모든 엔티티 읽기
    print("\n📖 PostgreSQL에서 엔티티 읽기 중...")
    entities = db.query(MapleDictionary).all()
    print(f"✅ {len(entities)}개 엔티티 로드")
    
    # 5. Q&A 생성
    print("\n🔨 Q&A 생성 중...")
    entities_dict = [entity.to_dict() for entity in entities]
    qa_list = qa_generator.generate_batch(entities_dict)
    print(f"✅ {len(qa_list)}개 Q&A 생성 완료")
    
    if not qa_list:
        print("⚠️  생성된 Q&A가 없습니다.")
        db.close()
        return
    
    # 6. Milvus 컬렉션 생성
    print("\n📦 Milvus 컬렉션 준비 중...")
    collection = create_qa_collection("maple_qa", drop_existing=drop_existing)
    
    # 7. 임베딩 생성 & Milvus 저장
    print(f"\n🧮 임베딩 생성 중... (총 {len(qa_list)}개)")
    
    # 배치 처리
    for i in range(0, len(qa_list), batch_size):
        batch = qa_list[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(qa_list) + batch_size - 1) // batch_size
        
        print(f"\n[Batch {batch_num}/{total_batches}] {len(batch)}개 처리 중...")
        
        # 임베딩 생성
        embedding_texts = [qa['embedding_text'] for qa in batch]
        embeddings_list = embeddings.embed_documents(embedding_texts)
        
        # Milvus 데이터 준비
        data = [
            [str(uuid4()) for _ in batch],  # id
            [qa['entity_id'] for qa in batch],  # entity_id
            [qa['entity_name'] for qa in batch],  # entity_name
            [qa['entity_type'] for qa in batch],  # entity_type
            [qa['question'] for qa in batch],  # question
            [qa['answer'] for qa in batch],  # answer
            [qa['qa_type'] for qa in batch],  # qa_type
            embeddings_list  # embedding
        ]
        
        # Milvus에 삽입
        try:
            collection.insert(data)
            print(f"   ✅ Batch {batch_num} 저장 완료")
        except Exception as e:
            print(f"   ❌ Batch {batch_num} 저장 실패: {e}")
    
    # 8. 인덱스 로드 (검색 가능하도록)
    print("\n📊 인덱스 로딩 중...")
    collection.load()
    
    # 9. 통계
    print("\n" + "="*80)
    print("📊 동기화 완료!")
    print("="*80)
    print(f"엔티티: {len(entities)}개")
    print(f"Q&A: {len(qa_list)}개")
    print(f"평균 Q&A/엔티티: {len(qa_list)/len(entities):.1f}개")
    
    # 카테고리별 통계
    from collections import Counter
    type_counts = Counter(qa['entity_type'] for qa in qa_list)
    print(f"\n카테고리별 Q&A:")
    for entity_type, count in type_counts.items():
        print(f"  {entity_type}: {count}개")
    
    qa_type_counts = Counter(qa['qa_type'] for qa in qa_list)
    print(f"\nQ&A 타입별:")
    for qa_type, count in sorted(qa_type_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {qa_type}: {count}개")
    
    print("\n" + "="*80)
    
    db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PostgreSQL → Milvus Q&A 동기화")
    parser.add_argument("--batch-size", type=int, default=100, help="배치 크기")
    parser.add_argument("--drop", action="store_true", help="기존 컬렉션 삭제 후 재생성")
    
    args = parser.parse_args()
    
    print("\n🚀 Milvus Q&A 동기화")
    if args.drop:
        print("⚠️  기존 데이터를 삭제하고 새로 생성합니다!")
        confirm = input("계속하시겠습니까? (y/N): ")
        if confirm.lower() != 'y':
            print("취소되었습니다.")
            sys.exit(0)
    
    try:
        sync_to_milvus(batch_size=args.batch_size, drop_existing=args.drop)
    except Exception as e:
        print(f"\n❌ 동기화 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
