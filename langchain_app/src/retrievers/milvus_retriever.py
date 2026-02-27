"""Milvus 벡터 검색기 (Async)"""
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from src.models.embeddings import embedding_model
from config.settings import settings
from typing import List, Dict
import asyncio
import logging

logger = logging.getLogger(__name__)


class MilvusRetriever:
    """Milvus 벡터 DB 검색기"""
    
    def __init__(self, collection_name: str = None):
        # Q&A 컬렉션 기본 사용 (maple_qa)
        self.collection_name = collection_name or "maple_qa"
        self.dimension = 384  # paraphrase-multilingual-MiniLM-L12-v2
        self._connect()
        self._load_collection()
    
    def _connect(self):
        """Milvus 연결"""
        try:
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT
            )
            logger.info(f"Connected to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        except Exception as e:
            logger.error(f"Milvus connection failed: {e}")
            raise
    
    def _load_collection(self):
        """기존 컬렉션 로드"""
        if not utility.has_collection(self.collection_name):
            logger.warning(f"컬렉션이 없습니다: {self.collection_name}")
            logger.warning("먼저 sync_to_milvus.py를 실행하세요!")
            raise ValueError(f"Collection '{self.collection_name}' does not exist")
        
        self.collection = Collection(self.collection_name)
        self.collection.load()
        logger.info(f"Loaded collection: {self.collection_name}")
    
    async def insert(self, chunks: List[Dict]) -> List[str]:
        """
        벡터 삽입
        
        chunks: [
            {
                "id": "uuid",
                "content": "텍스트",
                "document_id": "문서ID",
                "chunk_index": 0
            },
            ...
        ]
        """
        # 임베딩 생성
        contents = [chunk["content"] for chunk in chunks]
        embeddings = embedding_model.embed_texts(contents)
        
        # 데이터 준비
        entities = [
            [chunk["id"] for chunk in chunks],  # id
            embeddings,  # embedding
            contents,  # content
            [chunk["document_id"] for chunk in chunks],  # document_id
            [chunk["chunk_index"] for chunk in chunks],  # chunk_index
        ]
        
        # 삽입
        self.collection.insert(entities)
        self.collection.flush()
        
        logger.info(f"Inserted {len(chunks)} vectors into Milvus")
        return [chunk["id"] for chunk in chunks]
    
    def _search_sync(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Q&A 유사도 검색 (동기, 내부용)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 개수
            
        Returns:
            검색 결과 리스트
        """
        # 쿼리 임베딩
        query_embedding = embedding_model.embed_text(query)
        
        # 검색
        search_params = {"metric_type": "COSINE", "params": {"ef": 100}}
        
        # maple_qa 컬렉션의 필드: entity_id, entity_name, entity_type, question, answer, qa_type, embedding
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["entity_id", "entity_name", "entity_type", "question", "answer", "qa_type"]
        )
        
        # 결과 포맷팅
        retrieved = []
        for hits in results:
            for hit in hits:
                retrieved.append({
                    "id": hit.entity.get("entity_id"),
                    "canonical_name": hit.entity.get("entity_name"),
                    "category": hit.entity.get("entity_type"),
                    "question": hit.entity.get("question"),
                    "answer": hit.entity.get("answer"),
                    "qa_type": hit.entity.get("qa_type"),
                    "score": float(hit.score),
                    "distance": float(hit.distance) if hasattr(hit, 'distance') else 1.0 - float(hit.score)
                })
        
        return retrieved
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Q&A 유사도 검색 (비동기)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 개수
            
        Returns:
            검색 결과 리스트
        """
        # pymilvus는 async를 지원하지 않으므로 to_thread로 래핑
        return await asyncio.to_thread(self._search_sync, query, top_k)
    
    async def get_relevant_documents(self, query: str) -> List[Dict]:
        """
        LangChain Retriever 호환 메서드 (비동기)
        
        Args:
            query: 검색 쿼리
            
        Returns:
            검색 결과 (Document 형식이 아닌 Dict)
        """
        return await self.search(query, top_k=5)
    
    async def delete_by_document(self, document_id: str):
        """문서의 모든 벡터 삭제"""
        expr = f'document_id == "{document_id}"'
        self.collection.delete(expr)
        logger.info(f"Deleted vectors for document: {document_id}")
    
    async def get_stats(self) -> Dict:
        """통계 정보"""
        stats = self.collection.num_entities
        return {
            "collection_name": self.collection_name,
            "total_vectors": stats,
            "dimension": self.dimension
        }
