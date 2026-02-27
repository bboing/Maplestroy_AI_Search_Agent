"""문서 처리기 - 업로드, 청킹, 벡터화"""
from fastapi import UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from src.retrievers.milvus_retriever import MilvusRetriever
from config.settings import settings
import uuid
import tempfile
import os
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """문서 처리 및 벡터화"""
    
    def __init__(self):
        self.retriever = MilvusRetriever()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    async def process_upload(self, file: UploadFile) -> Dict:
        """파일 업로드 처리"""
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # 파일 타입별 로더 선택
            file_ext = os.path.splitext(file.filename)[1].lower()
            
            if file_ext == '.pdf':
                loader = PyPDFLoader(tmp_path)
            elif file_ext in ['.txt', '.md']:
                if file_ext == '.md':
                    loader = UnstructuredMarkdownLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # 문서 로드
            documents = loader.load()
            
            # 청킹
            chunks = self.text_splitter.split_documents(documents)
            
            # 문서 ID 생성
            document_id = str(uuid.uuid4())
            
            # 청크 데이터 준비
            chunk_data = []
            for i, chunk in enumerate(chunks):
                chunk_data.append({
                    "id": f"{document_id}_{i}",
                    "content": chunk.page_content,
                    "document_id": document_id,
                    "chunk_index": i
                })
            
            # Milvus에 삽입
            await self.retriever.insert(chunk_data)
            
            logger.info(f"Processed document: {file.filename} ({len(chunks)} chunks)")
            
            return {
                "document_id": document_id,
                "title": file.filename,
                "chunk_count": len(chunks),
                "status": "completed"
            }
            
        finally:
            # 임시 파일 삭제
            os.unlink(tmp_path)
    
    async def list_documents(self) -> List[Dict]:
        """문서 목록 (TODO: PostgreSQL에서 조회)"""
        # PostgreSQL 연동 필요
        return []
    
    async def delete_document(self, document_id: str):
        """문서 삭제"""
        await self.retriever.delete_by_document(document_id)
        # TODO: PostgreSQL에서도 삭제
        logger.info(f"Deleted document: {document_id}")
