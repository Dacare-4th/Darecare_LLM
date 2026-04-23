# 공용 전처리 함수 — PDF 로딩, 청킹, Chroma 저장 (각 ingest.py가 import해서 사용)
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import uuid

VECTORDB_PATH = "./vectordb"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)


def load_pdf(data_dir: str) -> list[tuple[int, str]]:
    """디렉토리 안의 모든 PDF를 로딩해서 (페이지번호, 텍스트) 리스트 반환"""
    pages = []
    for filename in os.listdir(data_dir):
        if not filename.endswith(".pdf"):
            continue
        loader = PyPDFLoader(os.path.join(data_dir, filename))
        for i, page in enumerate(loader.load()):
            pages.append((i + 1, page.page_content))
    return pages


def chunk_text(text: str) -> list[str]:
    """텍스트를 청크 리스트로 분할"""
    return splitter.split_text(text)


def save_to_collection(collection_name: str, chunks: list[str], metadatas: list[dict]):
    """청크와 메타데이터를 Chroma 컬렉션에 저장"""
    client = chromadb.PersistentClient(path=VECTORDB_PATH)
    collection = client.get_or_create_collection(collection_name)
    collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=chunks,
        metadatas=metadatas,
    )
