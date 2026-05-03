# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# graph/nodes/retrieve_node.py
# 역할 : ChromaDB 에서 관련 문서를 검색한다 (공통 RAG 검색 헬퍼)
#
# 파이프라인: 모든 파이프라인 노드(① ② ③ ④ ⑤ ⑥)에서
#             query_collection() / query_multi_collections() 을
#             import 해서 직접 호출하는 방식으로 사용
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

from pathlib import Path
import chromadb
from chromadb.config import Settings

from langchain_huggingface import HuggingFaceEmbeddings

# 상수

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VECTORDB_PATH = str(_PROJECT_ROOT / "vectordb")
DEFAULT_TOP_K = 5   # 기본 검색 결과 수

# 모델 + ChromaDB 클라이언트 모두 모듈 레벨 캐싱 (매 쿼리마다 재연결 방지)
_embedding_model: HuggingFaceEmbeddings | None = None
_chroma_client: chromadb.PersistentClient | None = None

def _get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path     = VECTORDB_PATH,
            settings = Settings(anonymized_telemetry=False),
        )
    return _chroma_client

def _get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name     = "BAAI/bge-m3",
            model_kwargs   = {"device": "cpu"},
            encode_kwargs  = {"normalize_embeddings": True},
        )
    return _embedding_model


# 공개 헬퍼 — 모든 파이프라인 노드에서 import 해서 사용

def query_collection(
    collection_name: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    where: dict | None = None,
) -> list[dict]:
    """
    단일 ChromaDB 컬렉션에서 유사 문서를 검색한다.

    Args:
        collection_name : 검색할 컬렉션 이름
        query           : 검색 쿼리 텍스트
        top_k           : 반환할 최대 문서 수
        where           : 메타데이터 필터 (예: {"source_type": "pdf_table"})

    Returns:
        [{"content": str, "metadata": dict, "score": float}, ...] 형태의 문서 리스트
        컬렉션이 없거나 오류 시 빈 리스트 반환
    """
    try:
        client     = _get_chroma_client()
        collection = client.get_collection(name=collection_name)

        model       = _get_embedding_model()
        query_vec   = model.embed_query(query)

        # 메타데이터 필터 포함 여부에 따라 쿼리 분기
        query_kwargs: dict = {
            "query_embeddings": [query_vec],
            "n_results"        : top_k,
            "include"          : ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = collection.query(**query_kwargs)

        # ChromaDB 결과를 통일 형식으로 변환
        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        results_list = [
            {
                "content" : doc,
                "metadata": meta,
                "score"   : round(1 - dist, 4),
            }
            for doc, meta, dist in zip(docs, metadatas, distances)
            if doc and doc.strip()
        ]

        for r in results_list:
            src  = r["metadata"].get("source") or r["metadata"].get("file_name", "unknown")
            page = r["metadata"].get("page", "")
            print(f"[RAG] {collection_name} | score={r['score']} | {src} p.{page}")

        return results_list

    except Exception as e:
        print(f"[retrieve_node] 검색 오류 ({collection_name}): {e}")
        return []


def query_multi_collections(
    collection_names: list[str],
    query: str,
    top_k_each: int = 5,
) -> dict[str, list[dict]]:
    """
    여러 컬렉션을 병렬로 검색한다. (② 보험사 비교 파이프라인 용)

    Args:
        collection_names : 검색할 컬렉션 이름 리스트
        query            : 검색 쿼리 텍스트
        top_k_each       : 컬렉션당 반환할 최대 문서 수

    Returns:
        {collection_name: [doc_dict, ...]} 형태
    """
    return {
        name: query_collection(name, query, top_k_each)
        for name in collection_names
    }
