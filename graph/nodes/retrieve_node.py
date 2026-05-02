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
# 05.01 - SentenceTransformerEmbeddingFunction 제거
# ingest_to_db.py는 HuggingFaceEmbeddings로 임베딩을 직접 넣기 때문에
# ChromaDB 컬렉션에 embedding_function이 등록되어 있지 않다(persisted: default).
# get_collection()에 다른 embedding_function을 전달하면 충돌 경고와 함께
# query_texts 기반 검색이 실패해 retrieved_docs가 항상 0이 됨.
# 쿼리도 ingest와 동일하게 HuggingFaceEmbeddings로 직접 임베딩 후 query_embeddings로 전달.
from langchain_huggingface import HuggingFaceEmbeddings

# 상수
# __file__ 기준 절대경로 사용 → CWD(현재 작업 디렉토리)와 무관하게 동작
# retrieve_node.py 위치: Dacare_LLM/graph/nodes/retrieve_node.py
# vectordb 위치:         Dacare_LLM/vectordb/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VECTORDB_PATH = str(_PROJECT_ROOT / "vectordb")
DEFAULT_TOP_K = 5   # 기본 검색 결과 수

# 05.01 - 모델을 매 쿼리마다 새로 로드하면 수백 MB를 반복 로딩하므로 모듈 레벨에서 캐싱
_embedding_model: HuggingFaceEmbeddings | None = None

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
        client     = chromadb.PersistentClient(
            path     = VECTORDB_PATH,
            settings = Settings(anonymized_telemetry=False),
        )
        # 05.01 - embedding_function 인자 제거
        # ingest_to_db.py가 embeddings를 직접 col.add()로 넣기 때문에
        # ChromaDB가 기록한 embedding_function은 'default'(none).
        # 여기에 다른 함수를 넘기면 충돌 경고 후 query_texts 검색 실패.
        collection = client.get_collection(name=collection_name)

        # 05.01 - query_texts → query_embeddings 로 변경
        # ingest와 동일한 BGE-M3 모델로 쿼리를 직접 임베딩해서 전달
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

        return [
            {
                "content" : doc,
                "metadata": meta,
                "score"   : round(1 - dist, 4),  # cosine distance → similarity
            }
            for doc, meta, dist in zip(docs, metadatas, distances)
            if doc and doc.strip()
        ]

    except Exception as e:
        # 컬렉션 미존재 또는 DB 오류 → 빈 결과 반환 (서비스 중단 방지)
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
