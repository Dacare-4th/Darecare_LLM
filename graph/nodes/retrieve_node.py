# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# graph/nodes/retrieve_node.py
# 역할 : ChromaDB 에서 관련 문서를 검색한다 (공통 RAG 검색 헬퍼)
#
# 파이프라인: 모든 파이프라인 노드(① ② ③ ④ ⑤ ⑥)에서
#             query_collection() / query_multi_collections() 을
#             import 해서 직접 호출하는 방식으로 사용
#
# 참고: retrieve() 노드 함수는 builder.py 에 등록되어 있으나
#       실제 그래프 흐름에서 호출되지 않음.
#       핵심은 공개 헬퍼 함수 query_collection() 임.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from utils.schemas import InsuranceState

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────
VECTORDB_PATH = "./vectordb"
DEFAULT_TOP_K = 5   # 기본 검색 결과 수


# ──────────────────────────────────────────────────────────────
# 노드 함수
# ──────────────────────────────────────────────────────────────

def retrieve(state: InsuranceState) -> dict:
    """
    [공통 RAG] 사용자 질의와 관련된 문서를 ChromaDB 에서 검색한다.

    읽는 state 필드:
        user_message : 검색 쿼리 원문
        intent       : 컬렉션 선택 기준 (procedure → general_guidelines)
        insurer      : 보험사별 컬렉션 선택 기준

    반환 dict (InsuranceState 업데이트):
        retrieved_docs : [{"content": str, "metadata": dict}, ...] 형태의 문서 리스트
    """
    query     = state["user_message"]
    intent    = state.get("intent", "")
    insurer   = state.get("insurer", "")

    # ── 컬렉션 선택 ────────────────────────────────────────────
    # intent 와 insurer 를 조합해 적절한 컬렉션을 선택한다.
    collection_name = _select_collection(intent, insurer)

    # ── ChromaDB 검색 ──────────────────────────────────────────
    docs = query_collection(
        collection_name = collection_name,
        query           = query,
        top_k           = DEFAULT_TOP_K,
    )

    return {"retrieved_docs": docs}


# ──────────────────────────────────────────────────────────────
# 공개 헬퍼 — 다른 노드(within, nhis, claim 등)에서도 직접 호출 가능
# ──────────────────────────────────────────────────────────────

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
        [{"content": str, "metadata": dict}, ...] 형태의 문서 리스트
        컬렉션이 없거나 오류 시 빈 리스트 반환
    """
    try:
        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name = "BAAI/bge-m3",  # ingest 시 사용한 모델과 반드시 동일해야 함
        )
        client     = chromadb.PersistentClient(
            path     = VECTORDB_PATH,
            settings = Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(
            name               = collection_name,
            embedding_function = embedding_fn,
        )

        # 메타데이터 필터 포함 여부에 따라 쿼리 분기
        query_kwargs: dict = {
            "query_texts": [query],
            "n_results"  : top_k,
            "include"    : ["documents", "metadatas", "distances"],
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


# ──────────────────────────────────────────────────────────────
# 내부 함수
# ──────────────────────────────────────────────────────────────

def _select_collection(intent: str, insurer: str) -> str:
    """
    intent 와 insurer 를 기반으로 검색할 컬렉션 이름을 반환한다.

    컬렉션 목록:
        {insurer}_plans     — 각 보험사 플랜 문서 (uhcg, cigna, tricare, msh_china)
        nhis                — NHIS 관련 문서
        general_guidelines  — 일반 보험 절차 문서
        claim_procedures    — 청구 절차 문서
    """
    if intent == "procedure":
        return "general_guidelines"
    if intent == "claim":
        return "claim_procedures"
    if intent == "nhis":
        return "nhis"
    if insurer:
        # 보험사 코드 → 해당 컬렉션 (예: "uhcg" → "uhcg_plans")
        if insurer == "nhis":
            return "nhis"
        return f"{insurer}_plans"

    # fallback — 일반 가이드라인
    return "general_guidelines"
