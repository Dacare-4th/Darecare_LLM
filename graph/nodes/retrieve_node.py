# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# graph/nodes/retrieve_node.py
# 역할 : ChromaDB 에서 관련 문서를 검색한다 (공통 RAG 검색 헬퍼)
#
# 파이프라인: 모든 파이프라인 노드(① ② ③ ④ ⑤ ⑥)에서
#             query_collection() / query_multi_collections() 을
#             import 해서 직접 호출하는 방식으로 사용
#
# [고도화 내역]
#   P1 - HyDE (Hypothetical Document Embeddings)
#        비영어 쿼리 → GPT-4o-mini 로 가상 영어 문서 생성 → Dense 검색에 활용
#        BM25 는 원본 쿼리 그대로 사용 (고유명사 플랜명 보존)
#   P2 - Hybrid Search (Dense + BM25 + RRF)
#        Dense 결과 + BM25 결과를 Reciprocal Rank Fusion 으로 병합
#        BM25 인덱스는 컬렉션별 메모리 캐시 (_bm25_cache)
#   Fix - ChromaDB 1.x 호환
#         Settings import 경로 변경 (chromadb.config → chromadb)
#         get_collection() 에 embedding_function=None 명시 (내장 임베딩 함수 비활성화)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from rank_bm25 import BM25Okapi

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VECTORDB_PATH = str(_PROJECT_ROOT / "vectordb")
DEFAULT_TOP_K  = 5    # 기본 검색 결과 수
_BM25_CANDIDATE = 4   # BM25/Dense 각각 top_k * _BM25_CANDIDATE 개 후보 수집
_RRF_K          = 60  # RRF 상수 (표준값)
_FALLBACK_MIN   = 3   # 필터 결과가 이 수 미만이면 필터 없이 재검색

# HyDE 가 활성화되는 언어 집합
_HYDE_LANGUAGES = {"ko", "en", "ja", "zh-cn", "zh-tw", "fr", "de", "es"}

_HYDE_SYSTEM_PROMPT = (
    "You are a health insurance document writer. "
    "Given a user question, write a short English passage (3-5 sentences) "
    "that would appear in a health insurance policy document and directly answers the question. "
    "Write ONLY the passage, no extra commentary."
)

# ──────────────────────────────────────────────────────────────
# 모듈 레벨 캐시 (매 쿼리마다 재초기화 방지)
# ──────────────────────────────────────────────────────────────

_embedding_model: HuggingFaceEmbeddings | None = None
_chroma_client:   chromadb.PersistentClient  | None = None

# BM25 캐시: {collection_name: (corpus_ids, BM25Okapi)}
_bm25_cache: dict[str, tuple[list[str], BM25Okapi]] = {}


# ──────────────────────────────────────────────────────────────
# 내부 초기화 헬퍼
# ──────────────────────────────────────────────────────────────

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
        device = os.getenv("EMBEDDING_DEVICE", "cpu")
        _embedding_model = HuggingFaceEmbeddings(
            model_name    = "BAAI/bge-m3",
            model_kwargs  = {"device": device},
            encode_kwargs = {"normalize_embeddings": True},
        )
    return _embedding_model


# ──────────────────────────────────────────────────────────────
# P1: HyDE — 가상 문서 확장
# ──────────────────────────────────────────────────────────────

def _hyde_expand(query: str, language: str) -> str:
    """
    비영어 쿼리를 GPT-4o-mini 로 가상 영어 보험 문서 문단으로 확장한다.
    영어 쿼리(language=="en") 도 HyDE 변환을 거쳐 문서 스타일로 재작성한다.
    실패 시 원본 쿼리 반환.
    """
    if language not in _HYDE_LANGUAGES:
        return query
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model    = "gpt-4o-mini",
            messages = [
                {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
                {"role": "user",   "content": query},
            ],
            max_tokens  = 200,
            temperature = 0.1,
        )
        expanded = response.choices[0].message.content.strip()
        print(f"[HyDE] '{query[:40]}' → '{expanded[:60]}...'")
        return expanded
    except Exception as e:
        print(f"[HyDE] 확장 실패 → 원본 쿼리 사용: {e}")
        return query


# ──────────────────────────────────────────────────────────────
# P2-A: Dense 검색
# ──────────────────────────────────────────────────────────────

def _dense_search(
    collection_name: str,
    query: str,
    top_k: int,
    where: dict | None = None,
) -> list[dict]:
    """BGE-M3 임베딩 기반 코사인 유사도 검색."""
    try:
        client     = _get_chroma_client()
        # embedding_function=None → ChromaDB 내장 임베딩 함수 비활성화 (BGE-M3 직접 사용)
        collection = client.get_collection(
            name               = collection_name,
            embedding_function = None,
        )

        model     = _get_embedding_model()
        query_vec = model.embed_query(query)

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_vec],
            "n_results"        : top_k,
            "include"          : ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results   = collection.query(**query_kwargs)
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
            print(f"[Dense] {collection_name} | score={r['score']} | {src} p.{page}")

        return results_list

    except Exception as e:
        print(f"[retrieve_node] Dense 검색 오류 ({collection_name}): {e}")
        return []


# ──────────────────────────────────────────────────────────────
# P2-B: BM25 검색
# ──────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """영문 소문자 + 숫자 토큰화 (BM25 용)."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _get_bm25_index(collection_name: str) -> tuple[list[str], BM25Okapi] | None:
    """
    컬렉션의 전체 문서를 읽어 BM25 인덱스를 빌드한다.
    결과는 _bm25_cache 에 캐싱하여 재사용한다.
    """
    if collection_name in _bm25_cache:
        return _bm25_cache[collection_name]

    try:
        client     = _get_chroma_client()
        collection = client.get_collection(
            name               = collection_name,
            embedding_function = None,
        )
        # 전체 문서 로드 (BM25 인덱스 빌드용) — include=["ids"] IDs는 항상 반환됨
        all_docs = collection.get(include=["documents", "metadatas"])
        corpus_texts = all_docs["documents"] or []
        corpus_meta  = all_docs["metadatas"] or []

        if not corpus_texts:
            return None

        tokenized = [_tokenize(t) for t in corpus_texts]
        bm25      = BM25Okapi(tokenized)

        # id 가 없으면 인덱스를 id 로 사용
        corpus_ids = all_docs.get("ids") or [str(i) for i in range(len(corpus_texts))]

        # 메타데이터도 같이 저장: (ids, texts, metadatas, BM25Okapi)
        _bm25_cache[collection_name] = (corpus_ids, corpus_texts, corpus_meta, bm25)
        print(f"[BM25] '{collection_name}' 인덱스 빌드 완료 ({len(corpus_texts)}개 문서)")
        return _bm25_cache[collection_name]

    except Exception as e:
        print(f"[BM25] 인덱스 빌드 실패 ({collection_name}): {e}")
        return None


def _bm25_search(
    collection_name: str,
    query: str,
    top_k: int,
    where: dict | None = None,
) -> list[dict]:
    """BM25 키워드 검색. where 필터는 후처리로 적용한다."""
    cached = _get_bm25_index(collection_name)
    if cached is None:
        return []

    corpus_ids, corpus_texts, corpus_meta, bm25 = cached
    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)

    # (index, score) 정렬 후 top_k * 2 개 후보 수집
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results: list[dict] = []
    for idx, score in ranked:
        if len(results) >= top_k:
            break
        if score <= 0:
            break
        meta = corpus_meta[idx] if idx < len(corpus_meta) else {}
        # where 필터 후처리
        if where:
            match = all(meta.get(k) == v for k, v in where.items())
            if not match:
                continue
        results.append({
            "content" : corpus_texts[idx],
            "metadata": meta,
            "score"   : round(float(score), 4),
            "_bm25_idx": idx,
        })

    return results


# ──────────────────────────────────────────────────────────────
# P2-C: RRF 병합
# ──────────────────────────────────────────────────────────────

def _rrf_fusion(
    dense_docs: list[dict],
    bm25_docs:  list[dict],
    top_k: int,
) -> list[dict]:
    """
    Dense + BM25 결과를 Reciprocal Rank Fusion 으로 병합한다.

    RRF score = Σ 1 / (k + rank_i),  k = _RRF_K (기본 60)
    content 를 키로 동일 문서를 식별한다.
    """
    rrf_scores: dict[str, float] = {}
    doc_store:  dict[str, dict]  = {}

    for rank, doc in enumerate(dense_docs, start=1):
        key = doc["content"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
        doc_store[key]  = doc

    for rank, doc in enumerate(bm25_docs, start=1):
        key = doc["content"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
        if key not in doc_store:
            doc_store[key] = doc

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for key, rrf_score in merged[:top_k]:
        doc = dict(doc_store[key])
        doc["score"] = round(rrf_score, 6)
        doc.pop("_bm25_idx", None)
        results.append(doc)

    return results


# ──────────────────────────────────────────────────────────────
# 공개 헬퍼 — 모든 파이프라인 노드에서 import 해서 사용
# ──────────────────────────────────────────────────────────────

def query_collection(
    collection_name: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    where: dict | None = None,
    hybrid: bool = True,
    hyde: bool = False,
    language: str = "en",
) -> list[dict]:
    """
    단일 ChromaDB 컬렉션에서 유사 문서를 검색한다.

    Args:
        collection_name : 검색할 컬렉션 이름
        query           : 검색 쿼리 텍스트
        top_k           : 반환할 최대 문서 수
        where           : 메타데이터 필터 (예: {"plan": "Gold"})
        hybrid          : True → Dense + BM25 + RRF, False → Dense only
        hyde            : True → HyDE 확장 (Dense 검색 쿼리에만 적용)
        language        : 언어 코드 (HyDE 트리거 판단에 사용)

    Returns:
        [{"content": str, "metadata": dict, "score": float}, ...] 형태의 문서 리스트
        컬렉션이 없거나 오류 시 빈 리스트 반환
    """
    candidate_k = top_k * _BM25_CANDIDATE

    # HyDE: Dense 검색용 쿼리만 확장 (BM25 는 원본 유지)
    dense_query = _hyde_expand(query, language) if hyde else query

    # Dense 검색: where 필터 시 매칭 문서가 적을 수 있으므로 top_k 그대로 사용
    # (candidate_k=20 으로 요청하면 ChromaDB 가 필터 결과 < n_results 일 때 예외 발생)
    dense_top_k = top_k if where else candidate_k
    dense_docs = _dense_search(collection_name, dense_query, dense_top_k, where)

    if not hybrid:
        return dense_docs[:top_k]

    # BM25 검색
    try:
        bm25_docs = _bm25_search(collection_name, query, candidate_k, where)
    except Exception as bm25_err:
        print(f"[BM25] 검색 오류 ({collection_name}): {bm25_err}")
        bm25_docs = []

    # RRF 병합
    return _rrf_fusion(dense_docs, bm25_docs, top_k)


def query_multi_collections(
    collection_names: list[str],
    query: str,
    top_k_each: int = DEFAULT_TOP_K,
    hybrid: bool = True,
    hyde: bool = False,
    language: str = "en",
) -> dict[str, list[dict]]:
    """
    여러 컬렉션을 순차 검색한다. (② 보험사 비교 파이프라인 용)

    Args:
        collection_names : 검색할 컬렉션 이름 리스트
        query            : 검색 쿼리 텍스트
        top_k_each       : 컬렉션당 반환할 최대 문서 수
        hybrid           : Hybrid Search 활성화 여부
        hyde             : HyDE 활성화 여부
        language         : 언어 코드

    Returns:
        {collection_name: [doc_dict, ...]} 형태
    """
    return {
        name: query_collection(
            name, query, top_k_each,
            hybrid=hybrid, hyde=hyde, language=language,
        )
        for name in collection_names
    }
