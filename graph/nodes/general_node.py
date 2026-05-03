# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# graph/nodes/general_node.py
# 역할 : 특정 카테고리에 해당하지 않는 일반 질문을 처리한다
#
# 파이프라인: ⑦ 일반 질문
# 진입 조건 : analyze_node 에서 intent == "general_query"
#             (예: "심리상담 커버돼?", "치과 보험 되나요?", 기타 문서 기반 질문)
#             (insurer 슬롯이 확정된 상태로 진입)
# 다음 노드  : END
#
# 흐름:
#   1. {insurer}_plans 컬렉션에서 RAG 검색
#   2. LLM 으로 문서 기반 답변 생성
#
# [고도화 내역]
#   P1 - HyDE → english_query 방식으로 교체
#        analyze_node에서 생성된 english_query를 검색에 사용 (hyde=False)
#        Dense + BM25 모두 영어 키워드로 검색 → 정확도 향상
#   P2 - slots["plan"] 기반 메타데이터 필터 적용
#        결과 < _FALLBACK_MIN 이면 필터 없이 재검색
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

from graph.nodes.generate_node import call_llm_with_docs, _build_sources, _call_llm_for_related_questions
from graph.nodes.retrieve_node import query_collection, query_multi_collections, _FALLBACK_MIN
from utils.schemas import InsuranceState

_ALL_INSURERS = ["uhcg", "cigna", "tricare", "msh_china"]

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────
_GENERAL_SYSTEM_PROMPT = """You are a helpful health insurance assistant.
Answer the user's question based ONLY on the provided reference documents.
If the documents do not contain enough information, say so clearly.
Always cite which document your answer is based on.
Keep your answer concise and structured."""


# ──────────────────────────────────────────────────────────────
# 노드 함수
# ──────────────────────────────────────────────────────────────

def general(state: InsuranceState) -> dict:
    """
    [파이프라인 ⑦] 특정 카테고리에 해당하지 않는 일반 질문에 문서 기반으로 답변한다.

    읽는 state 필드:
        user_message : 사용자 원문 질의
        language     : 응답 언어 코드
        insurer      : 보험사 코드
        slots        : 추출된 슬롯 (treatment, plan 등)

    반환 dict (InsuranceState 업데이트):
        retrieved_docs    : 검색된 문서 리스트
        answer            : 문서 기반 답변
        sources           : 참조 문서 출처 리스트
        related_questions : 연관 질문 리스트
    """
    user_msg      = state["user_message"]
    language      = state.get("language", "en")
    insurer       = state.get("insurer", "")
    slots         = state.get("slots", {})
    chat_history  = state.get("chat_history", [])
    english_query = state.get("english_query", "") or user_msg

    # P2: slots 에서 plan 추출 → 메타데이터 필터
    plan         = slots.get("plan", "")
    where_filter = {"plan": plan} if plan else None

    # ── Step 1: RAG 검색 ───────────────────────────────────────
    if insurer:
        # english_query 를 검색 쿼리로 사용 (한국어 등 비영어 쿼리의 BM25 매칭 향상)
        search_query = english_query

        docs = query_collection(
            collection_name = f"{insurer}_plans",
            query           = search_query,
            top_k           = 10,
            where           = where_filter,
            hyde            = False,
            language        = language,
        )

        # P2 fallback: 필터 결과가 너무 적으면 필터 없이 재검색
        if where_filter and len(docs) < _FALLBACK_MIN:
            print(f"[general_node] plan 필터 결과 {len(docs)}개 → 필터 없이 재검색")
            docs = query_collection(
                collection_name = f"{insurer}_plans",
                query           = search_query,
                top_k           = 10,
                hyde            = False,
                language        = language,
            )
    else:
        # insurer 미확정 시 전체 보험사 컬렉션 검색 후 score 기준 정렬
        multi = query_multi_collections(
            collection_names = [f"{ins}_plans" for ins in _ALL_INSURERS],
            query            = english_query,
            top_k_each       = 3,
            hyde             = False,
            language         = language,
        )
        docs = sorted(
            [doc for results in multi.values() for doc in results],
            key     = lambda d: d.get("score", 0),
            reverse = True,
        )[:8]

    # ── Step 2: LLM 문서 기반 답변 생성 ──────────────────────────
    answer = call_llm_with_docs(
        user_query     = user_msg,
        retrieved_docs = docs,
        language       = language,
        extra_context  = slots,
        system_prompt  = _GENERAL_SYSTEM_PROMPT,
    )

    sources           = _build_sources(docs)
    related_questions = _call_llm_for_related_questions(
        user_query = user_msg,
        answer     = answer,
        language   = language,
    )

    return {
        "retrieved_docs"   : docs,
        "answer"           : answer,
        "sources"          : sources,
        "related_questions": related_questions,
        "chat_history"     : chat_history + [{"role": "assistant", "content": answer}],
    }
