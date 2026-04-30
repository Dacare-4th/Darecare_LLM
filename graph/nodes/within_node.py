# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# graph/nodes/within_node.py
# 역할 : 동일 보험사 내 플랜을 비교한다
#
# 파이프라인: ① 보험 내 비교
# 진입 조건 : analyze_node 에서 intent == "within_compare"
# 다음 노드  : END
#
# 흐름:
#   1. insurer 컬렉션에서 RAG 검색 (플랜별 문서 분리 검색)
#   2. 비교 프롬프트 조립
#   3. LLM 으로 비교표 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

from graph.nodes.generate_node import call_llm_with_docs, _build_sources, _call_llm_for_related_questions
from graph.nodes.retrieve_node import query_collection
from utils.comparison import build_comparison_prompt, merge_docs_for_comparison, rerank_by_relevance
from utils.schemas import InsuranceState

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────
_WITHIN_SYSTEM_PROMPT = """You are a health insurance plan comparison specialist.
Compare the insurance plans based ONLY on the provided documents.
Present the comparison in a clear, structured table format.
Highlight key differences that would impact the user's decision.
If information is missing for a plan, clearly state "Not specified in documents".
End with a brief recommendation based on the user's apparent needs."""


# ──────────────────────────────────────────────────────────────
# 노드 함수
# ──────────────────────────────────────────────────────────────

def within(state: InsuranceState) -> dict:
    """
    [파이프라인 ①] 동일 보험사 내 플랜 비교표를 생성한다.

    읽는 state 필드:
        user_message : 사용자 원문 질의
        language     : 응답 언어 코드
        insurer      : 비교할 보험사 코드 (예: "uhcg")
        slots        : {"plan": "플랜명"} 등 추출된 슬롯

    반환 dict (InsuranceState 업데이트):
        retrieved_docs    : 검색된 문서 리스트
        answer            : 비교표가 포함된 최종 응답
        sources           : 참조 문서 출처 리스트
        related_questions : 연관 질문 리스트
    """
    user_msg = state["user_message"]
    insurer  = state.get("insurer", "")
    language = state.get("language", "en")
    slots    = state.get("slots", {})

    # ── Step 1: 컬렉션 선택 ────────────────────────────────────
    # 보험사 코드 → 컬렉션 이름 (예: "uhcg" → "uhcg_plans")
    collection_name = f"{insurer}_plans" if insurer else "general_guidelines"

    # ── Step 2: 플랜별 RAG 검색 ────────────────────────────────
    # 슬롯에서 플랜 정보를 추출해 플랜별로 개별 검색한다.
    plan = slots.get("plan", "")

    if plan:
        # 특정 플랜이 명시된 경우 — 해당 플랜 문서 집중 검색
        docs_by_plan = {
            plan: query_collection(
                collection_name = collection_name,
                query           = f"{user_msg} {plan}",
                top_k           = 5,
                where           = {"plan": plan} if plan else None,
            )
        }
        # 비교 대상 (다른 플랜)도 함께 검색
        other_docs = query_collection(
            collection_name = collection_name,
            query           = user_msg,
            top_k           = 5,
        )
        docs_by_plan["Other Plans"] = other_docs
    else:
        # 플랜 미명시 — 전체 검색 후 보험사 플랜 전체 비교
        all_docs = query_collection(
            collection_name = collection_name,
            query           = user_msg,
            top_k           = 8,
        )
        docs_by_plan = {f"{insurer.upper()} Plans": all_docs}

    # ── Step 3: 비교 프롬프트 조립 ─────────────────────────────
    # docs_by_plan 을 {subject: [text, ...]} 형태로 변환
    text_by_plan: dict[str, list[str]] = {
        subject: [d["content"] for d in docs]
        for subject, docs in docs_by_plan.items()
    }
    comparison_prompt = build_comparison_prompt(
        docs_by_subject = text_by_plan,
        user_query      = user_msg,
        language        = language,
    )

    # retrieved_docs 통합 (모든 플랜의 문서를 하나의 리스트로)
    all_retrieved = [doc for docs in docs_by_plan.values() for doc in docs]

    # ── Step 4: LLM 비교표 생성 ────────────────────────────────
    answer = call_llm_with_docs(
        user_query     = comparison_prompt,
        retrieved_docs = all_retrieved,
        language       = language,
        system_prompt  = _WITHIN_SYSTEM_PROMPT,
    )

    sources           = _build_sources(all_retrieved)
    related_questions = _call_llm_for_related_questions(
        user_query = user_msg,
        answer     = answer,
        language   = language,
    )

    return {
        "retrieved_docs"   : all_retrieved,
        "answer"           : answer,
        "sources"          : sources,
        "related_questions": related_questions,
    }
