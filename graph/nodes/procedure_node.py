# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# graph/nodes/procedure_node.py
# 역할 : 보험사 약관 기반 절차를 안내한다 (retrieve + generate 통합)
#
# 파이프라인: ④ 절차 안내
# 진입 조건 : analyze_node 에서 intent == "procedure"
#             (insurer 슬롯이 확정된 상태로 진입)
# 다음 노드  : END
#
# 흐름:
#   1. slots 에서 treatment / plan 을 꺼내 검색 쿼리를 보강
#   2. insurer 에 맞는 컬렉션 선택
#      - nhis              → "nhis" 컬렉션
#      - 일반 보험사       → "{insurer}_plans" 컬렉션
#      - insurer 미확정    → query_collection 내부에서 빈 컬렉션 처리
#   3. RAG 검색
#   4. LLM 으로 단계별 안내 + 필요 서류 목록 생성
#
# [고도화 내역]
#   P1 - HyDE → english_query 방식으로 교체
#        analyze_node에서 생성된 english_query를 검색에 사용 (hyde=False)
#   P2 - slots["plan"] 기반 메타데이터 필터 적용
#        결과 < _FALLBACK_MIN 이면 필터 없이 재검색
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

from graph.nodes.generate_node import call_llm_parallel, _build_sources
from graph.nodes.retrieve_node import query_collection, query_multi_collections, _FALLBACK_MIN
from utils.schemas import InsuranceState

_ALL_INSURERS = ["uhcg", "cigna", "tricare", "msh_china"]

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────
_PROCEDURE_SYSTEM_PROMPT = """You are a health insurance procedure guide.
Explain insurance procedures in clear, numbered steps.
Always include:
1. Step-by-step process (numbered list)
2. Required documents checklist
3. Estimated timeline
4. Important notes or warnings

Base your answer ONLY on the provided documents.
If a step is not covered in the documents, note that the user should contact the insurer directly."""


# ──────────────────────────────────────────────────────────────
# 노드 함수
# ──────────────────────────────────────────────────────────────

def procedure(state: InsuranceState) -> dict:
    """
    [파이프라인 ④] 보험 절차를 단계별로 안내한다.

    읽는 state 필드:
        user_message : 사용자 원문 질의
        language     : 응답 언어 코드
        insurer      : 보험사 코드 (보험사별 절차 우선 검색)
        slots        : 추출된 슬롯 (treatment, plan 등)

    반환 dict (InsuranceState 업데이트):
        retrieved_docs    : 검색된 절차 문서 리스트
        answer            : 단계별 절차 안내 + 필요 서류 목록
        sources           : 참조 문서 출처 리스트
        related_questions : 연관 질문 리스트
    """
    user_msg      = state["user_message"]
    language      = state.get("language", "en")
    insurer       = state.get("insurer", "")
    slots         = state.get("slots", {})
    english_query = state.get("english_query", "") or user_msg
    chat_history  = state.get("chat_history", [])

    # ── Step 1: 슬롯 기반 쿼리 보강 ───────────────────────────
    treatment = slots.get("treatment", "")
    plan      = slots.get("plan", "")

    # english_query 를 기반으로 슬롯 정보 보강
    query_parts = [english_query]
    if treatment and treatment.lower() not in english_query.lower():
        query_parts.append(treatment)
    if plan and plan.lower() not in english_query.lower():
        query_parts.append(plan)
    if "procedure" not in english_query.lower():
        query_parts.append("procedure")
    enriched_query = " ".join(query_parts)

    # P2: plan 메타데이터 필터
    where_filter = {"plan": plan} if plan else None

    # ── Step 2: 컬렉션 선택 + RAG 검색 ──────────────────────────
    if insurer == "nhis":
        # NHIS 전용 컬렉션
        docs = query_collection(
            collection_name = "nhis",
            query           = enriched_query,
            top_k           = 5,
            where           = where_filter,
            hyde            = False,
            language        = language,
        )
        if where_filter and len(docs) < _FALLBACK_MIN:
            print(f"[procedure_node] plan 필터 결과 {len(docs)}개 → 필터 없이 재검색")
            docs = query_collection(
                collection_name = "nhis",
                query           = enriched_query,
                top_k           = 5,
                hyde            = False,
                language        = language,
            )

    elif insurer:
        # 특정 보험사 컬렉션
        collection_name = f"{insurer}_plans"
        docs = query_collection(
            collection_name = collection_name,
            query           = enriched_query,
            top_k           = 5,
            where           = where_filter,
            hyde            = False,
            language        = language,
        )
        # P2 fallback: 필터 결과가 너무 적으면 필터 없이 재검색
        if where_filter and len(docs) < _FALLBACK_MIN:
            print(f"[procedure_node] plan 필터 결과 {len(docs)}개 → 필터 없이 재검색")
            docs = query_collection(
                collection_name = collection_name,
                query           = enriched_query,
                top_k           = 5,
                hyde            = False,
                language        = language,
            )

    else:
        # insurer 미확정 → 전체 보험사 컬렉션 검색 후 score 기준 병합
        print("[procedure_node] insurer 미확정 → 전체 컬렉션 검색")
        multi = query_multi_collections(
            collection_names = [f"{ins}_plans" for ins in _ALL_INSURERS],
            query            = enriched_query,
            top_k_each       = 3,
            hyde             = False,
            language         = language,
        )
        docs = sorted(
            [doc for results in multi.values() for doc in results],
            key     = lambda d: d.get("score", 0),
            reverse = True,
        )[:5]

    # ── Step 4: LLM 답변 + 연관질문 병렬 생성 ────────────────────
    answer, related_questions = call_llm_parallel(
        user_query     = user_msg,
        retrieved_docs = docs,
        language       = language,
        extra_context  = slots,
        system_prompt  = _PROCEDURE_SYSTEM_PROMPT,
    )

    return {
        "retrieved_docs"   : docs,
        "answer"           : answer,
        "sources"          : _build_sources(docs),
        "related_questions": related_questions,
        "chat_history"     : chat_history + [{"role": "assistant", "content": answer}],
    }
