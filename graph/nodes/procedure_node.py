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
#   1. {insurer}_plans 컬렉션에서 RAG 검색
#   2. 단계별 절차 프롬프트 조립
#   3. LLM 으로 단계별 안내 + 필요 서류 목록 생성
#
# 참고: insurer 는 analyze_node 에서 항상 확정되므로
#       general_guidelines 컬렉션은 사용하지 않는다.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

from graph.nodes.generate_node import call_llm_with_docs
from graph.nodes.retrieve_node import query_collection
from utils.schemas import InsuranceState

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
        retrieved_docs : 검색된 절차 문서 리스트
        answer         : 단계별 절차 안내 + 필요 서류 목록
    """
    user_msg = state["user_message"]
    language = state.get("language", "en")
    insurer  = state.get("insurer", "")
    slots    = state.get("slots", {})

    # ── Step 1: RAG 검색 ───────────────────────────────────────
    # insurer 는 analyze_node 에서 항상 확정되므로 전용 컬렉션만 검색
    docs = query_collection(
        collection_name = f"{insurer}_plans",
        query           = user_msg,
        top_k           = 5,
    )

    # ── Step 2: LLM 단계별 절차 생성 ──────────────────────────
    answer = call_llm_with_docs(
        user_query     = user_msg,
        retrieved_docs = docs,
        language       = language,
        extra_context  = slots,
        system_prompt  = _PROCEDURE_SYSTEM_PROMPT,
    )

    return {
        "retrieved_docs": docs,
        "answer"        : answer,
    }
