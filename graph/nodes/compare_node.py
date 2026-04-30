# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# graph/nodes/compare_node.py
# 역할 : 여러 보험사를 동시에 비교한다
#
# 파이프라인: ② 보험사 간 비교
# 진입 조건 : analyze_node 에서 intent == "cross_compare"
# 다음 노드  : END
#
# 흐름:
#   1. insurers 리스트의 각 컬렉션에서 병렬 RAG 검색
#   2. 결과 병합 + Re-ranking
#   3. 비교 프롬프트 조립
#   4. LLM 으로 보험사 비교표 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

from graph.nodes.generate_node import call_llm_with_docs, _build_sources, _call_llm_for_related_questions
from graph.nodes.retrieve_node import query_multi_collections
from utils.comparison import build_comparison_prompt, rerank_by_relevance
from utils.schemas import InsuranceState

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────
_CROSS_SYSTEM_PROMPT = """You are a health insurance comparison specialist.
Compare the insurance companies based ONLY on the provided documents.
Present a clear comparison table with the following sections:
1. Coverage & Benefits
2. Cost Structure (premiums, deductibles, copay)
3. Network & Access
4. Claim Process
5. Notable Advantages / Disadvantages

For each item, clearly label which insurer it belongs to.
If information is missing, state "Not available in documents".
Conclude with a neutral summary of each insurer's strengths."""


# ──────────────────────────────────────────────────────────────
# 노드 함수
# ──────────────────────────────────────────────────────────────

def compare(state: InsuranceState) -> dict:
    """
    [파이프라인 ②] 여러 보험사를 비교하는 답변을 생성한다.

    읽는 state 필드:
        user_message : 사용자 원문 질의
        language     : 응답 언어 코드
        insurers     : 비교할 보험사 코드 리스트 (예: ["uhcg", "cigna"])
        insurer      : insurers 가 비어있을 때 fallback 단일 보험사
        slots        : 추출된 슬롯 (treatment, plan 등)

    반환 dict (InsuranceState 업데이트):
        retrieved_docs    : 모든 보험사의 검색 문서 통합 리스트
        answer            : 보험사 비교표가 포함된 최종 응답
        sources           : 참조 문서 출처 리스트
        related_questions : 연관 질문 리스트
    """
    user_msg = state["user_message"]
    language = state.get("language", "en")
    insurers = state.get("insurers", [])
    slots    = state.get("slots", {})

    # ── 비교 대상 보험사 결정 ──────────────────────────────────
    if not insurers:
        # insurers 가 비어있으면 insurer 단일값 + 전체 컬렉션 fallback
        single = state.get("insurer", "")
        insurers = [single] if single else []

    if not insurers:
        # 비교 대상이 없으면 지원 보험사 전체를 대상으로 설정
        insurers = ["uhcg", "cigna", "tricare", "msh_china"]

    # ── Step 1: 보험사별 컬렉션 이름 매핑 ─────────────────────
    # NHIS 는 별도 컬렉션명 사용, 나머지는 {insurer}_plans
    collection_map: dict[str, str] = {
        ins: ("nhis" if ins == "nhis" else f"{ins}_plans")
        for ins in insurers
    }

    # ── Step 2: 병렬 멀티 컬렉션 RAG 검색 ─────────────────────
    results_by_collection = query_multi_collections(
        collection_names = list(collection_map.values()),
        query            = user_msg,
        top_k_each       = 5,
    )

    # 컬렉션명 → 보험사명으로 키 역매핑
    col_to_ins = {v: k for k, v in collection_map.items()}
    docs_by_insurer: dict[str, list[str]] = {}
    all_retrieved: list[dict] = []

    for col_name, docs in results_by_collection.items():
        insurer_name = col_to_ins.get(col_name, col_name).upper()
        # Re-ranking: 표 문서 우선 배치
        ranked       = rerank_by_relevance(
            docs      = [d["content"]  for d in docs],
            metadatas = [d["metadata"] for d in docs],
            top_k     = 5,
        )
        docs_by_insurer[insurer_name] = [d["content"] for d in ranked]
        all_retrieved.extend(ranked)

    # ── Step 3: 비교 프롬프트 조립 ─────────────────────────────
    comparison_prompt = build_comparison_prompt(
        docs_by_subject = docs_by_insurer,
        user_query      = user_msg,
        language        = language,
    )

    # ── Step 4: LLM 비교표 생성 ────────────────────────────────
    answer = call_llm_with_docs(
        user_query     = comparison_prompt,
        retrieved_docs = all_retrieved,
        language       = language,
        extra_context  = slots,
        system_prompt  = _CROSS_SYSTEM_PROMPT,
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
