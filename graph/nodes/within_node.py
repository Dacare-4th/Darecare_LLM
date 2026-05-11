# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# graph/nodes/within_node.py
# 역할 : 동일 보험사 내 플랜을 비교한다
#
# 파이프라인: ① 보험 내 비교
# 진입 조건 : analyze_node 에서 intent == "within_compare"
# 다음 노드  : END
#
# 흐름:
#   1. insurer 유효성 검증 (_validate_insurer)
#   2. 슬롯에서 플랜 목록 추출 (_resolve_plans)
#   3. 플랜별 RAG 검색 (_search_per_plan)
#   4. 비교 프롬프트 조립 (build_comparison_prompt)
#   5. LLM 직접 호출 — JSON 강제 (_call_llm_json)
#   6. 응답 파싱 (parse_compare_table)
#   7. sources 추출 후 state 반환
#
# [변경 내역]
#   - call_llm_with_docs() 제거 → LLM 직접 호출(_call_llm_json)로 교체
#     이유: response_format={"type":"json_object"} 를 써야 JSON 파싱이 안정적이며,
#           call_llm_with_docs()는 해당 파라미터를 지원하지 않음
#   - _validate_insurer() 추가: 지원 보험사 코드 검증
#   - _resolve_plans() 추가: slots에서 비교할 플랜 목록 추출
#   - _search_per_plan() 추가: 플랜별 RAG 검색 로직 분리
#   - 반환 dict에 compare_table, related_questions, sources 추가
#     이유: 프론트엔드가 비교표 렌더링 + 관련 질문 + 출처를 구조화된 데이터로 받아야 함
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import json   # LLM fallback 응답 직렬화에 사용
import os     # OPENAI_API_KEY 환경변수 읽기

from openai import OpenAI  # OpenAI 직접 import

# parse_compare_table 추가
from graph.nodes.retrieve_node import query_collection
from utils.comparison import build_comparison_prompt, parse_compare_table
from utils.schemas import InsuranceState


# 상수
# 지원 보험사 코드 집합 추가
#insurer 슬롯에 엉뚱한 값이 들어왔을 때 일반 컬렉션으로 fallback하기 위함
_SUPPORTED_INSURERS = {"uhcg", "cigna", "tricare", "msh_china"}

# 시스템 프롬프트 변경
# response_format={"type":"json_object"} 와 함께 일관성 유지
_WITHIN_SYSTEM_PROMPT = """You are a health insurance plan comparison specialist.
Compare the insurance plans based ONLY on the provided documents.
You MUST respond with valid JSON only — no other text, no markdown.
Highlight key differences that would impact the user's decision.
If information is missing for a plan, state "정보 없음"."""



# 노드 함수

def within(state: InsuranceState) -> dict:
    """
    [파이프라인 ①] 동일 보험사 내 플랜 비교표를 생성한다.

    읽는 state 필드:
        user_message : 사용자 원문 질의
        language     : 응답 언어 코드
        insurer      : 비교할 보험사 코드 (예: "uhcg")
        slots        : {"plan": "Gold"} 또는 {"plans": ["Gold", "Silver"]} 등

    반환 dict (InsuranceState 업데이트):
        retrieved_docs    : 검색된 문서 리스트
        answer            : 자연어 비교 요약 (LLM JSON의 "answer" 필드)
        compare_table     : {"header": [...], "body": [[...]]} 비교표 구조체
        related_questions : 관련 질문 3개 리스트
        sources           : 출처 정보 리스트 (최대 5개)
    """
    user_msg      = state["user_message"]
    language      = state.get("language", "en")
    slots         = state.get("slots", {})
    english_query = state.get("english_query", "") or user_msg
    chat_history  = state.get("chat_history", [])

    # insurer 유효성 검증
    #_validate_insurer() 로 지원 여부 확인 후 사용
    insurer = _validate_insurer(state)

    #비교할 플랜 목록 추출
    # _resolve_plans() 로 단일/복수 플랜 모두 처리
    plans = _resolve_plans(slots)

    # insurer 미확정이면 비교 불가 안내
    if not insurer:
        _answer = ("어떤 보험사의 플랜을 비교하시겠어요? (예: UHCG, Cigna, Tricare, MSH China)\n"
                   "Which insurer's plans would you like to compare? (e.g., UHCG, Cigna, Tricare, MSH China)")
        return {
            "retrieved_docs"   : [],
            "answer"           : _answer,
            "compare_table"    : {"header": [], "body": []},
            "related_questions": [],
            "sources"          : [],
            "chat_history"     : chat_history + [{"role": "assistant", "content": _answer}],
        }

    # 플랜 미지정 시 플랜 목록 안내 후 비교 유도 (Option B)
    if not plans:
        result = _guide_plan_selection(insurer, language)
        result["chat_history"] = chat_history + [{"role": "assistant", "content": result.get("answer", "")}]
        return result

    #  플랜별 RAG 검색
    #_search_per_plan() 로 분리 재사용 가능
    docs_by_plan, all_retrieved = _search_per_plan(insurer, plans, user_msg, english_query)

    #비교 프롬프트 조립 
    # text_by_plan: {플랜명: [문서 텍스트, ...]} 형태로 변환
    text_by_plan: dict[str, list[str]] = {
        subject: [d["content"] for d in docs]
        for subject, docs in docs_by_plan.items()
    }
    comparison_prompt = build_comparison_prompt(
        docs_by_subject = text_by_plan,
        user_query      = user_msg,
        language        = language,
    )

    # LLM 직접 호출 (JSON 강제) 
    # [수정]call_llm_with_docs() 사용 -> 자연어 텍스트 반환
    # _call_llm_json() 사용 ->  response_format={"type":"json_object"} 적용
    raw_response = _call_llm_json(comparison_prompt)

    # JSON 파싱 
    # parse_compare_table() 로 (compare_table, answer, related_questions) 추출
    compare_table, answer, related_questions = parse_compare_table(raw_response)

    # sources 추출 
    # 05.01 - file_name → source 로 변경 (팀 공통 메타데이터 형식: source = 파일명)
    # - PDF 문서: source 사용 (uhcg, tricare 모두 source = 파일명으로 저장)
    # - 웹 문서 : url 사용
    # - 최대 5개까지만 포함 (프론트엔드 표시 제한)
    sources = [
        {
            "document_name": doc["metadata"].get("source") or doc["metadata"].get("url", ""),
            "page"         : doc["metadata"].get("page") or None,
            "section"      : doc["metadata"].get("topic") or "",
        }
        for doc in all_retrieved[:5]
    ]

    # [수정] 반환 dict에 compare_table, related_questions, sources 추가
    # 기존: {"retrieved_docs": ..., "answer": ...} 만 반환
    # 변경: 세 필드 추가 → state["compare_table"], ["related_questions"], ["sources"] 업데이트
    return {
        "retrieved_docs"   : all_retrieved,
        "answer"           : answer,
        "compare_table"    : compare_table,
        "related_questions": related_questions,
        "sources"          : sources,
        "chat_history"     : chat_history + [{"role": "assistant", "content": answer}],
    }


_LIST_PLANS_PROMPT = {
    "ko": "위 문서를 바탕으로 {insurer}에서 제공하는 보험 플랜 종류를 목록으로 알려주고, 어떤 플랜끼리 비교할지 물어보세요. 간결하게 작성하세요.",
    "en": "Based on the documents above, list the available plans for {insurer} and ask which plans the user wants to compare. Be concise.",
}


def _guide_plan_selection(insurer: str, language: str) -> dict:
    """플랜 미지정 시 플랜 목록을 보여주고 비교할 플랜을 안내한다."""
    from graph.nodes.retrieve_node import query_collection
    from openai import OpenAI
    import os

    docs = query_collection(
        collection_name = f"{insurer}_plans",
        query           = f"{insurer} plan options coverage types",
        top_k           = 6,
    )
    context = "\n\n".join(d["content"][:400] for d in docs if d.get("content"))

    lang_prompt = _LIST_PLANS_PROMPT.get(language, _LIST_PLANS_PROMPT["en"])
    prompt = f"{context}\n\n{lang_prompt.format(insurer=insurer.upper())}"

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model       = "gpt-4o-mini",
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = 300,
            temperature = 0.3,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception:
        answer = f"어떤 플랜끼리 비교하시겠어요? (예: Gold, Silver, Basic)" if language == "ko" \
                 else "Which plans would you like to compare? (e.g. Gold, Silver, Basic)"

    return {
        "retrieved_docs"   : docs,
        "answer"           : answer,
        "compare_table"    : {"header": [], "body": []},
        "related_questions": [],
        "sources"          : [],
    }


# 내부[수정] 신규 추가 3개
def _validate_insurer(state: InsuranceState) -> str:
    """
    [신규] state["insurer"] 가 지원 보험사 코드인지 검증한다.

    지원 코드: uhcg, cigna, tricare, msh_china
    지원하지 않는 값이면 "" 를 반환 → 검색 시 general_guidelines 컬렉션으로 fallback
    """
    insurer = state.get("insurer", "").lower()
    return insurer if insurer in _SUPPORTED_INSURERS else ""


def _resolve_plans(slots: dict) -> list[str]:
    """
    [신규] slots 에서 비교할 플랜 이름 리스트를 추출한다.

    우선순위:
        1. slots["plans"] — 복수 플랜 리스트 (예: ["Gold", "Silver"])
        2. slots["plan"]  — 단일 플랜 문자열 (예: "Gold")
        3. 없으면 [] 반환 → 전체 플랜 비교로 fallback

    기존 로직(slots.get("plan") 단일 처리)을 일반화한 버전
    """
    plans = slots.get("plans", [])
    if isinstance(plans, list) and plans:
        return plans

    plan = slots.get("plan", "")
    if not plan:
        return []

    # LLM이 "Prime, Select"처럼 쉼표 구분 단일 문자열로 반환한 경우 분리
    # 예: "TRICARE Prime, TRICARE Select" -> ["TRICARE Prime", "TRICARE Select"]
    if "," in plan:
        return [p.strip() for p in plan.split(",") if p.strip()]

    return [plan]


def _search_per_plan(
    insurer: str,
    plans: list[str],
    user_msg: str,
    english_query: str = "",
) -> tuple[dict[str, list[dict]], list[dict]]:
    """
    [신규] 플랜별로 RAG 검색을 수행하고 결과를 반환한다.

    기존 within() 에 인라인으로 있던 검색 로직을 분리.
    재사용성 향상 + within() 함수 가독성 개선이 목적.

    Args:
        insurer       : 보험사 코드 (빈 문자열이면 general_guidelines 컬렉션 사용)
        plans         : 비교할 플랜 이름 리스트 (빈 리스트면 전체 검색)
        user_msg      : 사용자 원문 질의
        english_query : analyze_node에서 생성된 영어 검색 쿼리 (없으면 user_msg 사용)

    Returns:
        docs_by_plan  : {플랜명: [doc_dict, ...]}
        all_retrieved : 모든 플랜 문서를 합친 단일 리스트
    """
    collection_name = f"{insurer}_plans" if insurer else "general_guidelines"
    base_query = english_query or user_msg

    docs_by_plan: dict[str, list[dict]] = {}

    if plans:
        # 플랜이 명시된 경우: 플랜별로 개별 검색
        for plan in plans:
            docs = query_collection(
                collection_name = collection_name,
                query           = f"{base_query} {plan}",
                top_k           = 5,
                where           = {"plan": plan},
            )
            # 05.01 - where 필터로 결과가 없으면 필터 없이 재검색
            # uhcg_plans 등 plan 메타데이터가 빈 문자열('')인 경우
            # where={"plan": "Gold"} 조건이 매칭되지 않아 항상 0건 반환됨.
            # 필터 없이 재검색하면 전체 컬렉션에서 플랜명 포함 쿼리로 검색 가능.
            if not docs:
                docs = query_collection(
                    collection_name = collection_name,
                    query           = f"{base_query} {plan}",
                    top_k           = 5,
                )
            docs_by_plan[plan] = docs
    else:
        # 플랜 미명시: 전체 컬렉션 검색 후 통합 비교
        label = insurer.upper() if insurer else "Insurance Plans"
        docs_by_plan[label] = query_collection(
            collection_name = collection_name,
            query           = base_query,
            top_k           = 8,
        )

    all_retrieved = [doc for docs in docs_by_plan.values() for doc in docs]
    return docs_by_plan, all_retrieved


def _call_llm_json(prompt: str) -> str:
    """
    [신규] JSON 출력을 강제하는 LLM 호출 함수.

    기존 call_llm_with_docs() 와의 차이:
        - response_format={"type": "json_object"} 적용
          → GPT-4o 가 반드시 유효한 JSON 만 반환하도록 보장
        - max_tokens 2000 (비교표 데이터 특성상 출력량이 많아 1500 → 2000 으로 증가)
        - 오류 시 빈 compare_table 구조의 JSON 반환 (parse_compare_table() 가 처리 가능한 형태)
    """
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model           = "gpt-4o",
            messages        = [
                {"role": "system", "content": _WITHIN_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            # [핵심 변경] response_format 추가: JSON 파싱 실패 가능성을 구조적으로 차단
            response_format = {"type": "json_object"},
            max_tokens      = 2000,
            temperature     = 0.1,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        # LLM 호출 자체가 실패한 경우: parse_compare_table() 가 처리할 수 있는 JSON 반환
        return json.dumps({
            "header"           : [],
            "body"             : [],
            "answer"           : f"응답 생성 중 오류가 발생했습니다. ({type(e).__name__})",
            "related_questions": [],
        }, ensure_ascii=False)
