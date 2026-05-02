# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# graph/nodes/nhis_node.py
# 역할 : NHIS(국민건강보험) 관련 멀티턴 상담을 처리한다
#
# 파이프라인: ⑤ NHIS
# 진입 조건 : analyze_node 에서 intent == "nhis"
# 다음 노드  : END (일반 안내 완료 시)
#              claim_node (NHIS 적용 후 민간보험 청구로 이어질 때)
#
# 흐름 (멀티턴):
#   [1턴] nhis_step == "eligibility_check"
#         → 대상자 자격 확인 Q&A (재질문 또는 자격 판단)
#   [2턴] nhis_step == "info"
#         → NHIS 정보 RAG 검색 + 안내 (자격·보험료·급여범위·이용절차)
#   [3턴] nhis_step == "claim_link" (사용자가 민간보험 청구 원하는 경우)
#         → claim_node 로 라우팅 신호 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import json
import os

from openai import OpenAI

from graph.nodes.generate_node import call_llm_with_docs, _build_sources, _call_llm_for_related_questions
from graph.nodes.retrieve_node import query_collection
from utils.schemas import InsuranceState

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────

# NHIS 자격 확인 프롬프트
_ELIGIBILITY_SYSTEM_PROMPT = """You are an NHIS (National Health Insurance Service of Korea) eligibility checker.
Your task is to determine if the user is eligible for NHIS coverage.

Ask the user about (one question at a time):
1. Visa type (D-series, E-series, F-series, H-series, etc.)
2. Employment status (employed at a Korean company / self-employed / dependent of an insured employee)
3. Duration of stay in Korea (if not employed: must be 6+ months for mandatory enrollment)

Categories:
- 직장가입자 (Employee Insured): working at an NHIS-covered company
- 지역가입자 (Self-employed Insured): staying 6+ months with an eligible visa
- 피부양자 (Dependent): dependent of an employee insured

IMPORTANT: Always respond in this exact JSON format:
{
  "eligible": true | false | null,
  "response": "message to show to the user"
}
- eligible: true = eligible / false = not eligible / null = need more information, ask next question
- response: the message to show to the user (in the user's language)"""

# 자격 확인 최대 턴 수 — 초과 시 강제로 info 단계로 전환
_MAX_ELIGIBILITY_TURNS = 4

# NHIS 정보 안내 프롬프트
_NHIS_INFO_SYSTEM_PROMPT = """You are an NHIS (Korean National Health Insurance) specialist.
Provide accurate information based ONLY on the provided documents.
Cover these topics as relevant to the user's question:
- Eligibility & enrollment (자격 및 가입)
- Premium calculation (보험료)
- Benefit coverage (급여 범위)
- How to use / claim procedures (이용 및 청구 절차)

Always mention: "For the most up-to-date information, please visit nhis.or.kr or call 1577-1000." """

# 민간보험 연계 감지 키워드
_PRIVATE_INSURANCE_KEYWORDS = [
    "민간보험", "개인보험", "실손", "실비", "private insurance",
    "supplemental", "additional claim", "secondary claim",
    "나머지", "추가 청구", "잔액",
]


# ──────────────────────────────────────────────────────────────
# 노드 함수
# ──────────────────────────────────────────────────────────────

def nhis(state: InsuranceState) -> dict:
    """
    [파이프라인 ⑤] NHIS 상담을 멀티턴으로 처리한다.

    읽는 state 필드:
        user_message  : 사용자 원문 질의
        language      : 응답 언어 코드
        nhis_step     : 현재 NHIS 대화 단계
                        "eligibility_check" | "info" | "claim_link" | "done"
        nhis_eligible : 자격 확인 결과 (None=미확인, True=자격있음, False=없음)
        slots         : 추출된 슬롯

    반환 dict (InsuranceState 업데이트):
        nhis_step         : 업데이트된 대화 단계
        nhis_eligible     : 자격 확인 결과 (업데이트 시)
        retrieved_docs    : 검색된 NHIS 문서
        answer            : 이번 턴의 응답 텍스트
        intent            : "claim" 으로 변경 (민간보험 연계 감지 시)
        sources           : 참조 문서 출처 리스트
        related_questions : 연관 질문 리스트
    """
    user_msg      = state["user_message"]
    language      = state.get("language", "en")
    nhis_step     = state.get("nhis_step", "eligibility_check")
    nhis_eligible = state.get("nhis_eligible", None)
    nhis_history  = state.get("nhis_history") or []

    # ── 민간보험 연계 감지 ─────────────────────────────────────
    # 어느 단계에서든 민간보험 청구 의도 감지 시 claim_node 로 이동
    if _wants_private_claim(user_msg):
        docs              = query_collection("claim_procedures", user_msg, top_k=3)
        bridge_answer     = _claim_bridge_message(language)
        sources           = _build_sources(docs)
        related_questions = _call_llm_for_related_questions(
            user_query = user_msg,
            answer     = bridge_answer,
            language   = language,
        )
        return {
            "nhis_step"        : "claim_link",
            "intent"           : "claim",       # builder 가 다음 분기 결정에 사용
            "retrieved_docs"   : docs,
            "answer"           : bridge_answer,
            "sources"          : sources,
            "related_questions": related_questions,
        }

    # ── 단계별 처리 ────────────────────────────────────────────
    if nhis_step == "eligibility_check":
        return _handle_eligibility(user_msg, language, nhis_history)

    if nhis_step == "info":
        return _handle_info(user_msg, language, nhis_eligible)

    # "done" 또는 알 수 없는 단계 — 기본 안내
    return _handle_info(user_msg, language, nhis_eligible)


# ──────────────────────────────────────────────────────────────
# 단계별 처리 함수
# ──────────────────────────────────────────────────────────────

def _handle_eligibility(user_msg: str, language: str, nhis_history: list) -> dict:
    """
    [Step 1] 대상자 자격 확인 단계.

    LLM 이 사용자와 대화하며 자격 여부를 판단한다.
    - nhis_history 로 이전 대화 이력을 LLM 에 전달해 멀티턴 대화를 지원한다.
    - 최대 _MAX_ELIGIBILITY_TURNS 턴 초과 시 강제로 info 단계로 전환한다.
    - 자격 판단 완료되면 nhis_step 을 "info" 로 전환한다.
    """
    # ── 최대 턴 수 초과 → 강제 info 전환 (무한 루프 방지) ────────
    user_turn_count = sum(1 for m in nhis_history if m["role"] == "user")
    if user_turn_count >= _MAX_ELIGIBILITY_TURNS:
        return {
            "nhis_step"        : "info",
            "nhis_eligible"    : None,          # 판단 불가 상태로 info 진입
            "nhis_history"     : nhis_history,
            "retrieved_docs"   : [],
            "answer"           : _eligibility_max_turn_message(language),
            "sources"          : [],
            "related_questions": [],
        }

    try:
        client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # ── 대화 이력 포함해서 LLM 호출 ──────────────────────────
        messages = [
            {"role": "system", "content": _ELIGIBILITY_SYSTEM_PROMPT},
            *nhis_history,                      # 이전 대화 이력
            {"role": "user",   "content": user_msg},
        ]

        response = client.chat.completions.create(
            model           = "gpt-4o",
            messages        = messages,
            max_tokens      = 600,
            temperature     = 0.1,
            response_format = {"type": "json_object"},
        )
        raw    = response.choices[0].message.content
        result = json.loads(raw)

        is_eligible   = result.get("eligible", None)  # true / false / null
        response_text = result.get("response", "")    # 사용자에게 보낼 메시지

        # ── 대화 이력 업데이트 ────────────────────────────────────
        new_history = nhis_history + [
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": response_text},
        ]

        # 자격 판단 완료 → info 전환 / 미완료 → eligibility_check 유지
        next_step         = "info" if is_eligible is not None else "eligibility_check"
        related_questions = _call_llm_for_related_questions(
            user_query = user_msg,
            answer     = response_text,
            language   = language,
        )

        return {
            "nhis_step"        : next_step,
            "nhis_eligible"    : is_eligible,
            "nhis_history"     : new_history,
            "retrieved_docs"   : [],
            "answer"           : response_text,
            "sources"          : [],
            "related_questions": related_questions,
        }

    except Exception:
        # JSON 파싱 오류 등 — 자격 확인 질문으로 fallback (이력은 유지)
        fallback_answer = _eligibility_fallback(language)
        return {
            "nhis_step"        : "eligibility_check",
            "nhis_eligible"    : None,
            "nhis_history"     : nhis_history,
            "retrieved_docs"   : [],
            "answer"           : fallback_answer,
            "sources"          : [],
            "related_questions": [],
        }


def _handle_info(
    user_msg      : str,
    language      : str,
    nhis_eligible : bool | None,
) -> dict:
    """
    [Step 2] NHIS 정보 안내 단계.

    nhis_docs 컬렉션에서 RAG 검색 후 정보를 제공한다.
    자격이 없는 경우 별도 안내를 추가한다.
    """
    # NHIS 문서에서 관련 내용 검색
    docs = query_collection(
        collection_name = "nhis_plans",
        query           = user_msg,
        top_k           = 5,
    )

    # 자격 없는 사용자용 추가 컨텍스트
    extra = {}
    if nhis_eligible is False:
        extra["eligibility_note"] = (
            "이 사용자는 NHIS 적용 대상이 아닙니다. "
            "해당 사실을 인지하고 민간보험 활용을 안내해 주세요."
        )

    answer = call_llm_with_docs(
        user_query     = user_msg,
        retrieved_docs = docs,
        language       = language,
        extra_context  = extra,
        system_prompt  = _NHIS_INFO_SYSTEM_PROMPT,
    )

    sources           = _build_sources(docs)
    related_questions = _call_llm_for_related_questions(
        user_query = user_msg,
        answer     = answer,
        language   = language,
    )

    return {
        "nhis_step"        : "info",
        "retrieved_docs"   : docs,
        "answer"           : answer,
        "sources"          : sources,
        "related_questions": related_questions,
    }


# ──────────────────────────────────────────────────────────────
# 내부 유틸 함수
# ──────────────────────────────────────────────────────────────

def _wants_private_claim(user_msg: str) -> bool:
    """사용자가 NHIS 적용 후 민간보험 청구를 원하는지 감지한다."""
    lower = user_msg.lower()
    return any(kw.lower() in lower for kw in _PRIVATE_INSURANCE_KEYWORDS)


def _claim_bridge_message(language: str) -> str:
    """NHIS → 민간보험 청구 연계 안내 메시지"""
    messages = {
        "ko": (
            "NHIS 적용 후 민간보험 청구를 원하시는군요. "
            "청구 절차와 필요 서류를 안내해 드리겠습니다."
        ),
        "en": (
            "I'll help you with the private insurance claim after NHIS coverage. "
            "Let me guide you through the claim procedure and required documents."
        ),
    }
    return messages.get(language, messages["en"])


def _eligibility_max_turn_message(language: str) -> str:
    """최대 턴 수 초과 시 info 단계로 넘어가며 보내는 안내 메시지"""
    messages = {
        "ko": (
            "정확한 자격 확인을 위해 추가 정보가 필요하지만, "
            "일반적인 NHIS 정보를 먼저 안내해 드리겠습니다. "
            "자세한 자격 확인은 NHIS 고객센터(1577-1000)에 문의해 주세요."
        ),
        "en": (
            "I wasn't able to fully confirm your eligibility, "
            "but let me provide general NHIS information. "
            "For a detailed eligibility check, please contact NHIS at 1577-1000."
        ),
    }
    return messages.get(language, messages["en"])


def _eligibility_fallback(language: str) -> str:
    """자격 확인 오류 시 기본 질문 메시지"""
    messages = {
        "ko": (
            "NHIS 적용 가능 여부를 확인하겠습니다.\n"
            "다음 정보를 알려주세요:\n"
            "1. 한국 국적 여부 (내국인 / 외국인)\n"
            "2. 체류 비자 유형 (외국인의 경우)\n"
            "3. 고용 형태 (직장인 / 자영업자 / 피부양자)"
        ),
        "en": (
            "Let me check your NHIS eligibility.\n"
            "Please provide:\n"
            "1. Nationality (Korean citizen / Foreigner)\n"
            "2. Visa type (if foreigner)\n"
            "3. Employment status (employed / self-employed / dependent)"
        ),
    }
    return messages.get(language, messages["en"])
