# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# graph/nodes/analyze_node.py
# 역할 : 사용자 질의를 분석한다 (그래프의 진입점)
#
# 처리 순서:
#   1. 안전 필터 (check_blocked) — 악성/무관 요청 즉시 차단
#   2. 언어 감지 (detect_language)
#   3. Intent Router — LLM 으로 의도·슬롯 추출
#
# 진입 조건 : 모든 요청의 첫 번째 노드 (set_entry_point)
# 다음 노드  : builder.route_intent() 함수가 intent 값으로 결정
#              "within_compare" → within_node
#              "cross_compare"  → compare_node
#              "calculation"    → calculate_node
#              "procedure"      → retrieve_node
#              "nhis"           → nhis_node
#              "claim"          → claim_node
#              "general_query"  → general_node
#              "clarify"        → clarify_node
#              "blocked"        → END
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import json
import os

from openai import OpenAI

from utils.language import detect_language
from utils.safety import check_blocked
from utils.schemas import InsuranceState, Intent

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────
_MAX_MESSAGE_LENGTH = 500

# ──────────────────────────────────────────────────────────────
# Intent Router 시스템 프롬프트
# ──────────────────────────────────────────────────────────────
_INTENT_SYSTEM_PROMPT = """You are an intent classifier and slot extractor for a health insurance assistant.

Supported intents:
- within_compare  : User wants to compare plans within ONE specific insurer
- cross_compare   : User wants to compare MULTIPLE insurers against each other
- calculation     : User asks about exchange rates, copay amounts, or cost calculation
- procedure       : User asks about general insurance procedures or processes
- nhis            : User asks about NHIS (National Health Insurance Service / 국민건강보험)
- claim           : User wants to know about claim procedures or needs a claim form
- general_query   : General coverage or benefit inquiry that does not fit any category above
                    (e.g. "Is mental health covered?", "Does my plan cover dental?")
- recommendation  : User asks for insurance product recommendations, asks the AI to choose a plan,
                    or requests legal/medical final judgments (e.g. "어떤 보험이 좋아?", "이 보험 들어야 해?",
                    "내 증상은 어떤 병이야?", "소송에서 이길 수 있어?")
- clarify         : Not enough information to determine intent (ask user for more info)

Supported insurer codes: uhcg, cigna, tricare, msh_china

Respond ONLY with valid JSON in this exact format:
{
  "intents"      : ["primary_intent", "optional_second_intent"],
  "insurer"      : "insurer_code or empty string",
  "insurers"     : ["insurer1", "insurer2"],
  "slots"        : {
    "plan"       : "plan name or empty",
    "treatment"  : "treatment type or empty",
    "amount"     : 0,
    "currency"   : "USD or empty",
    "region"     : "region or empty"
  },
  "missing_slots": ["list of required but missing slot names"]
}

Rules:
- intents[0] is the PRIMARY intent used for routing
- For calculation: extract amount and currency from slots if mentioned
- For cross_compare: list all mentioned insurers in insurers[]
- missing_slots should list slots that are REQUIRED but not provided by user
- If truly ambiguous, use "clarify" as the intent"""


# ──────────────────────────────────────────────────────────────
# 노드 함수
# ──────────────────────────────────────────────────────────────

def analyze(state: InsuranceState) -> dict:
    """
    [진입점] 사용자 질의를 분석해 언어·의도·슬롯을 추출한다.

    읽는 state 필드:
        user_message : 사용자 원문 질의

    반환 dict (InsuranceState 업데이트):
        language      : 감지된 언어 코드
        intent        : 주 의도 (Intent 상수 참조)
        intents       : 복합 의도 리스트
        insurer       : 단일 보험사 코드 (파이프라인 ① 용)
        insurers      : 복수 보험사 코드 리스트 (파이프라인 ② 용)
        slots         : 추출된 슬롯 dict
        missing_slots : 누락된 필수 슬롯 목록
        answer        : 차단된 경우에만 오류 메시지 설정
    """
    user_msg = state["user_message"]

    # ── Step 0: 메시지 길이 검증 (최대 500자) ─────────────────
    if len(user_msg) > _MAX_MESSAGE_LENGTH:
        return {
            "intent" : Intent.BLOCKED,
            "intents": [Intent.BLOCKED],
            "answer" : (
                f"질문은 최대 {_MAX_MESSAGE_LENGTH}자까지 입력 가능합니다. "
                f"현재 {len(user_msg)}자입니다.\n\n"
                f"Your message exceeds the {_MAX_MESSAGE_LENGTH}-character limit "
                f"({len(user_msg)} chars)."
            ),
        }

    # ── Step 1: 안전 필터 ──────────────────────────────────────
    blocked_msg = check_blocked(user_msg)
    if blocked_msg:
        return {
            "intent" : Intent.BLOCKED,
            "intents": [Intent.BLOCKED],
            "answer" : blocked_msg,
        }

    # ── Step 2: 언어 감지 ──────────────────────────────────────
    language = detect_language(user_msg)

    # ── Step 3: Intent Router (LLM) ────────────────────────────
    analysis = _run_intent_router(user_msg)

    intents  = analysis.get("intents", [Intent.CLARIFY])
    primary  = intents[0] if intents else Intent.CLARIFY

    # ── Step 4: 추천/법적·의학적 판단 요청 차단 ───────────────
    if primary == Intent.RECOMMENDATION:
        return {
            "language"     : language,
            "intent"       : Intent.RECOMMENDATION,
            "intents"      : intents,
            "insurer"      : analysis.get("insurer", ""),
            "insurers"     : analysis.get("insurers", []),
            "slots"        : analysis.get("slots", {}),
            "missing_slots": analysis.get("missing_slots", []),
            "answer"       : (
                "죄송합니다. 보험 상품 추천, 플랜 선택 권유, 법적·의학적 최종 판단은 제공하지 않습니다. "
                "보험 혜택·절차·청구 등 구체적인 질문을 해주세요.\n\n"
                "Sorry, we do not provide insurance product recommendations, plan selection advice, "
                "or legal/medical final judgments. "
                "Please ask about coverage details, procedures, or claims."
            ),
        }

    return {
        "language"     : language,
        "intent"       : primary,
        "intents"      : intents,
        "insurer"      : analysis.get("insurer", ""),
        "insurers"     : analysis.get("insurers", []),
        "slots"        : analysis.get("slots", {}),
        "missing_slots": analysis.get("missing_slots", []),
    }


# ──────────────────────────────────────────────────────────────
# 내부 함수
# ──────────────────────────────────────────────────────────────

def _run_intent_router(user_msg: str) -> dict:
    """
    LLM 을 호출해 의도·슬롯을 추출한다.

    LLM 응답이 올바른 JSON 이 아닐 경우 clarify 를 반환한다.
    """
    try:
        client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model    = "gpt-4o-mini",
            messages = [
                {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens       = 300,
            temperature      = 0,
            response_format  = {"type": "json_object"},  # JSON 모드 강제
        )
        raw    = response.choices[0].message.content
        result = json.loads(raw)

        # intents 값 검증 — 허용된 Intent 상수만 통과
        valid_intents = {
            Intent.WITHIN_COMPARE, Intent.CROSS_COMPARE,
            Intent.CALCULATION, Intent.PROCEDURE,
            Intent.NHIS, Intent.CLAIM, Intent.GENERAL_QUERY,
            Intent.RECOMMENDATION, Intent.CLARIFY,
        }
        result["intents"] = [
            i for i in result.get("intents", []) if i in valid_intents
        ] or [Intent.CLARIFY]

        return result

    except Exception as e:
        # LLM 오류 또는 JSON 파싱 실패 → clarify 로 처리
        return {
            "intents"      : [Intent.CLARIFY],
            "insurer"      : "",
            "insurers"     : [],
            "slots"        : {},
            "missing_slots": [],
        }
