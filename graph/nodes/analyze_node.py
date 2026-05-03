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
                    (e.g. "Is mental health covered?", "Does my plan cover dental?",
                    "어떤 플랜이 있어?", "cigna 플랜 종류 알려줘", "What plans does Cigna offer?")
                    NOTE: Asking WHAT plans/benefits EXIST = general_query (factual lookup)
- recommendation  : User asks the AI to CHOOSE or RECOMMEND a plan for them, or requests
                    legal/medical final judgments
                    (e.g. "어떤 보험이 좋아?", "이 보험 들어야 해?", "어떤 플랜 선택해야 해?",
                    "내 증상은 어떤 병이야?", "소송에서 이길 수 있어?")
                    NOTE: "어떤 플랜이 있어?" is NOT recommendation — it asks what exists, not what to pick
- clarify         : Not enough information to determine intent (ask user for more info)

Supported insurer codes: uhcg, cigna, tricare, msh_china

Respond ONLY with valid JSON in this exact format:
{
  "intents"      : ["primary_intent", "optional_second_intent"],
  "insurer"      : "insurer_code or empty string",
  "insurers"     : ["insurer1", "insurer2"],
  "slots"        : {
    "plan"       : "plan name or empty (단일 플랜 질문일 때)",
    "plans"      : ["plan1_name", "plan2_name"],
    "treatment"  : "treatment type or empty",
    "amount"     : 0,
    "currency"   : "USD or empty",
    "region"     : "region or empty",
    "date":"YYYY-MM-DD format or empty",
    "deductible": 0,
    "copay_rate": 0.2
  },
  "missing_slots": ["list of required but missing slot names"]
}

Rules:
- intents[0] is the PRIMARY intent used for routing
- For calculation: extract amount and currency from slots if mentioned
- For calculation: extract amount, currency, and date if mentioned.
- Convert Korean dates like "2025년 3월 15일" to "2025-03-15".
- If users asks "한화로 얼마야",  classify as caluction.
- IF the user mentions an insurer or plan, extract insurer and plan
- Extract date from user query if presendt
- For cross_compare: list all mentioned insurers in insurers[]
- For within_compare: extract ALL mentioned plan names into slots["plans"] as a list (e.g. ["Prime", "Select"])
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
        insurer      : FastAPI request에서 전달된 보험사 코드
        comparison_criteria : compare 요청 시 비교 기준 배열

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
    response_reset = _reset_response_fields()

    # FastAPI request에서 넘어온 insurer를 먼저 보존한다.
    request_insurer = _normalize_insurer(state.get("insurer", ""))

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
    # compare 요청은 message가 짧고 comparison_criteria에 핵심 내용이 들어갈 수 있으므로
    # 안전 검사 입력에 criteria를 문자열로 붙인다.
    safety_text = _build_safety_text(
        user_msg=user_msg,
        comparison_criteria=state.get("comparison_criteria", []),
    )

    blocked_msg = check_blocked(safety_text)

    if blocked_msg:
        return {
            **response_reset,
            "intent": Intent.BLOCKED,
            "intents": [Intent.BLOCKED],
            "insurer": request_insurer,
            "insurers": state.get("insurers", []),
            "slots": state.get("slots", {}),
            "missing_slots": [],
            "answer": blocked_msg,
        }

    # ── Step 2: 언어 감지 ──────────────────────────────────────
    # 짧은 메시지(15자 이하, 단어 2개 이하)는 이전 턴 언어를 유지
    # 예: "gold", "yes", "core 1" 같은 슬롯 답변은 언어 감지가 무의미
    prev_language = state.get("language", "")
    words = user_msg.strip().split()
    if prev_language and len(user_msg.strip()) <= 15 and len(words) <= 2:
        language = prev_language
    else:
        language = detect_language(user_msg)

    # ── Step 2-1: NHIS 대화 진행 중 short-circuit ─────────────
    # 아래 두 경우 LLM 재분류 없이 바로 nhis 라우팅
    #   ① nhis_step == "eligibility_check" + nhis_history 비어있지 않음
    #      이유: "E-7 비자예요", "회사 다녀요" 같은 맥락 답변을
    #            intent router가 단독으로 보면 clarify로 오분류하기 때문
    #   ② nhis_step == "info"
    #      이유: 자격 확인 완료 후 NHIS 정보 질문(보험료·급여 등)을
    #            intent router가 procedure/general_query로 오분류하기 때문
    nhis_step = state.get("nhis_step")
    if (nhis_step == "info"
            or (nhis_step == "eligibility_check" and state.get("nhis_history"))):
        return {
            "language": language,
            "intent": Intent.NHIS,
            "intents": [Intent.NHIS],
            "insurer": request_insurer or state.get("insurer", ""),
            "insurers": state.get("insurers", []),
            "slots": state.get("slots", {}),
            "missing_slots": [],
        }

    # ── Step 3: Intent Router (LLM) ────────────────────────────
    chat_history = state.get("chat_history", [])
    analysis = _run_intent_router(user_msg, chat_history[-6:])
    
    analysis_insurer = _normalize_insurer(analysis.get("insurer", ""))

    # request.insurer == "compare"이면 compare_node로 강제 라우팅
    if request_insurer == "compare":
        # LLM이 추출한 insurers를 우선 사용, 없을 때만 전체 4개 fallback
        extracted_insurers = [
            _normalize_insurer(ins)
            for ins in analysis.get("insurers", [])
            if _normalize_insurer(ins) and _normalize_insurer(ins) != "compare"
        ]
        final_insurers = extracted_insurers if extracted_insurers else ["uhcg", "cigna", "tricare", "msh_china"]
        return {
            "language": language,
            "intent": Intent.CROSS_COMPARE,
            "intents": [Intent.CROSS_COMPARE],
            "insurer": "compare",
            "insurers": final_insurers,
            "slots": analysis.get("slots", {}),
            "missing_slots": [],
        }

    final_insurer = request_insurer or analysis_insurer

    intents = analysis.get("intents", [Intent.CLARIFY])
    primary = intents[0] if intents else Intent.CLARIFY

    # ── Step 4: 추천/법적·의학적 판단 요청 차단 ───────────────
    if primary == Intent.RECOMMENDATION:
        return {
            **response_reset,
            "language": language,
            "intent": Intent.RECOMMENDATION,
            "intents": intents,
            "insurer": final_insurer,
            "insurers": analysis.get("insurers", []),
            "slots": analysis.get("slots", {}),
            "missing_slots": analysis.get("missing_slots", []),
            "answer": (
                "죄송합니다. 보험 상품 추천, 플랜 선택 권유, 법적·의학적 최종 판단은 제공하지 않습니다. "
                "보험 혜택·절차·청구 등 구체적인 질문을 해주세요.\n\n"
                "Sorry, we do not provide insurance product recommendations, plan selection advice, "
                "or legal/medical final judgments. "
                "Please ask about coverage details, procedures, or claims."
            ),
        }

    return {
        **response_reset,
        "language": language,
        "intent": primary,
        "intents": intents,
        "insurer": final_insurer,
        "insurers": analysis.get("insurers", []),
        "slots": analysis.get("slots", {}),
        "missing_slots": analysis.get("missing_slots", []),
        "chat_history": chat_history + [{"role": "user", "content": user_msg}],
    }


# ──────────────────────────────────────────────────────────────
# 내부 함수
# ──────────────────────────────────────────────────────────────

def _run_intent_router(user_msg: str, chat_history: list | None = None) -> dict:
    """
    LLM 을 호출해 의도·슬롯을 추출한다.

    LLM 응답이 올바른 JSON 이 아닐 경우 clarify 를 반환한다.
    """
    try:
        client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        messages = [{"role": "system", "content": _INTENT_SYSTEM_PROMPT}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_msg})
        response = client.chat.completions.create(
            model    = "gpt-4o-mini",
            messages = messages,
            max_tokens       = 300,
            temperature      = 0,
            response_format  = {"type": "json_object"},  # JSON 모드 강제
        )
        raw    = response.choices[0].message.content
        result = json.loads(raw)

        # intents 값 검증 — 허용된 Intent 상수만 통과
        valid_intents = {
            Intent.WITHIN_COMPARE,
            Intent.CROSS_COMPARE,
            Intent.CALCULATION,
            Intent.PROCEDURE,
            Intent.NHIS,
            Intent.CLAIM,
            Intent.GENERAL_QUERY,
            Intent.RECOMMENDATION,
            Intent.CLARIFY,
        }
        result["intents"] = [
            intent for intent in result.get("intents", [])
            if intent in valid_intents
        ] or [Intent.CLARIFY]

        return result

    except Exception:
        # LLM 오류 또는 JSON 파싱 실패 → clarify 로 처리
        return {
            "intents"      : [Intent.CLARIFY],
            "insurer"      : "",
            "insurers"     : [],
            "slots"        : {},
            "missing_slots": [],
        }

def _normalize_insurer(insurer: str) -> str:
    """
    보험사명을 시스템 내부 코드로 정규화한다.
    """
    insurer = (insurer or "").lower().strip()

    aliases = {
        "uhc": "uhcg",
        "uhcg": "uhcg",
        "unitedhealth": "uhcg",
        "cigna": "cigna",
        "tricare": "tricare",
        "msh": "msh_china",
        "msh china": "msh_china",
        "msh_china": "msh_china",
        "nhis": "nhis",
        "compare": "compare",
    }

    return aliases.get(insurer, insurer)


def _reset_response_fields() -> dict:
    """
    매 턴마다 이전 응답 전용 state를 초기화한다.

    LangGraph SqliteSaver는 thread_id 기준으로 state를 유지하므로,
    이전 턴의 claim_form, compare_table, sources 등이 다음 응답에 남지 않도록
    analyze 단계에서 공통 초기화한다.

    이후 실행되는 claim_node / compare_node / procedure_node 등이
    필요한 값을 다시 채워 넣는다.
    """
    return {
        "retrieved_docs": [],
        "answer": "",
        "sources": [],
        "claim_form": [],
        "compare_table": {},
        "related_questions": [],
    }


def _build_safety_text(user_msg: str, comparison_criteria: object) -> str:
    """
    safety 검사에 사용할 텍스트를 만든다.

    comparison_criteria가 list로 들어오는 경우가 있으므로
    user_msg += comparison_criteria 처럼 문자열+리스트 연산을 하지 않는다.
    """
    if isinstance(comparison_criteria, list):
        criteria_text = " ".join(str(item) for item in comparison_criteria)
    elif comparison_criteria:
        criteria_text = str(comparison_criteria)
    else:
        criteria_text = ""

    return " ".join(
        part for part in [user_msg, criteria_text]
        if part and str(part).strip()
    )