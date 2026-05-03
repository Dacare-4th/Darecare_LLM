# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# graph/nodes/clarify_node.py
# 역할 : 슬롯이 부족하거나 의도가 불명확할 때 사용자에게 재질문한다
#
# 파이프라인: (독립 노드, 특정 파이프라인 없음)
# 진입 조건 : analyze_node 에서
#             intent == "clarify" 또는 missing_slots 가 있을 때
# 다음 노드  : END (사용자 응답 후 다음 턴에 다시 analyze 부터 시작)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import os

from openai import OpenAI

from utils.schemas import InsuranceState

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────
_CLARIFY_SYSTEM_PROMPT = """You are a health insurance assistant.
The user's request is unclear or missing important information.
Ask ONE clear, concise question to get the missing information.
Be friendly and helpful. Do not ask multiple questions at once."""

# 슬롯별 재질문 템플릿 (언어: ko / en)
_SLOT_QUESTIONS: dict[str, dict[str, str]] = {
    "insurer": {
        "ko": "어떤 보험사의 보험을 이용하고 계신가요? (예: UHCG, Cigna, Tricare, MSH China)",
        "en": "Which insurance provider are you with? (e.g., UHCG, Cigna, Tricare, MSH China)",
    },
    "plan": {
        "ko": "어떤 플랜을 이용하고 계신가요? (예: Gold, Silver, Basic)",
        "en": "What plan are you on? (e.g., Gold, Silver, Basic)",
    },
    "treatment": {
        "ko": "어떤 치료 또는 의료 서비스에 대해 궁금하신가요?",
        "en": "What type of treatment or medical service are you asking about?",
    },
    "amount": {
        "ko": "의료비 금액이 얼마인가요? (통화도 함께 알려주세요. 예: 500 USD)",
        "en": "What is the medical bill amount? (Please include the currency, e.g., 500 USD)",
    },
    "currency": {
        "ko": "어떤 통화로 청구되었나요? (예: USD, EUR, JPY)",
        "en": "In which currency is the bill? (e.g., USD, EUR, JPY)",
    },
}


# ──────────────────────────────────────────────────────────────
# 노드 함수
# ──────────────────────────────────────────────────────────────

def clarify(state: InsuranceState) -> dict:
    """
    [재질문] 슬롯 부족 또는 의도 불명확 시 사용자에게 재질문한다.

    읽는 state 필드:
        user_message  : 사용자 원문 질의
        language      : 응답 언어 코드
        missing_slots : 누락된 필수 슬롯 목록
        intent        : 현재 감지된 의도 (불명확하면 "clarify")

    반환 dict (InsuranceState 업데이트):
        answer : 재질문 메시지 (사용자가 답변하면 다음 턴에 다시 analyze)
    """
    user_msg      = state["user_message"]
    language      = state.get("language", "en")
    missing_slots = state.get("missing_slots", [])
    chat_history  = state.get("chat_history", [])

    # ── 누락 슬롯이 명확한 경우 → 템플릿 재질문 ──────────────
    if missing_slots:
        first_missing = missing_slots[0]
        question      = _slot_question(first_missing, language)
        if question:
            return {
                "answer"      : question,
                "chat_history": chat_history + [{"role": "assistant", "content": question}],
            }

    # ── 의도 자체가 불명확한 경우 → LLM 재질문 ───────────────
    answer = _llm_clarify(user_msg, language)
    return {
        "answer"      : answer,
        "chat_history": chat_history + [{"role": "assistant", "content": answer}],
    }


# ──────────────────────────────────────────────────────────────
# 내부 함수
# ──────────────────────────────────────────────────────────────

def _slot_question(slot_name: str, language: str) -> str:
    """슬롯 이름에 해당하는 재질문 문자열을 반환한다."""
    template = _SLOT_QUESTIONS.get(slot_name, {})
    return template.get(language, template.get("en", ""))


def _llm_clarify(user_msg: str, language: str) -> str:
    """
    LLM 을 호출해 의도 불명확 시 재질문을 생성한다.
    ONE question only 제약을 프롬프트에서 강제한다.
    """
    lang_instructions = {
        "ko": "반드시 한국어로 질문하세요.",
        "en": "Ask in English.",
        "ja": "日本語で質問してください。",
        "zh": "请用中文提问。",
        "fr": "Posez la question en français.",
        "de": "Stellen Sie die Frage auf Deutsch.",
        "es": "Haga la pregunta en español.",
    }
    lang_inst = lang_instructions.get(language, "Ask in English.")

    try:
        client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model    = "gpt-4o-mini",
            messages = [
                {
                    "role"   : "system",
                    "content": f"{_CLARIFY_SYSTEM_PROMPT}\n{lang_inst}",
                },
                {"role": "user", "content": user_msg},
            ],
            max_tokens  = 150,
            temperature = 0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception:
        # LLM 오류 시 기본 재질문
        defaults = {
            "ko": "보험 관련해서 어떤 도움이 필요하신가요? 조금 더 자세히 알려주세요.",
            "en": "Could you provide more details about what you need help with regarding insurance?",
        }
        return defaults.get(language, defaults["en"])
