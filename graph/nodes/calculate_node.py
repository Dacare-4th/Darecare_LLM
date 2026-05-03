# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# graph/nodes/calculate_node.py
# 역할 : 환율 조회 및 본인부담금 계산을 수행한다
#
# 파이프라인: ③ 계산 (환율 + 본인부담금 통합)
# 진입 조건 : analyze_node 에서 intent == "calculation"
# 다음 노드  : END
#
# 흐름:
#   1. slots 에서 금액·통화·공제액·부담률 추출
#   2. 실시간 환율 API 호출
#   3. 본인부담금 + KRW 환산 계산
#   4. 보험 컨텍스트 RAG 검색 (계산 결과와 함께 설명 제공용)
#   5. LLM 으로 계산 결과 + 설명 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import os

from openai import OpenAI

from graph.nodes.retrieve_node import query_collection
from utils.currency import calculate_copay, convert_to_krw, get_exchange_rate
from utils.schemas import InsuranceState

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────
_CALC_SYSTEM_PROMPT = """You are a health insurance cost calculation assistant.
Present the calculation results clearly and explain what each number means.
Use the provided reference documents for context about deductibles and copay rates.
Format numbers with commas for readability (e.g., 1,350,000 KRW).
If exchange rate data is real-time, mention the rate used."""


# ──────────────────────────────────────────────────────────────
# 노드 함수
# ──────────────────────────────────────────────────────────────

def calculate(state: InsuranceState) -> dict:
    """
    [파이프라인 ③] 환율 조회 및 본인부담금을 계산하고 설명한다.

    읽는 state 필드:
        user_message : 사용자 원문 질의
        language     : 응답 언어 코드
        insurer      : 보험사 코드 (컨텍스트 문서 검색용)
        slots        : {
            "amount"    : 금액 (숫자, 없으면 0),
            "currency"  : 통화 코드 (없으면 "USD"),
            "deductible": 공제액 (없으면 0),
            "copay_rate": 공동부담률 0.0~1.0 (없으면 0.2),
            "date" : 환율 기준일(예: "2025-03-15", 없으면 "")
        }

    반환 dict (InsuranceState 업데이트):
        calc_result    : 계산 결과 dict
        retrieved_docs : 보험 컨텍스트 문서 (설명용)
        answer         : 계산 결과 + 설명이 포함된 최종 응답
    """
    user_msg = state["user_message"]
    language = state.get("language", "en")
    insurer  = state.get("insurer", "")
    slots    = state.get("slots", {})

    # 슬롯에서 계산 파라미터 추출 
    amount     = float(slots.get("amount", 0))
    currency   = str(slots.get("currency", "USD")).upper()
    deductible = float(slots.get("deductible", 0))
    copay_rate = float(slots.get("copay_rate", 0.2))
    rate_date=str(slots.get("date", "") or "")
    plan=str(slots.get("plan", "") or "")

    # 플랜별 계산 정보가 필요한 경우 (조건 분기 추가)
    if amount > 0 and insurer and not plan:
        return {
            "calc_result": {},
            "retrieved_docs": [],
            "answer": ("보험사별/플랜별 공제액(deductible), 본인부담률(copay/consurance),보장 범위가 달라질 수 있습니다. 정확한 계산을 위해 가입한 플랜명을 알려주세요.")
        }
    #  환율 조회 및 계산 
    if amount > 0:
        # 금액이 명시된 경우 — 본인부담금 + KRW 환산 계산
        calc_result = calculate_copay(
            total_amount = amount,
            currency     = currency,
            deductible   = deductible,
            copay_rate   = copay_rate,
            rate_date= rate_date,
        )
    else:
        # 금액 미명시 — 환율만 조회
        rate = get_exchange_rate(currency, rate_date)
        calc_result = {
            "currency"     : currency,
            "exchange_rate": rate,
            "rate_date": rate_date or "latest",
            "note"         : "금액이 명시되지 않아 환율 정보만 제공합니다.",
        }

    # 보험 컨텍스트 RAG 검색 
    # 공제액·부담률 근거 문서를 함께 제공하기 위해 검색
    if insurer and insurer !="nhis":
        collection=f"{insurer}_plans"
        context_docs = query_collection(
            collection_name = collection,
            query           = f"deductible copay rate cost sharing {user_msg}",
            top_k           = 3,
        )
    else:
        context_docs=[]

    # ── Step 4: LLM 으로 계산 결과 설명 생성 ──────────────────
    calc_summary = _format_calc_result(calc_result)

    answer = _generate_calc_answer(
        user_query    = user_msg,
        calc_summary  = calc_summary,
        context_docs  = context_docs,
        language      = language,
    )

    return {
        "calc_result"  : calc_result,
        "retrieved_docs": context_docs,
        "answer"       : answer,
    }


# ──────────────────────────────────────────────────────────────
# 내부 함수
# ──────────────────────────────────────────────────────────────

def _format_calc_result(calc_result: dict) -> str:
    """계산 결과 dict 를 LLM 에 전달할 텍스트로 변환한다."""
    lines = ["[계산 결과]"]
    for key, val in calc_result.items():
        lines.append(f"  {key}: {val}")
    return "\n".join(lines)


def _generate_calc_answer(
    user_query  : str,
    calc_summary: str,
    context_docs: list[dict],
    language    : str,
) -> str:
    """계산 결과와 컨텍스트 문서를 바탕으로 LLM 설명 답변을 생성한다."""
    lang_instructions = {
        "ko": "반드시 한국어로 답변하세요.",
        "en": "Please respond in English.",
        "ja": "必ず日本語で回答してください。",
        "zh": "请用中文回答。",
        "fr": "Répondez en français.",
        "de": "Bitte antworten Sie auf Deutsch.",
        "es": "Por favor, responda en español.",
    }
    lang_inst = lang_instructions.get(language, "Please respond in English.")

    doc_blocks = "\n\n".join(
        f"[참조 문서 {i+1}]\n{d['content']}"
        for i, d in enumerate(context_docs[:3])
    )

    user_content = (
        f"{calc_summary}\n\n"
        f"참조 문서:\n{doc_blocks}\n\n"
        f"사용자 질문: {user_query}\n\n"
        f"{lang_inst}"
    )

    try:
        client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model       = "gpt-4o",
            messages    = [
                {"role": "system", "content": _CALC_SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens  = 1000,
            temperature = 0.2,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"계산 결과:\n{calc_summary}\n\n(응답 생성 오류: {type(e).__name__})"
