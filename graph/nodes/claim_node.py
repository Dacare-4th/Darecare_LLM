# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# graph/nodes/claim_node.py
# 역할 : 보험 청구 절차를 안내하고 청구서 양식을 제공한다
#
# 파이프라인: ⑥ 청구 절차 + 양식
# 진입 조건 : analyze_node 에서 intent == "claim"
#             또는 nhis_node 에서 민간보험 연계 청구 감지 시
# 다음 노드  : END
#
# 흐름:
#   1. claim_procedures 컬렉션에서 RAG 검색
#   2. 사용자 정보 슬롯 확인
#   3. LLM 으로 청구 절차 단계별 안내 생성
#   4. 청구서 양식 다운로드 링크 제공
#        (실제 파일 생성은 별도 파일 생성 서비스에서 처리)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

from graph.nodes.generate_node import call_llm_with_docs, _build_sources, _call_llm_for_related_questions
from graph.nodes.retrieve_node import query_collection
from utils.schemas import InsuranceState

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────
_CLAIM_SYSTEM_PROMPT = """You are a health insurance claim specialist.
Guide the user through the insurance claim process step by step.

Your response must include:
1. Step-by-step claim procedure (numbered list)
2. Required documents checklist:
   - Medical bills / receipts (의료비 영수증)
   - Doctor's diagnosis / treatment record (진단서 / 진료기록부)
   - Insurance claim form (보험청구서)
   - ID / Passport copy
   - Bank account information (for reimbursement)
   - Any insurer-specific additional documents
3. Submission method (online / mail / in-person)
4. Expected processing timeline
5. Contact information for questions

End with: "I can provide a pre-filled claim form template based on your information." """

# 보험사별 청구서 양식 경로 (실제 파일은 별도 관리)
# TODO: 실제 양식 파일 경로로 교체
_CLAIM_FORM_PATHS: dict[str, str] = {
    "uhcg"     : "./data/forms/uhcg_claim_form.pdf",
    "cigna"    : "./data/forms/cigna_claim_form.pdf",
    "tricare"  : "./data/forms/tricare_claim_form.pdf",
    "msh_china": "./data/forms/msh_china_claim_form.pdf",
    "nhis"     : "./data/forms/nhis_claim_form.pdf",
    "default"  : "./data/forms/generic_claim_form.pdf",
}


# ──────────────────────────────────────────────────────────────
# 노드 함수
# ──────────────────────────────────────────────────────────────

def claim(state: InsuranceState) -> dict:
    """
    [파이프라인 ⑥] 보험 청구 절차 안내 및 청구서 양식을 제공한다.

    읽는 state 필드:
        user_message : 사용자 원문 질의
        language     : 응답 언어 코드
        insurer      : 보험사 코드 (보험사 전용 양식 선택)
        slots        : 사용자 정보 슬롯
                       {"plan": str, "treatment": str, "amount": float, "currency": str}
        nhis_step    : "claim_link" 이면 NHIS 연계 청구 플로우

    반환 dict (InsuranceState 업데이트):
        retrieved_docs    : 검색된 청구 절차 문서
        answer            : 청구 절차 안내 + 필요 서류 목록 + 양식 안내
        sources           : 참조 문서 출처 리스트
        related_questions : 연관 질문 리스트
    """
    user_msg  = state["user_message"]
    language  = state.get("language", "en")
    insurer   = state.get("insurer", "")
    slots     = state.get("slots", {})
    nhis_step = state.get("nhis_step", "")

    # ── Step 1: 청구 절차 문서 RAG 검색 ──────────────────────
    claim_docs = query_collection(
        collection_name = "claim_procedures",
        query           = user_msg,
        top_k           = 5,
    )

    # 보험사 전용 문서 추가 검색
    insurer_docs: list[dict] = []
    if insurer and insurer not in ("", "nhis"):
        insurer_docs = query_collection(
            collection_name = f"{insurer}_plans",
            query           = f"claim procedure required documents {user_msg}",
            top_k           = 3,
        )
    elif insurer == "nhis" or nhis_step == "claim_link":
        insurer_docs = query_collection(
            collection_name = "nhis",
            query           = "claim procedure 청구 절차",
            top_k           = 3,
        )

    all_docs = insurer_docs + claim_docs

    # ── Step 2: NHIS 연계 청구 여부에 따라 컨텍스트 추가 ─────
    extra: dict = dict(slots)
    if nhis_step == "claim_link":
        extra["claim_type"] = "NHIS 적용 후 민간보험 추가 청구"
        extra["note"] = (
            "NHIS 적용 후 잔여 금액에 대한 민간보험 청구 절차를 안내해 주세요. "
            "NHIS 급여 확인서(요양급여확인서)가 추가 서류로 필요합니다."
        )

    # ── Step 3: 청구 절차 LLM 생성 ───────────────────────────
    procedure_answer = call_llm_with_docs(
        user_query     = user_msg,
        retrieved_docs = all_docs,
        language       = language,
        extra_context  = extra,
        system_prompt  = _CLAIM_SYSTEM_PROMPT,
    )

    # ── Step 4: 청구서 양식 안내 추가 ────────────────────────
    form_info = _get_form_info(insurer, language)
    final_answer = f"{procedure_answer}\n\n{form_info}"

    sources           = _build_sources(all_docs)
    related_questions = _call_llm_for_related_questions(
        user_query = user_msg,
        answer     = final_answer,
        language   = language,
    )

    return {
        "retrieved_docs"   : all_docs,
        "answer"           : final_answer,
        "sources"          : sources,
        "related_questions": related_questions,
    }


# ──────────────────────────────────────────────────────────────
# 내부 함수
# ──────────────────────────────────────────────────────────────

def _get_form_info(insurer: str, language: str) -> str:
    """
    보험사에 맞는 청구서 양식 안내 메시지를 반환한다.

    TODO: 실제 파일 생성 서비스(PDF/DOCX 생성 API) 연동 시
          이 함수에서 파일 생성 후 다운로드 URL 을 반환하도록 수정.
    """
    form_path = _CLAIM_FORM_PATHS.get(insurer, _CLAIM_FORM_PATHS["default"])

    form_messages = {
        "ko": (
            f"\n📄 **청구서 양식 안내**\n"
            f"보험사: {insurer.upper() if insurer else '일반'}\n"
            f"양식 경로: `{form_path}`\n\n"
            f"청구서 자동 작성이 필요하시면 다음 정보를 알려주세요:\n"
            f"- 성명, 생년월일\n"
            f"- 치료 날짜 및 병원명\n"
            f"- 진단명 및 치료 내용\n"
            f"- 청구 금액 및 통화"
        ),
        "en": (
            f"\n📄 **Claim Form**\n"
            f"Insurer: {insurer.upper() if insurer else 'General'}\n"
            f"Form path: `{form_path}`\n\n"
            f"For automatic form filling, please provide:\n"
            f"- Full name, date of birth\n"
            f"- Treatment date and hospital name\n"
            f"- Diagnosis and treatment details\n"
            f"- Claim amount and currency"
        ),
    }
    return form_messages.get(language, form_messages["en"])
