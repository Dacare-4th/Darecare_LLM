# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# utils/schemas.py
# 역할 : LangGraph 전역 상태 및 노드 간 데이터 계약 정의
#
# ※ 모든 노드는 InsuranceState를 읽고 dict를 반환한다.
#   반환된 dict의 키는 InsuranceState 필드와 일치해야 한다.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from typing import TypedDict, Optional


# ──────────────────────────────────────────────────────────────
# Intent 상수 — analyze_node 가 반환하는 값 / builder 라우팅 기준
# ──────────────────────────────────────────────────────────────
class Intent:
    WITHIN_COMPARE = "within_compare"   # ① 보험 내 플랜 비교
    CROSS_COMPARE  = "cross_compare"    # ② 보험사 간 비교
    CALCULATION    = "calculation"      # ③ 환율·본인부담금 계산
    PROCEDURE      = "procedure"        # ④ 일반 절차 안내
    NHIS           = "nhis"             # ⑤ NHIS 상담
    CLAIM          = "claim"            # ⑥ 청구 절차 + 양식
    CLARIFY        = "clarify"          # 슬롯 부족 → 재질문
    BLOCKED        = "blocked"          # 안전 필터 차단


# ──────────────────────────────────────────────────────────────
# InsuranceState — 모든 노드가 공유하는 LangGraph 전역 상태
# ──────────────────────────────────────────────────────────────
class InsuranceState(TypedDict):
    # ── 입력 (chat.py 에서 초기화) ────────────────────────────
    session_id   : str          # 대화 세션 식별자 (멀티턴 체크포인트용)
    user_message : str          # 사용자 원문 질의

    # ── 언어 감지 (analyze_node 에서 설정) ────────────────────
    language     : str          # 감지된 언어 코드
                                # "ko" | "en" | "ja" | "zh" | "fr" | "de" | "es"

    # ── 의도 분류 (analyze_node 에서 설정) ────────────────────
    intent       : str          # 주 의도 → builder 라우팅 기준 (Intent 상수 참조)
    intents      : list         # 복합 의도 리스트 (예: ["nhis", "calculation"])
                                # intent 는 intents[0] 과 동일

    # ── 슬롯 (analyze_node 에서 설정) ─────────────────────────
    insurer      : str          # 단일 보험사 코드 (파이프라인 ① 용)
                                # "uhcg" | "cigna" | "tricare" | "msh_china" | "nhis" | ""
    insurers     : list         # 복수 보험사 코드 리스트 (파이프라인 ② 용)
    slots        : dict         # 추출된 슬롯
                                # {"plan": "Gold", "treatment": "입원", "amount": 5000, ...}
    missing_slots: list         # 아직 확인되지 않은 필수 슬롯 목록
                                # 비어있으면 모든 슬롯 충족 → 바로 처리 가능

    # ── RAG 검색 결과 (retrieve_node / 각 파이프라인 내부에서 설정) ─
    retrieved_docs : list       # [{"content": str, "metadata": dict}, ...]

    # ── 계산 결과 (calculate_node 에서 설정) ──────────────────
    calc_result    : dict       # {"exchange_rate": float, "currency": str,
                                #  "amount_krw": float, "copay_krw": float}

    # ── NHIS 멀티턴 상태 (nhis_node 에서 관리) ────────────────
    nhis_step      : str        # NHIS 대화 단계
                                # "eligibility_check" | "info" | "claim_link" | "done"
    nhis_eligible  : Optional[bool]  # 자격 확인 결과 (None = 미확인)

    # ── 최종 응답 (각 파이프라인 노드 또는 generate_node 에서 설정) ─
    answer         : str        # 클라이언트에 전달할 최종 응답 텍스트
    
    # 클라이언트에 전달할 최종 응답
    sources: list
    claim_form: list
    compare_table: dict
    related_questions: list
    comparison_criteria: list



# ──────────────────────────────────────────────────────────────
# InsuranceState 초기값 헬퍼
# ──────────────────────────────────────────────────────────────
def initial_state(session_id: str, user_message: str) -> InsuranceState:
    """
    chat.py 에서 graph.invoke() 호출 시 사용하는 초기 상태 생성 헬퍼.
    모든 선택 필드를 안전한 기본값으로 채운다.
    """
    return InsuranceState(
        session_id    = session_id,
        user_message  = user_message,
        language      = "",
        intent        = "",
        intents       = [],
        insurer       = "",
        insurers      = [],
        slots         = {},
        missing_slots = [],
        retrieved_docs= [],
        calc_result   = {},
        nhis_step     = "eligibility_check",
        nhis_eligible = None,
        answer        = "",
        
        # 클라이언트에 전달할 최종 응답
        sources=[],
        claim_form=[],
        compare_table={},
        related_questions=[],
        comparison_criteria=[],
    )


# ──────────────────────────────────────────────────────────────
# DocumentMetadata — ChromaDB 메타데이터 표준
# ※ 모든 ingest.py 가 반드시 따라야 하는 스키마
# ──────────────────────────────────────────────────────────────
class DocumentMetadata(TypedDict):
    insurer      : str   # "uhcg" | "cigna" | "tricare" | "msh_china" | "nhis"
    source_type  : str   # "pdf" | "pdf_table" | "web"
    file_name    : str   # PDF 파일명 또는 웹 토픽명
    page         : int   # PDF 페이지 번호 (웹은 0)
    year         : str   # 문서 연도 (예: "2024")
    plan         : str   # 플랜명 (없으면 "")
    language     : str   # "en" | "ko"
    url          : str   # 웹 소스 URL (PDF는 "")
    topic        : str   # 문서 토픽 (예: "eligibility", "contribution", "annual_report")
    table_json   : str   # 표 JSON 직렬화 문자열 (source_type=="pdf_table" 일 때만 사용, 나머지 "")