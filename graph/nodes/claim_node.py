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
#   1. 보험사별 plans 컬렉션에서 RAG 검색
#      - procedure_docs   : 청구 절차 답변 생성용 문서
#      - claim_form_docs  : 청구서 파일 제공용 문서
#   2. 사용자 정보 슬롯 확인
#   3. LLM 으로 청구 절차 단계별 안내 생성
#   4. 청구서 양식 다운로드 링크 제공
#        - metadata.doc_type == "claim_form" 문서 확인
#        - metadata.source 값을 파일명으로 사용
#        - /download/{insurer}/{filename} 형태로 FastAPI 다운로드 URL 제공
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

from pathlib import Path
from typing import Any

from graph.nodes.generate_node import call_llm_parallel
from graph.nodes.retrieve_node import query_collection, query_multi_collections
from utils.schemas import InsuranceState

_ALL_INSURERS = ["uhcg", "cigna", "tricare", "msh_china"]

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

End with: "I can provide a claim form download link if a matching form is available." """


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
        retrieved_docs     : 검색된 청구 절차 문서 + 청구서 양식 문서
        answer             : 청구 절차 안내 + 필요 서류 목록
        sources            : 출처 목록
        claim_form         : 청구서 다운로드 정보 목록
        compare_table      : 비교 응답이 아니므로 빈 dict
        related_questions  : 추천 후속 질문
    """
    user_msg      = state["user_message"]
    language      = state.get("language", "en")
    insurer       = _normalize_insurer(state.get("insurer", ""))
    slots         = state.get("slots", {})
    nhis_step     = state.get("nhis_step", "")
    english_query = state.get("english_query", "") or user_msg
    chat_history  = state.get("chat_history", [])

    treatment = slots.get("treatment", "")
    plan = slots.get("plan", "")

    # ── Step 1: 청구 절차 문서 + 청구서 양식 chunk 검색 ───────
    procedure_docs: list[dict] = []
    claim_form_docs: list[dict] = []

    if insurer and insurer not in ("", "nhis"):
        collection_name = f"{insurer}_plans"

        # 답변 생성용 검색 쿼리 (english_query 기반으로 BM25 매칭 향상)
        procedure_query = _join_query_parts(
            insurer,
            plan,
            treatment,
            english_query,
            "claim procedure reimbursement submission required documents timeline",
        )

        # 파일 제공용 검색 쿼리
        # 특정 보험사명이나 파일명을 하드코딩하지 않고,
        # request.insurer + user_msg + slots + 공통 claim form 키워드를 함께 사용한다.
        claim_form_query = _join_query_parts(
            insurer,
            plan,
            treatment,
            user_msg,
            "claim form reimbursement out of pocket direct billing non-network hospital invoice payment receipt medical vision dental"
        )

        claim_form_docs = query_collection(
            collection_name=f"{insurer}_plans",
            query=claim_form_query,
            top_k=5,
            where={
                "doc_type": "claim_form"
            },
        )

        procedure_docs = query_collection(
            collection_name=collection_name,
            query=procedure_query,
            top_k=5,
        )

    elif insurer == "nhis" or nhis_step == "claim_link":
        procedure_docs = query_collection(
            collection_name="nhis",
            query="claim procedure 청구 절차 reimbursement required documents",
            top_k=5,
        )

        claim_form_docs = query_collection(
            collection_name="nhis",
            query="claim form reimbursement medical expenses required documents",
            top_k=5,
            where={"doc_type": "claim_form"},
        )

    # ── insurer 미확정 fallback: 전체 보험사 컬렉션 검색 ─────────
    if not procedure_docs and insurer not in ("nhis",):
        print("[claim_node] insurer 미확정 또는 검색 결과 없음 → 전체 컬렉션 검색")
        multi = query_multi_collections(
            collection_names = [f"{ins}_plans" for ins in _ALL_INSURERS],
            query            = _join_query_parts(
                english_query,
                treatment,
                "claim procedure reimbursement submission required documents",
            ),
            top_k_each = 3,
            hyde       = False,
        )
        procedure_docs = sorted(
            [doc for results in multi.values() for doc in results],
            key     = lambda d: d.get("score", 0),
            reverse = True,
        )[:5]

        if not claim_form_docs:
            form_query = _join_query_parts(
                english_query,
                treatment,
                "claim form reimbursement",
            )
            all_form_docs: list[dict] = []
            for ins in _ALL_INSURERS:
                all_form_docs.extend(
                    query_collection(
                        collection_name = f"{ins}_plans",
                        query           = form_query,
                        top_k           = 3,
                        where           = {"doc_type": "claim_form"},
                        hyde            = False,
                    )
                )
            claim_form_docs = sorted(
                all_form_docs,
                key     = lambda d: d.get("score", 0),
                reverse = True,
            )[:3]

    all_docs = procedure_docs + claim_form_docs

    # 디버깅용 로그
    print("🔥 claim_node 진입")
    print("insurer:", insurer)
    print("procedure_docs count:", len(procedure_docs))
    print("claim_form_docs count:", len(claim_form_docs))
    print("all_docs count:", len(all_docs))

    for i, doc in enumerate(all_docs):
        print(f"[claim_node DOC {i}] metadata:", _get_metadata(doc))

    # ── Step 2: NHIS 연계 청구 여부에 따라 컨텍스트 추가 ─────
    extra: dict = dict(slots)

    if nhis_step == "claim_link":
        extra["claim_type"] = "NHIS 적용 후 민간보험 추가 청구"
        extra["note"] = (
            "NHIS 적용 후 잔여 금액에 대한 민간보험 청구 절차를 안내해 주세요. "
            "NHIS 급여 확인서(요양급여확인서)가 추가 서류로 필요합니다."
        )

    # ── Step 3: 청구 절차 LLM 생성 + 연관질문 병렬 생성 ─────────
    procedure_answer, related_questions = call_llm_parallel(
        user_query     = user_msg,
        retrieved_docs = all_docs,
        language       = language,
        extra_context  = extra,
        system_prompt  = _CLAIM_SYSTEM_PROMPT,
    )

    # ── Step 4: 청구서 양식 다운로드 링크 제공 ────────────────
    claim_forms = _build_claim_forms(
        docs=claim_form_docs,
        fallback_insurer=insurer,
    )

    print("claim_forms:", claim_forms)

    return {
        "retrieved_docs"   : all_docs,
        "answer"           : procedure_answer,
        "sources"          : _build_sources(all_docs),
        "claim_form"       : claim_forms,
        "compare_table"    : {},
        "related_questions": related_questions,
        "chat_history"     : chat_history + [{"role": "assistant", "content": procedure_answer}],
    }


# ──────────────────────────────────────────────────────────────
# 내부 함수
# ──────────────────────────────────────────────────────────────

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


def _join_query_parts(*parts: Any) -> str:
    """
    빈 문자열/None 값을 제거하고 검색 쿼리를 구성한다.
    """
    return " ".join(
        str(part).strip()
        for part in parts
        if part is not None and str(part).strip()
    )


def _build_claim_forms(docs: list[Any], fallback_insurer: str = "") -> list[dict]:
    claim_forms: list[dict] = []
    seen: set[str] = set()

    for doc in docs:
        metadata = _get_metadata(doc)

        if metadata.get("doc_type") != "claim_form":
            continue

        form_insurer = _normalize_insurer(metadata.get("insurer", "")) or fallback_insurer
        file_name = metadata.get("file_name") or metadata.get("source")
        # file_path = metadata.get("file_path")

        if not form_insurer or not file_name:
            continue

        file_ext = Path(file_name).suffix.replace(".", "").lower()

        # FastAPI download endpoint 기준
        claim_form_path = f"/download/{form_insurer}/{file_name}"

        if claim_form_path in seen:
            continue

        seen.add(claim_form_path)

        claim_forms.append({
            "claim_form_path": claim_form_path,
            "claim_form_name": file_name,
            "claim_form_ext": file_ext,
            # 필요하면 프론트 디버깅용으로만 사용
            # "origin_file_path": file_path,
        })

    return claim_forms


def _build_sources(docs: list[Any]) -> list[dict]:
    """
    RAG 검색 결과에서 프론트 표시용 출처 목록을 생성한다.
    """
    sources: list[dict] = []
    seen: set[tuple] = set()

    for doc in docs:
        metadata = _get_metadata(doc)

        document_name = metadata.get("source", "")
        page = metadata.get("page") or None
        section = metadata.get("section") or metadata.get("doc_type", "")

        key = (document_name, page, section)

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            {
                "document_name": document_name,
                "page": page,
                "section": section,
            }
        )

    return sources


def _get_metadata(doc: Any) -> dict:
    """
    query_collection 반환값이 dict 또는 LangChain Document 형태일 수 있으므로
    둘 다 대응한다.
    """
    if isinstance(doc, dict):
        return doc.get("metadata", {}) or {}

    return getattr(doc, "metadata", {}) or {}