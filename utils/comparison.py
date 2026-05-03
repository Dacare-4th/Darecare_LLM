# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# utils/comparison.py
# 역할 : 보험사/플랜 비교표 데이터 조립 헬퍼
#
# 사용처:
#   - within_node.py  (① 보험 내 플랜 비교)
#   - compare_node.py (② 보험사 간 비교)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import json

# 공개 API


def parse_compare_table(raw_response: str) -> tuple[dict, str, list]:
    """
    LLM JSON 응답을 (compare_table, answer, related_questions) 로 파싱한다.
    파싱 실패 시 빈 구조체를 반환한다.
    """
    try:
        data = json.loads(raw_response) if isinstance(raw_response, str) else raw_response
    except (json.JSONDecodeError, TypeError):
        data = {}

    compare_table = data.get("compare_table", {"header": [], "body": []})
    if not isinstance(compare_table, dict):
        compare_table = {"header": [], "body": []}

    answer = data.get("answer", raw_response if isinstance(raw_response, str) else "")
    related_questions = data.get("related_questions", [])
    if not isinstance(related_questions, list):
        related_questions = []

    return compare_table, answer, related_questions


def build_comparison_prompt(
    docs_by_subject: dict[str, list[str]],
    user_query: str,
    language: str,
) -> str:
    """
    RAG 검색 결과를 비교표 생성용 LLM 프롬프트로 조립한다.

    Args:
        docs_by_subject: 비교 대상별 검색 문서 리스트
                         예) {"UHCG Gold": ["doc1", "doc2"], "Cigna": ["doc3"]}
        user_query     : 사용자 원문 질의 (컨텍스트 유지)
        language       : 응답 언어 코드 (예: "ko", "en")

    Returns:
        LLM user 메시지로 전달할 비교 프롬프트 문자열
    """
    lang_instruction = _language_instruction(language)

    # 비교 대상별 문서 블록 조립
    context_blocks = []
    for subject, docs in docs_by_subject.items():
        combined = "\n---\n".join(docs[:5])  # 대상당 최대 5개 문서 사용
        context_blocks.append(f"[{subject}]\n{combined}")

    context_str = "\n\n".join(context_blocks)

    # 헤더를 docs_by_subject의 실제 키(플랜명/보험사명)로 동적 생성
    # 하드코딩("대상 1", "대상 2")하면 LLM이 그대로 따라써서 실제 이름이 안 나옴
    subjects = list(docs_by_subject.keys())
    header_example = ["비교 항목"] + subjects
    body_placeholder = ["문서 기반 내용"] * len(subjects)
    body_example = [
        ["보장 범위"] + body_placeholder,
        ["보장 한도"] + body_placeholder,
        ["제외 항목"] + body_placeholder,
        ["본인부담금 구조"] + body_placeholder,
        ["특이 사항"] + body_placeholder,
    ]

    import json as _json
    format_example = _json.dumps({
        "compare_table": {"header": header_example, "body": body_example},
        "answer": "비교 결과를 자연어로 요약하세요.",
        "related_questions": ["관련 후속 질문 1", "관련 후속 질문 2", "관련 후속 질문 3"],
    }, ensure_ascii=False, indent=2)

    prompt = f"""{lang_instruction}

다음은 비교 대상별로 검색된 보험 문서입니다:

{context_str}

사용자 질의: {user_query}

위 문서를 바탕으로 비교표를 작성해 주세요.
반드시 아래 JSON 형식으로만 응답하세요.
마크다운, 설명 문장, 코드블록은 절대 포함하지 마세요.
{format_example}
규칙:
- header의 열 이름({', '.join(subjects)})을 그대로 유지하세요.
- 각 항목에 대해 문서에 명시된 내용만 사용하고, 불확실한 경우 "정보 없음"으로 표시하세요.
- answer도 반드시 {language} 언어로 작성하세요.
- related_questions는 반드시 3개로 작성하세요."""

    return prompt


def merge_docs_for_comparison(
    results_by_subject: dict[str, dict],
) -> dict[str, list[str]]:
    """
    ChromaDB 멀티 컬렉션 검색 결과를 비교표 빌더 형식으로 변환한다.

    Args:
        results_by_subject: {subject: chromadb_query_result}
                            chromadb_query_result 형식:
                            {"documents": [[doc1, doc2, ...]], "metadatas": [[meta1, ...]]}

    Returns:
        {subject: [doc_text_list]}
    """
    merged: dict[str, list[str]] = {}
    for subject, result in results_by_subject.items():
        docs = result.get("documents", [[]])[0]  # 첫 번째 쿼리 결과만 사용
        merged[subject] = [d for d in docs if d and d.strip()]
    return merged


def rerank_by_relevance(
    docs: list[str],
    metadatas: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    ChromaDB 검색 결과를 메타데이터 기준으로 정렬한다.

    현재 전략: source_type=="pdf_table" 을 우선 배치
    (표 데이터가 구조화돼 있어 비교에 더 유리)

    Args:
        docs     : 문서 텍스트 리스트
        metadatas: 각 문서의 메타데이터 리스트
        top_k    : 반환할 최대 문서 수

    Returns:
        [{"content": str, "metadata": dict}, ...] (정렬 후 top_k 개)
    """
    paired = [
        {"content": doc, "metadata": meta}
        for doc, meta in zip(docs, metadatas)
        if doc and doc.strip()
    ]

    # pdf_table 우선 정렬 → 나머지는 원래 순서(ChromaDB cosine 점수 순) 유지
    tables  = [p for p in paired if p["metadata"].get("source_type") == "pdf_table"]
    others  = [p for p in paired if p["metadata"].get("source_type") != "pdf_table"]

    return (tables + others)[:top_k]


# ──────────────────────────────────────────────────────────────
# 내부 함수
# ──────────────────────────────────────────────────────────────

def _language_instruction(language: str) -> str:
    """언어 코드 → LLM 응답 언어 지시문"""
    instructions = {
        "ko": "반드시 한국어로 답변하세요.",
        "en": "Please respond in English.",
        "ja": "必ず日本語で回答してください。",
        "zh": "请用中文回答。",
        "fr": "Répondez en français.",
        "de": "Bitte antworten Sie auf Deutsch.",
        "es": "Por favor, responda en español.",
    }
    return instructions.get(language, "Please respond in English.")
