# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# graph/nodes/generate_node.py
# 역할 : RAG 검색 결과를 바탕으로 최종 답변을 생성한다 (공통 생성 노드)
#
# 파이프라인: ④ 절차 안내에서 retrieve_node 다음에 호출
#             (① ② ③ ⑤ ⑥ 는 각 노드가 내부적으로 직접 생성)
#
# 진입 조건 : retrieve_node 다음 (intent == "procedure")
# 다음 노드  : END
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import json
import os

from openai import OpenAI

from utils.schemas import InsuranceState

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────
_GENERATE_SYSTEM_PROMPT = """You are a helpful health insurance assistant.
Answer the user's question based ONLY on the provided reference documents.
If the documents do not contain enough information, say so clearly.
Always cite which document your answer is based on.
Keep your answer concise and structured."""

_RELATED_QUESTIONS_SYSTEM_PROMPT = """You are a helpful health insurance assistant.
Based on the user's question and the answer provided, generate exactly 3 follow-up questions
the user might want to ask next. Return ONLY a JSON array of 3 strings, nothing else.
Example: ["Question 1?", "Question 2?", "Question 3?"]"""

# 답변 생성용 언어 지시문 (글자 수 제한 포함)
_LANGUAGE_INSTRUCTION = {
    "ko": "반드시 한국어로 답변하세요. 답변은 공백 포함 최대 1500자 이내로 작성하세요.",
    "en": "Please respond in English. Keep your response within 1500 characters including spaces.",
    "ja": "必ず日本語で回答してください。回答は空白を含め1500文字以内にしてください。",
    "zh": "请用中文回答。回答请控制在1500个字符以内（含空格）。",
    "fr": "Répondez en français. Limitez votre réponse à 1500 caractères espaces compris.",
    "de": "Bitte antworten Sie auf Deutsch. Halten Sie Ihre Antwort auf maximal 1500 Zeichen inkl. Leerzeichen.",
    "es": "Por favor, responda en español. Limite su respuesta a 1500 caracteres incluyendo espacios.",
}

# 연관 질문 생성용 언어 지시문 (질문 생성 맥락에 맞게 별도 관리)
_RELATED_QUESTIONS_LANGUAGE_INSTRUCTION = {
    "ko": "반드시 한국어로 질문을 작성하세요.",
    "en": "Write the questions in English.",
    "ja": "必ず日本語で質問を作成してください。",
    "zh": "请用中文撰写问题。",
    "fr": "Rédigez les questions en français.",
    "de": "Schreiben Sie die Fragen auf Deutsch.",
    "es": "Escriba las preguntas en español.",
}


# ──────────────────────────────────────────────────────────────
# 노드 함수
# 직접 import 해서 쓰는 파일 없음 -> Dead Node, Just 예시용
# ──────────────────────────────────────────────────────────────

def generate(state: InsuranceState) -> dict:
    """
    [공통 생성] retrieved_docs 를 기반으로 LLM 답변을 생성한다.

    읽는 state 필드:
        user_message   : 사용자 원문 질의
        language       : 응답 언어 코드
        retrieved_docs : retrieve_node 가 반환한 문서 리스트
        slots          : 추출된 슬롯 (추가 컨텍스트로 활용)

    반환 dict (InsuranceState 업데이트):
        answer             : 생성된 최종 응답 텍스트
        sources            : 참조 문서 출처 리스트
        related_questions  : 연관 질문 리스트
    """
    retrieved_docs = state.get("retrieved_docs", [])

    answer = call_llm_with_docs(
        user_query     = state["user_message"],
        retrieved_docs = retrieved_docs,
        language       = state.get("language", "en"),
        extra_context  = state.get("slots", {}),
    )

    sources = _build_sources(retrieved_docs)

    related_questions = _call_llm_for_related_questions(
        user_query = state["user_message"],
        answer     = answer,
        language   = state.get("language", "en"),
    )

    return {
        "answer"            : answer,
        "sources"           : sources,
        "related_questions" : related_questions,
    }


# ──────────────────────────────────────────────────────────────
# 공개 헬퍼 — 다른 노드에서도 직접 호출 가능
# ──────────────────────────────────────────────────────────────

def call_llm_with_docs(
    user_query    : str,
    retrieved_docs: list[dict],
    language      : str,
    extra_context : dict | None = None,
    system_prompt : str | None  = None,
) -> str:
    """
    검색된 문서를 컨텍스트로 LLM 을 호출해 답변을 생성한다.

    모든 파이프라인 노드가 최종 답변 생성 시 이 함수를 사용한다.

    Args:
        user_query     : 사용자 원문 질의
        retrieved_docs : [{"content": str, "metadata": dict}, ...] 문서 리스트
        language       : 응답 언어 코드 (예: "ko", "en")
        extra_context  : 추가 컨텍스트 dict (슬롯 정보 등)
        system_prompt  : 커스텀 시스템 프롬프트 (None 이면 기본값 사용)

    Returns:
        LLM 생성 답변 텍스트. 오류 시 오류 안내 메시지 반환.
    """
    # ── 언어 지시문 ────────────────────────────────────────────
    lang_inst = _LANGUAGE_INSTRUCTION.get(language, "Please respond in English.")

    # ── 시스템 프롬프트 조립 ───────────────────────────────────
    sys_prompt = (system_prompt or _GENERATE_SYSTEM_PROMPT) + f"\n\n{lang_inst}"

    # ── 참조 문서 블록 조립 ─────────────────────────────────────
    if retrieved_docs:
        doc_blocks = []
        for i, doc in enumerate(retrieved_docs[:7], start=1):  # 최대 7개 사용
            meta   = doc.get("metadata", {})
            source = _format_source(meta)
            doc_blocks.append(f"[문서 {i} | {source}]\n{doc['content']}")
        context_str = "\n\n".join(doc_blocks)
    else:
        context_str = "(검색된 문서 없음)"

    # ── 추가 컨텍스트 (슬롯 정보) ─────────────────────────────
    context_note = ""
    if extra_context:
        context_note = f"\n\n추가 정보: {extra_context}"

    # ── LLM 호출 ───────────────────────────────────────────────
    user_content = (
        f"참조 문서:\n{context_str}"
        f"{context_note}\n\n"
        f"질문: {user_query}"
    )

    try:
        client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model       = "gpt-4o",
            messages    = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_content},
            ],
            max_tokens  = 1500,
            temperature = 0.1,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return (
            f"죄송합니다. 응답 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.\n"
            f"Sorry, an error occurred while generating a response. Please try again.\n"
            f"(Error: {type(e).__name__})"
        )


# ──────────────────────────────────────────────────────────────
# 내부 함수
# ──────────────────────────────────────────────────────────────

def _format_source(metadata: dict) -> str:
    """메타데이터를 간결한 출처 표기 문자열로 변환한다."""
    source_type = metadata.get("source_type", "")
    if source_type == "web":
        return f"Web | {metadata.get('topic', '')} | {metadata.get('url', '')}"
    if source_type in ("pdf", "pdf_table"):
        page = metadata.get("page", "")
        return f"PDF | {metadata.get('file_name', '')} | p.{page}"
    return metadata.get("file_name", "unknown")


def _build_sources(retrieved_docs: list[dict]) -> list[dict]:
    """
    retrieved_docs 메타데이터를 sources 스키마로 변환한다.

    document_name 은 항상 포함되며, source_type 에 따라 추가 필드를 선택적으로 포함한다.
        - pdf / pdf_table : page, section (topic)
        - web             : url, topic
    """
    sources = []
    for doc in retrieved_docs[:7]:
        meta        = doc.get("metadata", {})
        source_type = meta.get("source_type", "")
        source: dict = {"document_name": meta.get("file_name", "unknown")}

        if source_type in ("pdf", "pdf_table"):
            if meta.get("page"):
                source["page"] = meta["page"]
            if meta.get("topic"):
                source["section"] = meta["topic"]
        elif source_type == "web":
            if meta.get("url"):
                source["url"] = meta["url"]
            if meta.get("topic"):
                source["topic"] = meta["topic"]

        sources.append(source)
    return sources


def _call_llm_for_related_questions(
    user_query: str,
    answer    : str,
    language  : str,
) -> list[str]:
    """
    answer 를 바탕으로 연관 질문 3개를 생성한다. (별도 LLM 호출)

    Returns:
        연관 질문 문자열 리스트. 오류 시 빈 리스트 반환.
    """
    lang_inst  = _RELATED_QUESTIONS_LANGUAGE_INSTRUCTION.get(language, "Write the questions in English.")
    sys_prompt = _RELATED_QUESTIONS_SYSTEM_PROMPT + f"\n\n{lang_inst}"

    user_content = (
        f"User question: {user_query}\n\n"
        f"Answer: {answer}\n\n"
        f'Return ONLY a JSON array like: ["Q1?", "Q2?", "Q3?"]'
    )

    try:
        client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model       = "gpt-4o",
            messages    = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_content},
            ],
            max_tokens  = 300,
            temperature = 0.7,
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)

    except Exception:
        return []
