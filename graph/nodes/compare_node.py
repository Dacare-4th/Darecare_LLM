# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# graph/nodes/compare_node.py
# 역할 : 여러 보험사를 동시에 비교한다
#
# 파이프라인: ② 보험사 간 비교
# 진입 조건 : analyze_node 에서 intent == "cross_compare"
#             또는 request.insurer == "compare"
# 다음 노드  : END
#
# 흐름:
#   1. insurers 리스트의 각 컬렉션에서 병렬 RAG 검색
#   2. 결과 병합 + Re-ranking
#   3. 비교 프롬프트 조립
#   4. LLM 직접 호출 — JSON 강제 (_call_llm_json)
#   5. 응답 파싱
#   6. sources 추출 후 state 반환
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from graph.nodes.retrieve_node import query_multi_collections
from utils.comparison import rerank_by_relevance
from utils.schemas import InsuranceState

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────

_COMPARE_INSURERS = ["uhcg", "cigna", "tricare", "msh_china"]

_COMPARE_LABELS = {
    "uhcg": "UHCG",
    "cigna": "Cigna",
    "tricare": "Tricare",
    "msh_china": "MSH China",
    "nhis": "NHIS",
}

_CROSS_SYSTEM_PROMPT = """You are a health insurance comparison specialist.

Compare the insurance companies based ONLY on the provided documents.

You MUST respond with valid JSON only.
Do not wrap JSON in markdown.
Do not include explanations outside JSON.

Required JSON format:
{
  "answer": "brief comparison summary",
  "compare_table": {
    "header": ["**Comparison Criteria**", "**UHCG**", "**Cigna**", "**Tricare**", "**MSH China**"],
    "body": [
      ["**Annual Coverage Limit**", "value", "value", "value", "value"]
    ]
  },
  "related_questions": [
    "Question 1?",
    "Question 2?",
    "Question 3?"
  ]
}

Rules:
- Use ONLY the retrieved documents.
- If information is missing, write "Not available in documents".
- Do not invent benefit limits, deductibles, copay, or coverage rules.
- Keep each table cell short and table-friendly.
"""


# ──────────────────────────────────────────────────────────────
# 노드 함수
# ──────────────────────────────────────────────────────────────

def compare(state: InsuranceState) -> dict:
    user_msg = state["user_message"]
    language = state.get("language", "en")
    insurers = state.get("insurers", [])
    slots = state.get("slots", {})
    criteria = state.get("comparison_criteria", [])

    # ── 비교 기준 결정 ───────────────────────────────────────
    if not criteria:
        criteria = _default_criteria_from_message(user_msg)

    # ── 비교 대상 보험사 결정 ─────────────────────────────────
    insurers = _normalize_insurers(insurers, state.get("insurer", ""))

    # ── 보험사별 컬렉션 이름 매핑 ─────────────────────────────
    collection_map: dict[str, str] = {
        ins: ("nhis" if ins == "nhis" else f"{ins}_plans")
        for ins in insurers
    }

    # ── 병렬 멀티 컬렉션 RAG 검색 ─────────────────────────────
    search_query = _build_search_query(user_msg, criteria, slots)

    results_by_collection = query_multi_collections(
        collection_names=list(collection_map.values()),
        query=search_query,
        top_k_each=5,
    )

    col_to_ins = {v: k for k, v in collection_map.items()}

    docs_by_insurer: dict[str, list[str]] = {}
    all_retrieved: list[dict] = []

    for col_name, docs in results_by_collection.items():
        insurer_code = col_to_ins.get(col_name, col_name)
        insurer_label = _COMPARE_LABELS.get(insurer_code, insurer_code.upper())

        if not docs:
            docs_by_insurer[insurer_label] = []
        continue

    try:
        ranked = rerank_by_relevance(
            docs=[d.get("content", "") for d in docs],
            metadatas=[d.get("metadata", {}) for d in docs],
            top_k=5,
        )
    except Exception:
        ranked = docs[:5]

    docs_by_insurer[insurer_label] = [d.get("content", "") for d in ranked]
    all_retrieved.extend(ranked)

    # ── 비교 프롬프트 조립 ───────────────────────────────────
    comparison_prompt = _build_comparison_prompt(
        user_query=user_msg,
        language=language,
        criteria=criteria,
        insurers=insurers,
        docs_by_insurer=docs_by_insurer,
    )

    # ── LLM 직접 호출 ───────────────────────────────────────
    raw_response = _call_llm_json(comparison_prompt)

    parsed = _safe_parse_compare_response(
        raw_response=raw_response,
        criteria=criteria,
        insurers=insurers,
    )

    return {
        "retrieved_docs": all_retrieved,
        "answer": parsed["answer"],
        "compare_table": parsed["compare_table"],
        "related_questions": parsed["related_questions"],
        "sources": _build_sources(all_retrieved),
        "claim_form": [],
    }


# ──────────────────────────────────────────────────────────────
# 내부 함수
# ──────────────────────────────────────────────────────────────

def _normalize_insurers(insurers: list[str], fallback_insurer: str) -> list[str]:
    normalized = []

    for ins in insurers or []:
        code = _normalize_insurer(ins)
        if code and code != "compare":
            normalized.append(code)

    fallback = _normalize_insurer(fallback_insurer)

    if not normalized and fallback and fallback != "compare":
        normalized.append(fallback)

    if not normalized:
        normalized = list(_COMPARE_INSURERS)

    # 중복 제거
    result = []
    seen = set()

    for ins in normalized:
        if ins not in seen:
            result.append(ins)
            seen.add(ins)

    return result


def _normalize_insurer(insurer: str) -> str:
    insurer = (insurer or "").lower().strip()

    aliases = {
        "uhc": "uhcg",
        "uhcg": "uhcg",
        "cigna": "cigna",
        "tricare": "tricare",
        "msh": "msh_china",
        "msh china": "msh_china",
        "msh_china": "msh_china",
        "nhis": "nhis",
        "compare": "compare",
    }

    return aliases.get(insurer, insurer)


def _default_criteria_from_message(user_msg: str) -> list[str]:
    msg = (user_msg or "").lower()
    criteria = []

    if "annual" in msg or "limit" in msg:
        criteria.append("Annual Coverage Limit")

    if "cost" in msg or "deductible" in msg or "copay" in msg or "co-pay" in msg:
        criteria.append("Cost-Sharing Structure")

    if "outpatient" in msg:
        criteria.append("Outpatient Coverage")

    if "maternity" in msg or "prenatal" in msg:
        criteria.append("Maternity and Prenatal Coverage")

    if not criteria:
        criteria = [
            "Annual Coverage Limit",
            "Cost-Sharing Structure",
            "Outpatient Coverage",
            "Maternity and Prenatal Coverage",
        ]

    return criteria


def _build_search_query(user_msg: str, criteria: list[str], slots: dict) -> str:
    treatment = slots.get("treatment", "")
    plan = slots.get("plan", "")

    return " ".join(
        str(part).strip()
        for part in [
            user_msg,
            treatment,
            plan,
            " ".join(criteria),
            "benefits coverage limit deductible copay outpatient maternity claim process reimbursement",
        ]
        if part
    )


def _build_comparison_prompt(
    user_query: str,
    language: str,
    criteria: list[str],
    insurers: list[str],
    docs_by_insurer: dict[str, list[str]],
) -> str:
    header = ["**Comparison Criteria**"] + [
        f"**{_COMPARE_LABELS.get(ins, ins.upper())}**"
        for ins in insurers
    ]

    return f"""
User question:
{user_query}

Language:
{language}

Comparison criteria:
{json.dumps(criteria, ensure_ascii=False)}

Required table header:
{json.dumps(header, ensure_ascii=False)}

Retrieved documents grouped by insurer:
{json.dumps(docs_by_insurer, ensure_ascii=False)}

Create compare_table.body rows in exactly the same order as comparison criteria.
Each row must have exactly {len(header)} columns.
Each row must start with the bold comparison criterion.

Return JSON only.
"""


def _call_llm_json(prompt: str) -> str:
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _CROSS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=2000,
            temperature=0.1,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return json.dumps(
            {
                "answer": f"응답 생성 중 오류가 발생했습니다. ({type(e).__name__})",
                "compare_table": {
                    "header": [],
                    "body": [],
                },
                "related_questions": [],
            },
            ensure_ascii=False,
        )


def _safe_parse_compare_response(
    raw_response: str,
    criteria: list[str],
    insurers: list[str],
) -> dict:
    try:
        parsed = json.loads(raw_response)
    except Exception:
        parsed = {}

    answer = parsed.get("answer") or "Here is the comparison based on the retrieved insurance documents."
    compare_table = parsed.get("compare_table") or {}

    if not isinstance(compare_table, dict):
        compare_table = {}

    header = compare_table.get("header")
    body = compare_table.get("body")

    if not isinstance(header, list) or not isinstance(body, list):
        compare_table = _fallback_compare_table(criteria, insurers)

    related_questions = parsed.get("related_questions")

    if not isinstance(related_questions, list):
        related_questions = [
            "Which insurer has better outpatient coverage?",
            "Which plan has the lowest cost-sharing?",
            "Does this benefit require pre-authorization?",
        ]

    return {
        "answer": answer,
        "compare_table": compare_table,
        "related_questions": related_questions[:3],
    }


def _fallback_compare_table(criteria: list[str], insurers: list[str]) -> dict:
    header = ["**Comparison Criteria**"] + [
        f"**{_COMPARE_LABELS.get(ins, ins.upper())}**"
        for ins in insurers
    ]

    body = []

    for criterion in criteria:
        body.append(
            [f"**{criterion}**"]
            + ["Not available in documents" for _ in insurers]
        )

    return {
        "header": header,
        "body": body,
    }


def _build_sources(docs: list[Any]) -> list[dict]:
    sources = []
    seen = set()

    for doc in docs[:10]:
        metadata = doc.get("metadata", {}) if isinstance(doc, dict) else getattr(doc, "metadata", {})

        document_name = (
            metadata.get("source")
            or metadata.get("file_name")
            or metadata.get("url")
            or ""
        )
        page = metadata.get("page")
        section = (
            metadata.get("section")
            or metadata.get("topic")
            or metadata.get("doc_type")
            or ""
        )

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