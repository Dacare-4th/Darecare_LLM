import re
import json
from pathlib import Path

import pdfplumber

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "tricare" / "claim"

PDF = [
    {
        "pdf_path": DATA_DIR / "dd2642.pdf",
        "doc_id": "tricare_dd2642_sep2024",
        "source": "dd2642.pdf",
        "form_name": "DD Form 2642",
        "form_title": "TRICARE DoD/CHAMPUS Medical Claim",
        "doc_type": "claim_form",
        "retrieval_scope": "procedure",
        "updated_at": "2025-09-23",
    },
    {
        "pdf_path": DATA_DIR / "dd2527.pdf",
        "doc_id": "tricare_dd2527_mar2020",
        "source": "dd2527.pdf",
        "form_name": "DD Form 2527",
        "form_title": "Statement of Personal Injury - Possible Third Party Liability",
        "doc_type": "injury_statement_form",
        "retrieval_scope": "procedure",
        "updated_at": "2026-01-13",
    },
]


COMMON_METADATA = {
    "insurer": "tricare",
    "is_latest": True,
    "language": "en",
    "plan": None,
    "audience": "foreign_residents_in_korea",
    "country_context": "Korea",
    "keep_for_rag": True,
}


def clean_common(text: str) -> str:
    patterns = [
        r"PREVIOUS EDITION IS OBSOLETE\.?",
        r"Page \d of \d",
        r"DD FORM \d+, [A-Z]{3} \d{4}",
        r"\(Updated \d+\)",
        r"OMB No\.\s*\d+-\d+",
        r"Exp\.:.*",
        r"CUI \(when filled in\)",
        r"DEFENSE HEALTH AGENCY",
    ]

    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned)

    return normalize_text(cleaned)


def clean_dd2642(text: str) -> str:
    cleaned = clean_common(text)

    patterns = [
        r"Prescribed by:.*",
        r"April 2015 & TRICARE Operations Manual.*",
        r"OMB approval expires.*",
        r"PATIENT'S REQUEST FOR MEDICAL PAYMENT\s*\d{8}",
        r"The public reporting burden[\s\S]*?currently valid OMB control number\.",
    ]

    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned)

    return normalize_text(cleaned)


def clean_dd2527(text: str) -> str:
    cleaned = clean_common(text)

    patterns = [
        r"The public reporting burden[\s\S]*?currently valid OMB control number\.",
        r"PLEASE DO NOT RETURN YOUR FORM TO THE ABOVE ORGANIZATION\.",
    ]

    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned)

    return normalize_text(cleaned)


def clean_text(text: str, source_file: str) -> str:
    if source_file == "dd2642.pdf":
        return clean_dd2642(text)

    if source_file == "dd2527.pdf":
        return clean_dd2527(text)

    return clean_common(text)


def normalize_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_pdf_pages(pdf_path: str, source_file: str) -> list[dict]:
    pages = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            raw_text = page.extract_text() or ""

            pages.append({
                "page": page.page_number,
                "raw_text": raw_text,
                "clean_text": clean_text(raw_text, source_file),
            })

    return pages


def extract_between(text: str, start: str, end: str | None = None) -> str:
    start_idx = text.find(start)

    if start_idx == -1:
        return ""

    if end is None:
        return text[start_idx:].strip()

    end_idx = text.find(end, start_idx + len(start))

    if end_idx == -1:
        return text[start_idx:].strip()

    return text[start_idx:end_idx].strip()


SECTION_RULES = {
    "dd2642.pdf": [
        {
            "section_id": "claim_overview",
            "section_title": "Claim Form Overview",
            "page": 1,
            "start": "TRICARE DoD/CHAMPUS MEDICAL CLAIM",
            "end": "PRIVACY ACT STATEMENT",
            "topic": ["claims", "overview"],
        },
        {
            "section_id": "important_claim_instructions",
            "section_title": "Important Claim Instructions",
            "page": 1,
            "start": "IMPORTANT - READ CAREFULLY",
            "end": "WHERE TO OBTAIN ADDITIONAL FORMS",
            "topic": ["claims", "itemized_bill", "deadline", "prescription", "overseas_claim"],
        },
        {
            "section_id": "how_to_fill_out_form",
            "section_title": "How to Fill Out Claim Form",
            "page": 2,
            "start": "HOW TO FILL OUT THE TRICARE/CHAMPUS FORM",
            "end": None,
            "topic": ["claims", "form_instruction", "other_health_insurance"],
        },
    ],
    "dd2527.pdf": [
        {
            "section_id": "injury_form_instructions",
            "section_title": "Personal Injury Form Instructions",
            "page": 1,
            "start": "INSTRUCTIONS",
            "end": None,
            "topic": ["claims", "injury", "third_party_liability", "deadline"],
        },
        {
            "section_id": "injury_general_information",
            "section_title": "General Information Fields",
            "page": 2,
            "start": "SECTION I - GENERAL INFORMATION",
            "end": "SECTION II - TYPE AND CAUSE OF INJURY",
            "topic": ["injury", "form_fields"],
        },
        {
            "section_id": "injury_type_and_cause",
            "section_title": "Type and Cause of Injury",
            "page": 2,
            "start": "SECTION II - TYPE AND CAUSE OF INJURY",
            "end": "SECTION III - MISCELLANEOUS",
            "topic": ["injury", "accident", "third_party_liability"],
        },
        {
            "section_id": "injury_miscellaneous",
            "section_title": "Miscellaneous Injury Information",
            "page": 2,
            "start": "SECTION III - MISCELLANEOUS",
            "end": None,
            "topic": ["injury", "insurance", "lawyer", "signature"],
        },
    ],
}


def preprocess_pdf(config: dict) -> list[dict]:
    if not config["pdf_path"].exists():
        print(f"[WARN] 파일 없음, 건너뜀: {config['source']}")
        return []
    pages = load_pdf_pages(config["pdf_path"], config["source"])
    rules = SECTION_RULES[config["source"]]

    results = []

    for rule in rules:
        page_text = pages[rule["page"] - 1]["clean_text"]

        content = extract_between(
            page_text,
            rule["start"],
            rule["end"],
        )

        if not content:
            print(f"섹션 추출 실패: {config['source_file']} / {rule['section_id']}")
            continue

        results.append({
            **COMMON_METADATA,
            **config,
            "section_id": rule["section_id"],
            "section_title": rule["section_title"],
            "page": rule["page"],
            "page_end": rule["page"],
            "topic": rule["topic"],
            "text": content,
        })

    return results

import sys
sys.path.append(str(BASE_DIR))
from utils.ingest_to_db import ingest

def main():
    all_sections = []

    for config in PDF:
        sections = preprocess_pdf(config)
        all_sections.extend(sections)

    output_path = BASE_DIR / "data" / "output" / "tricare" /"tricare_forms.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_sections, f, ensure_ascii=False, indent=2, default=str)

    print(f"저장 완료: {output_path}")
    print(f"총 섹션 수: {len(all_sections)}")

    for item in all_sections:
        print(item["source"], "|", item["section_id"], "|", len(item["text"]))
    chunks = [
        {
            "chunk_id": f"tricare_{s['section_id']}",
            "text": s["text"],
            "metadata": {k: v for k, v in s.items() if k != "text"},
        }
        for s in all_sections
    ]
    ingest(chunks)


if __name__ == "__main__":
    main()