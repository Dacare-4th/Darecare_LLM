"""
preprocess_msh.py — MSH PDF 전처리 → JSON 청크 저장

실행:
    python preprocess_msh.py
    python preprocess_msh.py --debug-sidebar   # 사이드바 x0 좌표 확인용
    python preprocess_msh.py --output-dir ./chunks

출력:
    chunks/msh_chunks.json
"""

from __future__ import annotations

import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "msh"
OUTPUT_DIR = BASE_DIR / "data" / "output" / "msh" 
# Member Guide 좌측 사이드바 제거 x 기준 (pt)
# --debug-sidebar 로 실제 값 확인 후 조정
SIDEBAR_X_THRESHOLD = 140.0

INSURER = "msh_china"

# ---------------------------------------------------------------------------
# 파일 목록
# ---------------------------------------------------------------------------
FILES: List[Dict[str, Any]] = [
    {
        "path":                DATA_DIR / "MSH_Members_Guide.pdf",
        "doc_type":            "member_guide",
        "doc_year":            2024,
        "plan":                "all",
        "language":            "en",
    },
    {
        "path":                DATA_DIR / "MSH_Claim_Form.pdf",
        "doc_type":            "claim_form",
        "doc_year":            2024,
        "plan":                "all",
        "language":            "en",
    },
]

# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """일반 PDF — 전체 텍스트 추출 (Claim Form 등)."""
    pages: List[Tuple[int, str]] = []
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append((i + 1, text))
    finally:
        doc.close()
    return pages


def read_pdf_pages_no_sidebar(
    pdf_path: Path,
    x_threshold: float = SIDEBAR_X_THRESHOLD,
) -> List[Tuple[int, str]]:
    """
    좌측 고정 사이드바 PDF — x0 >= x_threshold 블록만 추출 (Member Guide 등).
    block 구조: (x0, y0, x1, y1, text, block_no, block_type)
    block_type=0 이 텍스트 블록.
    """
    pages: List[Tuple[int, str]] = []
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            blocks = page.get_text("blocks")
            lines = [
                b[4] for b in blocks
                if b[6] == 0 and b[0] >= x_threshold
            ]
            text = clean_text("\n".join(lines))
            if text:
                pages.append((i + 1, text))
    finally:
        doc.close()
    return pages


def build_metadata(
    file_info: Dict[str, Any],
    page: int,
    section: Optional[str] = None,
    subsection: Optional[str] = None,
    chunk_type: str = "section_text",
    question: Optional[str] = None,
) -> Dict[str, Any]:
    """
    팀 공통 메타데이터 스키마.

    공통 키:
        insurer, doc_type, source, page, doc_year,
        is_latest, language, section, chunk_type, korea_applicability

    MSH 특화 키:
        plan_package
    """
    meta: Dict[str, Any] = {
        # 보험사 공통
        "insurer":             INSURER,
        "doc_type":            file_info["doc_type"],
        "source":              file_info["path"].name,
        "page":                page,
        "language":            file_info.get("language", "en"),
        "plan":                file_info.get("plan", "all"),
    }

    if section:
        meta["section"] = section
    if subsection:
        meta["subsection"] = subsection
    if question:
        meta["question"] = question

    return meta


def make_chunk(
    chunk_id: str,
    content: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "content":  content,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Member Guide 청킹
# ---------------------------------------------------------------------------
GUIDE_SECTION_PATTERNS = [
    (r"^your plan$",                   "Your Plan"),
    (r"^your reimbursements?$",        "Your Reimbursements"),
    (r"^your online services?$",       "Your Online Services"),
    (r"^faq",                          "FAQs"),
    (r"^frequently asked questions?$", "FAQs"),
    (r"^legal information$",           "Legal Information"),
    (r"^contacts?$",                   "Contacts"),
]

GUIDE_SUBSECTION_PATTERNS = [
    r"^your login details",
    r"^your certificate of insurance",
    r"^your insurance card",
    r"^your zone of coverage",
    r"^your benefits?$",
    r"^your medical evacuation benefit",
    r"^how to manage your plan",
    r"^how to submit your claims?",
    r"^limit co.payment",
    r"^application of upper limits",
    r"^precertification agreement",
    r"^your members[' ]+area",
    r"^your mobile application",
    r"^your medical network",
    r"^role of the msh medical team",
    r"^your telemedicine service",
]

GUIDE_NOISE_PATTERNS = [
    r"^\d+\s*$",
    r"^[◄►▶◀]\s*\d+\s*[◄►▶◀]?$",
    r"^good to know$",
    r"^important$",
    r"^to find out more$",
    r"^useful (tips|information)$",
]

FAQ_QUESTION_PATTERNS = [
    r"^when does",
    r"^until what age",
    r"^i am travel",
    r"^what support",
    r"^how to change",
    r"^how to add",
    r"^i am unsure",
    r"^when can i terminate",
    r"^what is the difference",
    r".+\?$",
]


def _guide_section(line: str) -> Optional[str]:
    s = line.strip().lower()
    for pattern, label in GUIDE_SECTION_PATTERNS:
        if re.search(pattern, s):
            return label
    return None


def _guide_subsection(line: str) -> bool:
    stripped = line.strip()
    s = stripped.lower()

    if not any(re.search(p, s) for p in GUIDE_SUBSECTION_PATTERNS):
        return False

    # 헤더 구분 조건:
    # 1. 마침표로 끝나는 문장은 본문 → 제외
    if stripped.endswith("."):
        return False
    # 2. 너무 긴 문장은 본문 → 제외 (헤더는 보통 70자 이내)
    if len(stripped) > 70:
        return False
    # 3. 소문자 단어가 3개 이상 연속이면 본문 → 제외
    words = stripped.split()
    lower_words = sum(1 for w in words if w[0].islower())
    if lower_words >= 3:
        return False

    return True


def _guide_noise(line: str) -> bool:
    s = line.strip().lower()
    return not s or any(re.search(p, s) for p in GUIDE_NOISE_PATTERNS)


def _faq_question(line: str) -> bool:
    stripped = line.strip()
    s = stripped.lower()

    # 너무 짧은 조각은 질문으로 보지 않음 (멀티라인 질문의 뒷부분 방지)
    if len(stripped) < 20:
        return False
    # 소문자로 시작하면 앞 줄의 연속 → 제외
    if stripped and stripped[0].islower():
        return False

    return any(re.search(p, s) for p in FAQ_QUESTION_PATTERNS)


def chunk_member_guide(
    pages: List[Tuple[int, str]],
    file_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    stem = file_info["path"].stem
    idx  = 0

    current_section:    str           = "General"
    current_subsection: Optional[str] = None
    current_page:       int           = 1
    current_lines:      List[str]     = []
    current_question:   Optional[str] = None

    def flush(chunk_type: str = "section_text"):
        nonlocal idx, current_lines, current_question
        text = "\n".join(current_lines).strip()
        if len(text) < 80:
            current_lines   = []
            current_question = None
            return

        q = current_question if chunk_type == "faq_qa" else None
        chunks.append(make_chunk(
            chunk_id=f"{stem}_chunk_{idx}",
            content=text,
            metadata=build_metadata(
                file_info=file_info,
                page=current_page,
                section=current_section,
                subsection=current_subsection,
                chunk_type=chunk_type,
                question=q,
            ),
        ))
        idx += 1
        current_lines    = []
        current_question = None

    for page_num, text in pages:
        for line in text.split("\n"):
            stripped = line.strip()
            if _guide_noise(stripped):
                continue

            sec = _guide_section(stripped)
            if sec:
                # 이미 같은 섹션이면 페이지 디자인 반복 레이블 → 노이즈로 처리
                if sec == current_section:
                    continue
                flush()
                current_section    = sec
                current_subsection = None
                current_page       = page_num
                current_lines      = [stripped]
                continue

            if _guide_subsection(stripped):
                flush()
                current_subsection = stripped
                current_page       = page_num
                current_lines      = [stripped]
                continue

            if current_section == "FAQs" and _faq_question(stripped):
                flush(chunk_type="faq_qa")
                current_page     = page_num
                current_question = stripped
                current_lines    = [stripped]
                continue

            current_lines.append(stripped)

    flush(chunk_type="faq_qa" if current_section == "FAQs" else "section_text")
    return chunks


# ---------------------------------------------------------------------------
# Claim Form 청킹
# ---------------------------------------------------------------------------
FORM_SECTION_PATTERNS = [
    r"^insured participant",
    r"^dependent",
    r"^medical procedures? or supplies?",
    r"^signature",
    r"^\d+\s*insured",
    r"^\d+\s*dependent",
    r"^\d+\s*medical",
    r"^\d+\s*signature",
]

FORM_NOISE_PATTERNS = [
    r"^dd\s*/\s*mm\s*/\s*yyyy$",
    r"^yes\s+no$",
    r"^check\s+bank transfer",
    r"^s฀",
    r"^\s*฀\s*$",
    r"^www\.",
    r"^page \d+",
    r"^msh-dr-",
]

FORM_SUMMARIES = {
    "insured": (
        "This section captures the insured participant's personal details: "
        "name, date of birth, personal insurance number, address, country of "
        "expatriation, employer, and preferred reimbursement method."
    ),
    "dependent": (
        "This section captures information about the insured member's dependents "
        "(spouse, children): name, date of birth, relationship, and whether they "
        "are enrolled in social security or another basic insurance."
    ),
    "medical": (
        "This section lists the medical procedures, services, or supplies being "
        "claimed. Each line covers: date, patient name, description of treatment, "
        "nature of illness or injury, amount paid, currency, doctor or healthcare "
        "establishment name, and country."
    ),
    "signature": (
        "This section contains the member's declaration and signature certifying "
        "accuracy of information. Includes accident circumstances if applicable "
        "and maternity/pregnancy due date if relevant."
    ),
}


def _form_section(line: str) -> bool:
    s = line.strip().lower()
    return any(re.search(p, s) for p in FORM_SECTION_PATTERNS)


def _form_noise(line: str) -> bool:
    s = line.strip().lower()
    return not s or any(re.search(p, s) for p in FORM_NOISE_PATTERNS)


def _form_summary(section: str) -> str:
    s = section.lower()
    for key, text in FORM_SUMMARIES.items():
        if key in s:
            return text
    return "This section describes information required in the MSH claim form."


def chunk_claim_form(
    pages: List[Tuple[int, str]],
    file_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    stem = file_info["path"].stem
    idx  = 0

    current_section = "General"
    current_fields: List[str] = []
    current_page = 1

    def flush():
        nonlocal idx, current_fields
        seen:   set       = set()
        unique: List[str] = []
        for f in current_fields:
            norm = f.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                unique.append(f)
        if not unique:
            current_fields = []
            return

        summary  = _form_summary(current_section)
        content  = "\n".join([
            f"Form: MSH International Health Care Claim Form",
            f"Section: {current_section}",
            summary,
            f"Fields: {', '.join(unique)}",
        ])
        chunks.append(make_chunk(
            chunk_id=f"{stem}_chunk_{idx}",
            content=content,
            metadata=build_metadata(
                file_info=file_info,
                page=current_page,
                section=current_section,
                chunk_type="form_section",
            ),
        ))
        idx += 1
        current_fields = []

    for page_num, text in pages:
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped or _form_noise(stripped):
                continue
            if _form_section(stripped):
                flush()
                current_section = stripped
                current_page    = page_num
                continue
            if 2 <= len(stripped) <= 200:
                current_fields.append(stripped)

    flush()
    return chunks


# ---------------------------------------------------------------------------
# 메인 전처리
# ---------------------------------------------------------------------------
def preprocess(
    file_list: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []
    targets = file_list or FILES

    for file_info in targets:
        path     = file_info["path"]
        doc_type = file_info["doc_type"]

        print(
            f"[INFO] 처리 중: {path.name} | "
            f"type={doc_type} | "
            f"plan={file_info.get('plan','all')} | "
            f"year={file_info['doc_year']}"
        )

        if not path.exists():
            print(f"[WARN] 파일 없음: {path}")
            continue

        if doc_type == "member_guide":
            pages  = read_pdf_pages_no_sidebar(path)
            chunks = chunk_member_guide(pages, file_info)

        elif doc_type == "claim_form":
            pages  = read_pdf_pages(path)
            chunks = chunk_claim_form(pages, file_info)

        elif doc_type == "policy_wording":
            print(f"[WARN] policy_wording 미구현: {path.name}")
            continue

        else:
            print(f"[WARN] 지원하지 않는 doc_type: {doc_type}")
            continue

        print(f"[INFO] 청크 생성: {len(chunks)}개 ← {path.name}")
        all_chunks.extend(chunks)

    return all_chunks


def save_chunks(
    chunks: List[Dict[str, Any]],
    output_dir: Path,
    filename: str = "msh_guide_claim_form_chunk.json",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[DONE] {len(chunks)}개 청크 저장 → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 사이드바 threshold 튜닝 헬퍼
# ---------------------------------------------------------------------------
def debug_sidebar(pdf_path: Path, max_pages: int = 3) -> None:
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            blocks = page.get_text("blocks")
            print(f"\n[PAGE {i+1}] 텍스트 블록 x0 분포:")
            for b in sorted(blocks, key=lambda x: x[0]):
                if b[6] == 0:
                    preview = b[4].replace("\n", " ")[:60]
                    print(f"  x0={b[0]:6.1f}  '{preview}'")
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MSH PDF 전처리 → JSON")
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="JSON 출력 디렉토리"
    )
    parser.add_argument(
        "--debug-sidebar", action="store_true",
        help="Member Guide PDF 블록 x0 좌표 출력 (threshold 튜닝용)"
    )
    args = parser.parse_args()

    if args.debug_sidebar:
        guide_path = DATA_DIR / "MSH_Members_Guide.pdf"
        debug_sidebar(guide_path)
        return
    chunks = preprocess()
    description_chunks = [          
            {
            "chunk_id": "msh_china_claim_form_description",
            "content": "MSH International Health Care Claim Form. Used to request reimbursement for medical expenses. Requires original receipts and medical reports.",
            "metadata": {
                "insurer": "msh_china",
                "doc_type": "claim_form",
                "source": "MSH_Claim_Form.pdf",
                "page": 1,
                "language": "en",
                "plan": "all",
            }
        },
    ]
    chunks.extend(description_chunks)
    save_chunks(chunks, args.output_dir)  # 이것도 추가
    import sys
    sys.path.append(str(BASE_DIR))
    from utils.ingest_to_db import ingest
    ingest(chunks)

    print(f"\n[SUMMARY]")
    print(f"  전체 청크: {len(chunks)}")
    for doc_type in ["member_guide", "claim_form", "policy_wording"]:
        n = sum(1 for c in chunks if c["metadata"]["doc_type"] == doc_type)
        if n:
            print(f"  {doc_type}: {n}개")


if __name__ == "__main__":
    main()
