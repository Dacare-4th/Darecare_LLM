# Cigna PDF 전처리 — 텍스트 청킹 + 표 분리 저장 (B안)
# 벡터DB 저장은 하지 않음. 결과는 data/cigna/processed/ 에 JSON으로 저장.
#
# 사용법: python -m plugins.cigna.ingest
#
# 출력 파일:
#   data/cigna/processed/cigna_{doc_type}_{year}.json

import json
import os
import re
from pathlib import Path

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = str(BASE_DIR / "data" / "cigna" / "processed")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ──────────────────────────────────────────
# PDF 소스 목록
# ──────────────────────────────────────────

PDF_SOURCES = [
    # Benefits_Summary
    {
        "path":     BASE_DIR / "data/cigna/Benefits_Summary/591116 Cigna_Global_International_Health_Plans_Benefits_Summary_USD_EN_0523.pdf",
        "doc_type": "benefits_summary",
        "year":     "2023",
        "is_latest": False,
    },
    {
        "path":     BASE_DIR / "data/cigna/Benefits_Summary/591116 Cigna Global Benefits Summary USD_EN_0924.pdf",
        "doc_type": "benefits_summary",
        "year":     "2024",
        "is_latest": False,
    },
    {
        "path":     BASE_DIR / "data/cigna/Benefits_Summary/591116-cigna-global-benefits-summary-usd_en_02_2025.pdf",
        "doc_type": "benefits_summary",
        "year":     "2025",
        "is_latest": True,
    },
    # Customer_Guide
    {
        "path":     BASE_DIR / "data/cigna/Customer_Guide/200008 CGHO Customer Guide EN_05_2019.pdf",
        "doc_type": "customer_guide",
        "year":     "2019",
        "is_latest": False,
    },
    {
        "path":     BASE_DIR / "data/cigna/Customer_Guide/591048 CGHO Customer Guide EN_05_2022.pdf",
        "doc_type": "customer_guide",
        "year":     "2022",
        "is_latest": False,
    },
    {
        "path":     BASE_DIR / "data/cigna/Customer_Guide/591048-cgho-customer-guide-en_05_2023.pdf",
        "doc_type": "customer_guide",
        "year":     "2023",
        "is_latest": False,
    },
    {
        "path":     BASE_DIR / "data/cigna/Customer_Guide/Cigna-Global-Health-Options-Customer-Guide_02_2026.pdf",
        "doc_type": "customer_guide",
        "year":     "2026",
        "is_latest": True,
    },
    # Policy_Rules
    {
        "path":     BASE_DIR / "data/cigna/Policy_Rules/200008 CGHO Customer Guide EN_05_2019.pdf",
        "doc_type": "policy_rules",
        "year":     "2019",
        "is_latest": False,
    },
    {
        "path":     BASE_DIR / "data/cigna/Policy_Rules/CGHO Policy Rules CGIC NA_EN_05_2023.pdf",
        "doc_type": "policy_rules",
        "year":     "2023",
        "is_latest": False,
    },
    {
        "path":     BASE_DIR / "data/cigna/Policy_Rules/CGHO Policy Rules CGIC_EN_02_2024.pdf",
        "doc_type": "policy_rules",
        "year":     "2024",
        "is_latest": False,
    },
    {
        "path":     BASE_DIR / "data/cigna/Policy_Rules/CGHO Policy Rules CGIC_EN_02_2025.pdf",
        "doc_type": "policy_rules",
        "year":     "2025",
        "is_latest": False,
    },
    {
        "path":     BASE_DIR / "data/cigna/Policy_Rules/CGHP Policy Rules CGIC EN 02_2026.pdf",
        "doc_type": "policy_rules",
        "year":     "2026",
        "is_latest": True,
    },
]

# ──────────────────────────────────────────
# 텍스트 청킹
# ──────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


# ──────────────────────────────────────────
# Cigna 전용 표 전처리 헬퍼
# ──────────────────────────────────────────

_CHECK = {"", "", "✓", "✔", "☑", "✅"}
_CROSS = {"⊗", "✗", "✘", "❌", "×", "☒", "⦻", ""}
_BADGE = re.compile(r"^(Updated|New\b|\d+\s*MONTHS?)", re.IGNORECASE)


def _cvt(cell, is_data: bool) -> str:
    """셀 값 변환: 체크마크 → Covered, 크로스 → Not Covered."""
    if cell is None or not str(cell).strip():
        return "Not Covered" if is_data else ""
    t = str(cell).strip().replace("\n", " ")
    chk = any(g in t for g in _CHECK)
    crs = any(g in t for g in _CROSS)
    clean = t
    for g in _CHECK | _CROSS:
        clean = clean.replace(g, "")
    clean = clean.strip()
    if crs:
        return "Not Covered"
    if chk:
        return f"Covered{' - ' + clean if clean else ''}"
    return clean or ("Not Covered" if is_data else "")


def _is_data(row) -> bool:
    """데이터 행 여부: 금액/퍼센트/키워드 포함 시 True."""
    return any(
        c and re.search(r"[$€\xa3\d]|covered|paid|n/a|refund|no coverage", str(c), re.I)
        for c in row
    )


def _is_rotated(text) -> bool:
    """세로 방향 텍스트(사이드바 레이블) 여부."""
    t = (text or "").strip()
    return bool(t) and "\n" in t and t.replace("\n", "").replace(" ", "").isupper() and len(t) > 6


def _clean_benefit(text, extra_badges=()) -> str:
    """혜택 이름에서 Updated/12 MONTHS 등 뱃지를 괄호로 이동."""
    if not text and not extra_badges:
        return ""
    parts = [p.strip() for p in re.split(r"[\n/]+", text or "") if p.strip()]
    if len(parts) <= 1:
        badges  = list(extra_badges)
        content = [text.strip()] if text and text.strip() else []
    else:
        badges  = [p for p in parts if _BADGE.match(p)] + list(extra_badges)
        content = [p for p in parts if not _BADGE.match(p)]
    result = " ".join(content)
    if badges:
        result = f"{result} ({' '.join(badges)})" if result else f"({' '.join(badges)})"
    return result.strip()


def _col_map(table) -> dict | None:
    """Benefits Summary 표에서 Silver/Gold/Platinum 열 인덱스를 찾는다."""
    s = g = p = hdr = None
    for i, row in enumerate(table[:8]):
        v = [str(c or "").split("\n")[0].strip() for c in row]
        if "Silver" in v and "Gold" in v:
            s, g = v.index("Silver"), v.index("Gold")
            p = next((j for j, x in enumerate(v) if x == "Platinum"), None)
            hdr = i
            break
    if s is None:
        return None
    b = max(0, s - 1)
    for row in table[hdr + 1:]:
        if not _is_data(row):
            continue
        for ci in range(s - 1, -1, -1):
            v = str(row[ci] or "").strip()
            if v and not _is_rotated(v):
                b = ci
                break
        break
    return {"b": b, "s": s, "g": g, "p": p}


def _table_to_md(table) -> str:
    """pdfplumber 표 → 마크다운 (Cigna 전용: Silver/Gold/Platinum 컬럼 처리)."""
    cm = _col_map(table)
    lines = []
    hdr_done = False
    for row in table:
        data = _is_data(row)
        if cm:
            n          = len(row)
            b_raw      = str(row[cm["b"]] or "").strip() if cm["b"] < n else ""
            mid_badges = []
            for ci in range(cm["b"] + 1, cm["s"]):
                v = str(row[ci] or "").strip()
                if v and _BADGE.match(v):
                    mid_badges.append(v)
                elif v:
                    b_raw = f"{b_raw} {v}".strip()
            s_raw = row[cm["s"]] if cm["s"] < n else None
            g_raw = row[cm["g"]] if cm["g"] < n else None
            p_raw = row[cm["p"]] if cm["p"] and cm["p"] < n else None
            if not any(str(v or "").strip() for v in [s_raw, g_raw, p_raw]):
                data = False
            b_txt = _clean_benefit(b_raw, mid_badges)
            cells = [b_txt, _cvt(s_raw, data), _cvt(g_raw, data)]
            if cm["p"]:
                cells.append(_cvt(p_raw, data))
        else:
            cells = [_cvt(c, data) for c in row]
        if not any(str(c).strip() for c in cells):
            continue
        line = "| " + " | ".join(str(c) for c in cells) + " |"
        lines.append(line)
        if not hdr_done:
            lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
            hdr_done = True
    return "\n".join(lines)


# ──────────────────────────────────────────
# JSON 저장
# ──────────────────────────────────────────

def save_json(data: dict, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  저장: {path}")


# ──────────────────────────────────────────
# PDF 전처리 (B안: 텍스트 청크 + 표 분리)
# ──────────────────────────────────────────

def load_pdf_cigna(source: dict) -> dict:
    """Cigna PDF 한 파일 → 텍스트 청크 + 마크다운 표 분리 저장.

    반환 형식:
        {
            "metadata": {...},
            "chunks":   [{"chunk_id": ..., "text": ..., "metadata": {...}}, ...],
            "tables":   [{"table_index": int, "page": int, "markdown": str}, ...],
        }
    """
    path      = source["path"]
    doc_type  = source["doc_type"]
    year      = source["year"]
    is_latest = source["is_latest"]
    filename  = Path(path).name

    all_chunks: list[dict] = []
    all_tables: list[dict] = []

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw_text   = (page.extract_text() or "").strip()
            raw_tables = page.extract_tables() or []

            # 텍스트 청킹
            if raw_text:
                for idx, text_val in enumerate(chunk_text(raw_text), start=1):
                    all_chunks.append({
                        "chunk_id": f"cigna_{doc_type}_{year}_p{page_num}_{idx}",
                        "text": text_val,
                        "metadata": {
                            "insurer":     "cigna",
                            "doc_type":    doc_type,
                            "source":      filename,
                            "source_type": "pdf",
                            "topic":       doc_type,
                            "page":        page_num,
                            "is_latest":   is_latest,
                            "language":    "en",
                            "plan":        None,
                            "year":        year,
                            "chunk_type":  "page_text",
                            "audience":    "foreigner_only",
                        },
                    })

            # 표 마크다운 분리 저장
            for t_idx, table in enumerate(raw_tables):
                if not table or len(table) < 2:
                    continue
                md = _table_to_md(table)
                if md:
                    all_tables.append({
                        "table_index": t_idx,
                        "page":        page_num,
                        "markdown":    md,
                    })

    return {
        "metadata": {
            "insurer":     "cigna",
            "source_type": "pdf",
            "doc_type":    doc_type,
            "source":      filename,
            "language":    "en",
            "year":        year,
            "is_latest":   is_latest,
        },
        "chunks": all_chunks,
        "tables": all_tables,
    }


# ──────────────────────────────────────────
# ingest_pdf: 전체 실행
# ──────────────────────────────────────────

def ingest_pdf():
    print("[Cigna PDF] 전처리 시작")
    total_chunks = total_tables = processed = 0
    all_chunks: list[dict] = []

    for source in PDF_SOURCES:
        path     = source["path"]
        doc_type = source["doc_type"]
        year     = source["year"]
        filename = Path(path).name

        print(f"\n  [{doc_type} {year}] {filename}")

        if not Path(path).exists():
            print(f"  [오류] 파일 없음: {path}")
            continue

        result = load_pdf_cigna(source)

        n_chunks = len(result["chunks"])
        n_tables = len(result["tables"])
        all_chunks.extend(result["chunks"])
        print(f"  → 청크 {n_chunks}개 / 표 {n_tables}개")

        save_json(result, f"cigna_{doc_type}_{year}.json")

        total_chunks += n_chunks
        total_tables += n_tables
        processed    += 1

    print(f"\n[Cigna PDF] 완료 — {processed}개 파일 / 청크 {total_chunks}개 / 표 {total_tables}개")

    # 저장된 파일 목록
    saved = sorted(
        f for f in os.listdir(OUTPUT_DIR)
        if f.startswith("cigna_") and f.endswith(".json")
    )
    print(f"\n저장된 파일 ({len(saved)}개):")
    for f in saved:
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) // 1024
        print(f"  {f} ({size} KB)")

    return all_chunks
    
   

# ──────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────


import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.ingest_to_db import ingest

def run():
    all_chunks = ingest_pdf()
    description_chunks = [
        {
          "text": "Cigna Dental claim form. Required when the member has paid for dental treatment out of pocket and wants reimbursement from Cigna — this is the reimbursement claim procedure for dental expenses. Applies to all dental treatments covered under the optional International Dental benefit: preventative care (check-ups, X-rays, scaling), minor treatments (fillings, root canals, extractions), and major treatments (crowns, bridges, periodontal surgery, dentures, inlays). Not used for medical or vision claims (use the separate Medical and Vision claim form). The treating dentist must complete the treatment code section with procedure codes, number of units, dates and charges, and sign and stamp the form. Submit together with original dental invoice and payment receipt. Submitted by the member, not the dental clinic.",
          "metadata": {
            "insurer": "cigna",
            "doc_type": "claim_form",
            "form_type": "dental",
            "source_type": "form",
            "language": "en",
            "is_latest": True,
            "file_name": "592094_GIH_Dental_claim_form_EN_1125.pdf",
            "file_path": "data/cigna/claim_forms/592094_GIH_Dental_claim_form_EN_1125.pdf"
          }
        },
        {
          "text": "Cigna Medical and Vision claim form. Required when the member has already paid for treatment out of pocket and wants reimbursement from Cigna — this is the reimbursement claim procedure (as opposed to direct billing, where the hospital bills Cigna directly). Typical situations: outpatient consultations, prescribed drugs, diagnostic tests, physiotherapy, vision expenses (eye test, prescription glasses, contact lenses), or any treatment at a non-network hospital where direct billing was not arranged, including emergency cases where the member paid first. Not used for dental treatment (use the separate Dental claim form). The treating physician must complete and sign Section D with diagnosis and treatment details. Submit together with original hospital invoice and payment receipt within 12 months of treatment date. Submitted by the member, not the hospital.",
          "metadata": {
            "insurer": "cigna",
            "doc_type": "claim_form",
            "form_type": "medical_vision",
            "source_type": "form",
            "language": "en",
            "is_latest": True,
            "file_name": "591797_Medical_and_Vision_claim_form_EN_1125.pdf",
            "file_path": "data/cigna/claim_forms/591797_Medical_and_Vision_claim_form_EN_1125.pdf"
          }
        }
    ]
    all_chunks.extend(description_chunks)
    ingest(all_chunks)


if __name__ == "__main__":
    run()
