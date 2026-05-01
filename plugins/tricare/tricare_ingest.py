"""
tricare_ingest.py
─────────────────
TRICARE 벡터 DB 구축 전용 스크립트

위치: plugins/tricare/tricare_ingest.py

경로는 모두 이 파일 위치 기준으로 자동 계산됨.
DATA_DIR, DB 경로 등을 수동으로 수정할 필요 없음.

프로젝트 구조 (자동 인식):
    Darecare_LLM/                  <- BASE_DIR (루트)
    ├── data/tricare/guide/        <- DATA_DIR  (PDF/CSV 여기)
    ├── plugins/tricare/
    │   └── tricare_ingest.py      <- 이 파일
    ├── vectordb/                  <- DB 저장 위치 (tricare_plans 컬렉션)
    └── .env

실행 방법:
    python plugins/tricare/tricare_ingest.py        # 증분 추가
    python plugins/ingest_all.py                    # 전체 보험사 일괄 실행

# 05.01 - ingest_to_db 방식으로 통합
# 기존: langchain Chroma를 직접 사용 → vectordb/tricare/ 별도 디렉토리에 저장
#       (tricare_rag, tricare_cost_tables 컬렉션)
# 변경: ingest_to_db.ingest() 사용 → vectordb/에 tricare_plans 컬렉션으로 저장
# 이유: retrieve_node.py가 {insurer}_plans 패턴 컬렉션만 조회하므로
#       기존 방식으로는 guide 데이터가 within_node 검색에 포함되지 않음
"""

from __future__ import annotations

import re
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any

import json
import sys

import fitz          # PyMuPDF  (pip install pymupdf)
import pdfplumber    # (pip install pdfplumber)
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# plugins/tricare/tricare_ingest.py
#   .parents[0] → plugins/tricare/
#   .parents[1] → plugins/
#   .parents[2] → Darecare_LLM/  (루트)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "tricare" / "guide"

# 05.01 - ingest_to_db import (cigna/ingest.py와 동일한 방식)
sys.path.append(str(BASE_DIR))
from utils.ingest_to_db import ingest as ingest_to_db

load_dotenv(dotenv_path=BASE_DIR / ".env")



MIN_CHUNK_CHARS = 80   # PDF 단락 최소 길이 — 헤더/페이지번호 노이즈 제거



# location : "OCONUS" → 전 페이지 사용
#            "BOTH"   → OCONUS_KEYWORDS 포함 페이지만 사용
# table    : True → chroma_db2(표 저장소)에도 pdfplumber로 추가

RAG_PDF_FILES: List[Dict[str, Any]] = [
    {
        "path": DATA_DIR / "Costs_Fees.pdf",
        "location": "BOTH",
        "table": True,
        "doc_type": "cost_guide",
        "plan": "all",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "Overseas_HB(해외 프로그램 안내서).pdf",
        "location": "OCONUS",
        "table": False,
        "doc_type": "handbook",
        "plan": "all",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "Pharmacy_HB(tricare 약국 프로그램 안내서).pdf",
        "location": "BOTH",
        "table": True,
        "doc_type": "handbook",
        "plan": "all",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "TOP_Handbook_AUG_2023_FINAL_092223_508 (1).pdf",
        "location": "OCONUS",
        "table": False,
        "doc_type": "handbook",
        "plan": "TRICARE Overseas Program",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "NGR_HB(국가방위군 및 예비군을 위한 트라이케어 안내서).pdf",
        "location": "BOTH",
        "table": True,
        "doc_type": "handbook",
        "plan": "NGR",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "TFL_HB(평생 트라이케어).pdf",
        "location": "BOTH",
        "table": True,
        "doc_type": "handbook",
        "plan": "TRICARE For Life",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "TFL_HB(2022).pdf",
        "location": "BOTH",
        "table": True,
        "doc_type": "handbook",
        "plan": "TRICARE For Life",
        "is_latest": False,
    },
    {
        "path": DATA_DIR / "TRICARE_ADDP_HB_FINAL_508c(현역 군인 치과 프로그램 안내서).pdf",
        "location": "BOTH",
        "table": False,
        "doc_type": "handbook",
        "plan": "ADDP",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "ADDP_Brochure_FINAL_122624_508c.pdf",
        "location": "BOTH",
        "table": False,
        "doc_type": "brochure",
        "plan": "ADDP",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "Maternity_Br (1).pdf",
        "location": "BOTH",
        "table": False,
        "doc_type": "brochure",
        "plan": "all",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "Retiring_NGR_Br.pdf",
        "location": "BOTH",
        "table": False,
        "doc_type": "brochure",
        "plan": "NGR",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "QLEs_FS (2).pdf",
        "location": "BOTH",
        "table": False,
        "doc_type": "fact_sheet",
        "plan": "all",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "Medicare_Turning_65_Br.pdf",
        "location": "BOTH",
        "table": False,
        "doc_type": "brochure",
        "plan": "TRICARE For Life",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "Medicare_Under_65_Br_7.pdf",
        "location": "BOTH",
        "table": False,
        "doc_type": "brochure",
        "plan": "TRICARE For Life",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "Plans_Overview_FS_1.pdf",
        "location": "BOTH",
        "table": False,
        "doc_type": "fact_sheet",
        "plan": "all",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "Retiring_AD_Br.pdf",
        "location": "BOTH",
        "table": False,
        "doc_type": "brochure",
        "plan": "all",
        "is_latest": True,
    },
]

CSV_FILES: Dict[str, Path] = {
    "mental":     DATA_DIR / "mental_health_services.csv",
    "costs":      DATA_DIR / "Health_Plan_Costs.csv",
    "plans":      DATA_DIR / "TricarePlans.csv",
    "exclusions": DATA_DIR / "tricare_exclusions.csv",
}

CSV_METADATA: Dict[str, Dict[str, Any]] = {
    "mental":     {"doc_type": "csv_data", "plan": "all", "chunk_type": "text"},
    "costs":      {"doc_type": "csv_data", "plan": "all", "chunk_type": "table"},
    "plans":      {"doc_type": "csv_data", "plan": "all", "chunk_type": "text"},
    "exclusions": {"doc_type": "csv_data", "plan": "all", "chunk_type": "text"},
}




def clean_text(text: str) -> str:
    """PDF 특수문자/중복 공백 정제"""
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_cell(cell: str) -> str:
    """표 셀 체크마크 → Covered / Not covered 변환"""
    if not cell:
        return "N/A"
    cell = cell.strip()
    if cell in ["✓", "√", "v", "V", "●", "Yes", "yes", "Y"]:
        return "Covered"
    if cell in ["✗", "×", "x", "X", "✘", "No", "no", "N"]:
        return "Not covered"
    return cell


def is_noise_line(line: str) -> bool:
    """반복 헤더/URL 등 노이즈 줄 판별"""
    noise_patterns = [
        r"^group a.*group b",
        r"^covered service.*group",
        r"^are you in group",
        r"^this is an overview",
        r"^visit www\.tricare",
        r"^for more information.*go to",
        r"^updated (january|february|march|april|may|june|july|august|"
        r"september|october|november|december)",
    ]
    line_lower = line.lower().strip()
    return any(re.match(pat, line_lower) for pat in noise_patterns)


OCONUS_KEYWORDS = [
    "overseas", "oconus", "outside the continental",
    "south korea", "korea", "usfk",
    "outside the u.s.", "outside the united states",
    "international", "host nation", "foreign country",
    "tricare prime overseas", "tricare select overseas",
    "tricare prime remote overseas",
    "near patient", "overseas claim",
]


def is_oconus_relevant(text: str) -> bool:
    """BOTH 문서에서 해외/한국 키워드 포함 페이지만 통과"""
    return any(kw in text.lower() for kw in OCONUS_KEYWORDS)


TRICARE_SEARCH_TAGS = [
    "coverage", "covered", "not covered", "benefit", "cost", "copay",
    "deductible", "enrollment fee", "cost-share", "prior authorization",
    "TRICARE Prime", "TRICARE Select", "TRICARE For Life",
    "Group A", "Group B", "active duty", "retiree", "reserve",
    "OCONUS", "overseas", "USFK", "주한미군",
    "보장", "비용", "본인부담금", "사전승인", "해외",
]


def enrich_tricare_text(text: str, location: str) -> str:
    """
    본문 뒤에 [search_tags] 블록 추가.
    임베딩엔 포함되지만 LLM 전달 시엔 tricare_core.format_docs()가 제거함.
    """
    tags = "\n".join([
        "[search_tags]",
        "insurer: TRICARE",
        f"location: {location}",
        "keywords: " + ", ".join(TRICARE_SEARCH_TAGS),
    ])
    return f"{text}\n\n{tags}"




def check_files() -> None:
    """실행 전 PDF/CSV 파일 존재 여부 출력"""
    print("=" * 60)
    print("[ 경로 확인 ]")
    print(f"  BASE_DIR : {BASE_DIR}")
    print(f"  DATA_DIR : {DATA_DIR}")

    print("\n  [ RAG PDF ]")
    for f in RAG_PDF_FILES:
        status = "OK     " if f["path"].exists() else "MISSING"
        print(f"  [{status}] {f['path'].name}")

    print("\n  [ CSV ]")
    for key, path in CSV_FILES.items():
        status = "OK     " if path.exists() else "MISSING"
        print(f"  [{status}] {path.name}")
    print("=" * 60 + "\n")


# PDF 로드 + 청킹

def load_pdf_chunks() -> List[Document]:
    all_chunks: List[Document] = []
    total_skipped = 0

    for file_info in RAG_PDF_FILES:
        path: Path    = file_info["path"]
        location: str = file_info["location"]
        fname         = path.name

        if not path.exists():
            print(f"[WARN] 파일 없음, 건너뜀: {fname}")
            continue

        doc_fitz   = fitz.open(str(path))
        kept_pages: List[Document] = []
        skipped    = 0

        for i, page in enumerate(doc_fitz):
            raw = clean_text(page.get_text("text"))
            if not raw:
                skipped += 1
                continue
            if location == "BOTH" and not is_oconus_relevant(raw):
                skipped += 1
                continue

            lines   = [ln for ln in raw.split("\n") if not is_noise_line(ln)]
            cleaned = "\n".join(lines)

            kept_pages.append(Document(
                page_content=cleaned,
                metadata={
                    "insurer":        "TRICARE",
                    "doc_type":       file_info.get("doc_type", "handbook"),
                    "document_group": "guide",
                    "source":         fname,
                    "page":           i + 1,
                    "is_latest":      file_info.get("is_latest", True),
                    "language":       "en",
                    "plan":           file_info.get("plan", "all"),
                    "location":       location,
                    "chunk_type":     "text",
                }
            ))

        doc_fitz.close()
        total_skipped += skipped
        print(
            f"[INFO] {fname} [{location}]: "
            f"{len(kept_pages) + skipped}p → {len(kept_pages)}p 사용"
        )

        for doc in kept_pages:
            paragraphs = [
                p.strip() for p in doc.page_content.split("\n\n")
                if len(p.strip()) >= MIN_CHUNK_CHARS
            ]
            if not paragraphs and len(doc.page_content.strip()) >= MIN_CHUNK_CHARS:
                paragraphs = [doc.page_content.strip()]

            for para in paragraphs:
                all_chunks.append(Document(
                    page_content=enrich_tricare_text(para, location),
                    metadata=doc.metadata.copy()
                ))

    print(f"\n[INFO] PDF 총 청크: {len(all_chunks)}개 | 제외 페이지: {total_skipped}개")
    return all_chunks


# CSV 로드 + 청킹

def _load_csv_mental(path: Path) -> List[Document]:
    docs: List[Document] = []
    meta = CSV_METADATA["mental"]
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            service = row.get("service", row.get("서비스", "")).strip()
            content = " | ".join(f"{k}: {v}" for k, v in row.items() if v and v.strip())
            if not content:
                continue
            docs.append(Document(
                page_content=f"서비스: {service}\n내용: {content}",
                metadata={
                    "insurer":        "TRICARE",
                    "doc_type":       meta["doc_type"],
                    "document_group": "guide",
                    "source":         path.name,
                    "page":           0,
                    "is_latest":      True,
                    "language":       "en",
                    "plan":           meta["plan"],
                    "chunk_type":     meta["chunk_type"],
                    "service_name":   service,
                }
            ))
    return docs


def _load_csv_costs(path: Path) -> List[Document]:
    docs: List[Document] = []
    meta = CSV_METADATA["costs"]
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = " | ".join(f"{k}: {v}" for k, v in row.items() if v and v.strip())
            if not content:
                continue
            docs.append(Document(
                page_content=content,
                metadata={
                    "insurer":        "TRICARE",
                    "doc_type":       meta["doc_type"],
                    "document_group": "guide",
                    "source":         path.name,
                    "page":           0,
                    "is_latest":      True,
                    "language":       "en",
                    "plan":           meta["plan"],
                    "chunk_type":     meta["chunk_type"],
                }
            ))
    return docs


def _load_csv_plans(path: Path) -> List[Document]:
    docs: List[Document] = []
    meta = CSV_METADATA["plans"]
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plan_name = row.get("plan_name", row.get("플랜명", "")).strip()
            content   = " | ".join(f"{k}: {v}" for k, v in row.items() if v and v.strip())
            if not content:
                continue
            docs.append(Document(
                page_content=f"플랜명: {plan_name}\n내용: {content}",
                metadata={
                    "insurer":        "TRICARE",
                    "doc_type":       meta["doc_type"],
                    "document_group": "guide",
                    "source":         path.name,
                    "page":           0,
                    "is_latest":      True,
                    "language":       "en",
                    "plan":           plan_name or meta["plan"],
                    "chunk_type":     meta["chunk_type"],
                }
            ))
    return docs


def _load_csv_exclusions(path: Path) -> List[Document]:
    docs: List[Document] = []
    meta = CSV_METADATA["exclusions"]
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name    = row.get("name", row.get("항목", "")).strip()
            content = row.get("content", row.get("내용", "")).strip()
            url     = row.get("url", "").strip()
            if not content:
                continue
            docs.append(Document(
                page_content=f"제외항목: {name}\n내용: {content}",
                metadata={
                    "insurer":        "TRICARE",
                    "doc_type":       meta["doc_type"],
                    "document_group": "guide",
                    "source":         path.name,
                    "page":           0,
                    "is_latest":      True,
                    "language":       "en",
                    "plan":           meta["plan"],
                    "chunk_type":     meta["chunk_type"],
                    "exclusion_name": name,
                    "url":            url,
                }
            ))
    return docs


def load_csv_chunks() -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    loaders = {
        "mental":     (_load_csv_mental,     CSV_FILES["mental"]),
        "costs":      (_load_csv_costs,      CSV_FILES["costs"]),
        "plans":      (_load_csv_plans,      CSV_FILES["plans"]),
        "exclusions": (_load_csv_exclusions, CSV_FILES["exclusions"]),
    }

    all_chunks: List[Document] = []
    for key, (loader_fn, path) in loaders.items():
        if not path.exists():
            print(f"[WARN] CSV 없음, 건너뜀: {path.name}")
            continue
        raw_docs = loader_fn(path)
        chunks   = text_splitter.split_documents(raw_docs)
        all_chunks.extend(chunks)
        print(f"[INFO] {path.name}: {len(raw_docs)}행 → {len(chunks)}청크")

    print(f"[INFO] CSV 총 청크: {len(all_chunks)}개")
    return all_chunks


# 표 전용 청킹 (pdfplumber)

def load_table_chunks() -> List[Document]:
    all_chunks: List[Document] = []

    for file_info in RAG_PDF_FILES:
        if not file_info.get("table"):
            continue

        path: Path = file_info["path"]
        fname      = path.name
        if not path.exists():
            print(f"[WARN] 파일 없음, 건너뜀: {fname}")
            continue

        table_count = 0
        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for table in (page.extract_tables() or []):
                    if not table:
                        continue
                    rows_text = [
                        " | ".join(normalize_cell(str(c or "")) for c in row)
                        for row in table
                    ]
                    table_text = "\n".join(rows_text)
                    if len(table_text.strip()) < 30:
                        continue
                    all_chunks.append(Document(
                        page_content=table_text,
                        metadata={
                            "insurer":        "TRICARE",
                            "doc_type":       "cost_table",
                            "document_group": "guide",
                            "source":         fname,
                            "page":           page_num,
                            "is_latest":      file_info.get("is_latest", True),
                            "language":       "en",
                            "plan":           file_info.get("plan", "all"),
                            "location":       file_info["location"],
                            "chunk_type":     "table",
                        }
                    ))
                    table_count += 1

        print(f"[INFO] {fname}: 표 {table_count}개 추출")

    # Health_Plan_Costs.csv도 표 저장소에 추가
    costs_path = CSV_FILES["costs"]
    if costs_path.exists():
        costs_docs = _load_csv_costs(costs_path)
        all_chunks.extend(costs_docs)
        print(f"[INFO] Health_Plan_Costs.csv: {len(costs_docs)}행 → 표 저장소 추가")

    print(f"[INFO] 표 총 청크: {len(all_chunks)}개")
    return all_chunks


# 05.01 - ingest_to_db 통합을 위한 변환 + 진입점

def docs_to_chunks(documents: List[Document]) -> list:
    """
    langchain Document 리스트 → ingest_to_db.ingest() 입력 형식으로 변환.

    ingest_to_db는 insurer 필드(소문자)로 컬렉션을 결정하므로
    metadata의 "TRICARE" → "tricare" 로 변환한다.
    """
    chunks = []
    for idx, doc in enumerate(documents):
        meta       = doc.metadata.copy()
        source_id  = re.sub(r"[^\w]", "_", meta.get("source", "unknown"))
        page       = meta.get("page", 0)
        chunk_type = meta.get("chunk_type", "text")
        meta["insurer"] = "tricare"   # INSURER_TO_COLLECTION 매핑 키와 일치
        chunks.append({
            "chunk_id": f"tricare_{source_id}_{chunk_type}_{page}_{idx}",
            "text"    : doc.page_content,
            "metadata": meta,
        })
    return chunks


def run() -> None:
    """plugins/ingest_all.py 진입점 — tricare_plans 컬렉션에 저장"""
    check_files()

    pdf_chunks   = load_pdf_chunks()
    csv_chunks   = load_csv_chunks()
    table_chunks = load_table_chunks()

    all_docs = pdf_chunks + csv_chunks + table_chunks
    print(f"\n[tricare] 총 청크: {len(all_docs)}개 (PDF+CSV+표)")

    ingest_to_db(docs_to_chunks(all_docs))


if __name__ == "__main__":
    run()
