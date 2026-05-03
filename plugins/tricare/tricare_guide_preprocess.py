"""
tricare_guide_preprocess.py
TRICARE guide PDFs + CSVs → tricare_plans 컬렉션 인제스트

실행:
    python plugins/tricare/tricare_guide_preprocess.py
"""

from __future__ import annotations

import re
import csv
import sys
from pathlib import Path

import fitz

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from utils.ingest_to_db import ingest

DATA_DIR = BASE_DIR / "data" / "tricare" / "guide"
MIN_CHUNK = 80

OCONUS_KEYWORDS = [
    "overseas", "oconus", "outside the continental",
    "south korea", "korea", "usfk",
    "outside the u.s.", "outside the united states",
    "international", "host nation", "foreign country",
    "tricare prime overseas", "tricare select overseas",
    "near patient", "overseas claim",
]

PDF_FILES = [
    {"name": "Overseas_HB(해외 프로그램 안내서).pdf",                       "plan": "all",                   "filter": "oconus"},
    {"name": "TOP_Handbook_AUG_2023_FINAL_092223_508 (1).pdf",            "plan": "TRICARE Overseas Program","filter": "oconus"},
    {"name": "Costs_Fees.pdf",                                             "plan": "all",                   "filter": "both"},
    {"name": "Pharmacy_HB(tricare 약국 프로그램 안내서).pdf",               "plan": "all",                   "filter": "both"},
    {"name": "NGR_HB(국가방위군 및 예비군을 위한 트라이케어 안내서).pdf",   "plan": "NGR",                   "filter": "both"},
    {"name": "TFL_HB(평생 트라이케어).pdf",                                "plan": "TRICARE For Life",      "filter": "both"},
    {"name": "TRICARE_ADDP_HB_FINAL_508c(현역 군인 치과 프로그램 안내서).pdf","plan": "ADDP",               "filter": "both"},
    {"name": "ADDP_Brochure_FINAL_122624_508c.pdf",                        "plan": "ADDP",                  "filter": "both"},
    {"name": "Maternity_Br (1).pdf",                                       "plan": "all",                   "filter": "both"},
    {"name": "QLEs_FS (2).pdf",                                            "plan": "all",                   "filter": "both"},
]

CSV_FILES = [
    {"name": "mental_health_services.csv", "plan": "all"},
    {"name": "Health_Plan_Costs.csv",      "plan": "all"},
    {"name": "TricarePlans.csv",           "plan": "all"},
    {"name": "tricare_exclusions.csv",     "plan": "all"},
]


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ").replace("​", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_oconus(text: str) -> bool:
    return any(kw in text.lower() for kw in OCONUS_KEYWORDS)


def load_pdf_chunks(file_info: dict) -> list[dict]:
    path = DATA_DIR / file_info["name"]
    if not path.exists():
        print(f"[WARN] 파일 없음: {file_info['name']}")
        return []

    chunks = []
    doc = fitz.open(str(path))
    try:
        for i, page in enumerate(doc):
            text = clean_text(page.get_text("text"))
            if not text:
                continue
            if file_info["filter"] == "both" and not is_oconus(text):
                continue

            for para in re.split(r"\n\n+", text):
                para = para.strip()
                if len(para) < MIN_CHUNK:
                    continue
                chunks.append({
                    "chunk_id": f"tricare_{path.stem}_p{i+1}_{len(chunks)}",
                    "content":  para,
                    "metadata": {
                        "insurer":   "tricare",
                        "doc_type":  "handbook",
                        "source":    file_info["name"],
                        "page":      i + 1,
                        "plan":      file_info["plan"],
                        "language":  "en",
                        "is_latest": True,
                    },
                })
    finally:
        doc.close()

    print(f"[INFO] {file_info['name']}: {len(chunks)}청크")
    return chunks


def load_csv_chunks(file_info: dict) -> list[dict]:
    path = DATA_DIR / file_info["name"]
    if not path.exists():
        print(f"[WARN] CSV 없음: {file_info['name']}")
        return []

    chunks = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            content = " | ".join(f"{k}: {v}" for k, v in row.items() if v and v.strip())
            if len(content) < MIN_CHUNK:
                continue
            chunks.append({
                "chunk_id": f"tricare_{path.stem}_row{i}",
                "content":  content,
                "metadata": {
                    "insurer":   "tricare",
                    "doc_type":  "csv_data",
                    "source":    file_info["name"],
                    "page":      0,
                    "plan":      file_info["plan"],
                    "language":  "en",
                    "is_latest": True,
                },
            })

    print(f"[INFO] {file_info['name']}: {len(chunks)}청크")
    return chunks


def main():
    all_chunks = []

    for f in PDF_FILES:
        all_chunks.extend(load_pdf_chunks(f))
    for f in CSV_FILES:
        all_chunks.extend(load_csv_chunks(f))

    print(f"\n[INFO] 총 청크: {len(all_chunks)}개")
    ingest(all_chunks)


if __name__ == "__main__":
    main()
