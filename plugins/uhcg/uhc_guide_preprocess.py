from pathlib import Path
import json
import re

import pdfplumber
from langchain_community.document_loaders import PyMuPDFLoader


BASE_DIR = Path(__file__).resolve().parent.parent.parent
GUIDE_DIR = BASE_DIR / "data" / "uhc" / "guide"
OUTPUT_DIR = BASE_DIR / "data" / "output" / "uhcg" / "uhcg_guide"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INSURER = "uhcg"
IS_LATEST = True


def load_pdf_pages(file_path):
    loader = PyMuPDFLoader(str(file_path))
    docs = loader.load()

    pages = []

    for doc in docs:
        pages.append({
            "page": doc.metadata.get("page", 0) + 1,
            "text": doc.page_content
        })

    return pages


def clean_text(text):
    text = text.replace("\x0c", " ")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"©.*", "", text)
    return text.strip()


def is_noise_text(text):
    text_lower = text.lower()

    noise_keywords = [
        "welcome",
        "explore the ways",
        "try a virtual visit",
        "learn more",
        "customer support",
        "client name",
    ]

    for kw in noise_keywords:
        if kw in text_lower:
            return True

    if len(text.strip()) < 30:
        return True

    return False



# 1. Welcome Guide
EXCLUDE_PAGES = [1, 15, 16]
REAL_TABLE_PAGES = [8]


def extract_table_rows(file_path, page_num):
    rows_data = []

    country_fallback = [
        "Africa",
        "Australia",
        "Bahrain, Jordan, Kuwait, Lebanon, Kingdom of Saudi Arabia, Oman, Qatar, United Arab Emirates",
        "Canada",
        "Europe, plus Austria, Belgium and Luxembourg",
        "Japan",
        "India"
    ]

    with pdfplumber.open(file_path) as pdf:
        page = pdf.pages[page_num - 1]
        tables = page.extract_tables()

        for table_idx, table in enumerate(tables, start=1):
            if not table or len(table) < 2:
                continue

            header = [
                "When you are in",
                "The locally licensed insurer or administrator will be",
                "Carry the following ID cards in this country",
                "For assistance, contact"
            ]

            rows = table[1:]

            for row_idx, row in enumerate(rows, start=1):
                row = list(row)

                while len(row) < 4:
                    row.append("")

                row = row[:4]

                if not row[0] and row_idx <= len(country_fallback):
                    row[0] = country_fallback[row_idx - 1]

                row_text_parts = []

                for h, cell in zip(header, row):
                    if cell and str(cell).strip():
                        row_text_parts.append(f"{h}: {str(cell).strip()}")

                if row_text_parts:
                    rows_data.append({
                        "table_idx": table_idx,
                        "row_idx": row_idx,
                        "content": "\n".join(row_text_parts)
                    })

    return rows_data


def make_welcome_text_chunks(file_path, table_pages):
    pages = load_pdf_pages(file_path)
    chunks = []

    for page in pages:
        page_num = page["page"]

        if page_num in EXCLUDE_PAGES:
            continue

        if page_num in table_pages:
            continue

        cleaned = clean_text(page["text"])

        if is_noise_text(cleaned):
            continue

        chunks.append({
            "chunk_id": f"{file_path.stem}_p{page_num}_text",
            "text": cleaned,
            "metadata": {
                "insurer": INSURER,
                "doc_type": "member_guide",
                "source": file_path.name,
                "page": page_num,
                "is_latest": IS_LATEST,
                "language": "en",
                "plan": None,
                "document_group": "guide",
                "chunk_type": "page_text",
                "region_scope": "global",
                "korea_applicability": "conditional"
            }
        })

    return chunks


def make_welcome_table_chunks(file_path, table_pages):
    chunks = []

    for page_num in table_pages:
        rows = extract_table_rows(file_path, page_num)

        for row in rows:
            chunks.append({
                "chunk_id": f"{file_path.stem}_p{page_num}_table{row['table_idx']}_row{row['row_idx']}",
                "text": row["content"],
                "metadata": {
                    "insurer": INSURER,
                    "doc_type": "member_guide",
                    "source": file_path.name,
                    "page": page_num,
                    "is_latest": IS_LATEST,
                    "language": "en",
                    "plan": None,
                    "document_group": "guide",
                    "chunk_type": "table_row",
                    "region_scope": "global",
                    "korea_applicability": "conditional"
                }
            })

    return chunks


def make_welcome_guide_chunks(file_path):
    table_pages = REAL_TABLE_PAGES

    text_chunks = make_welcome_text_chunks(file_path, table_pages)
    table_chunks = make_welcome_table_chunks(file_path, table_pages)

    return text_chunks + table_chunks



# 2. BeHealthy SOB
def extract_sob_table_rows(file_path):
    rows_data = []

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()

            for table_idx, table in enumerate(tables, start=1):
                if not table or len(table) < 2:
                    continue

                header = table[0]
                rows = table[1:]

                for row_idx, row in enumerate(rows, start=1):
                    row = list(row)

                    while len(row) < len(header):
                        row.append("")

                    row = row[:len(header)]

                    row_text_parts = []

                    for h, cell in zip(header, row):
                        if h and cell and str(cell).strip():
                            row_text_parts.append(
                                f"{str(h).strip()}: {str(cell).strip()}"
                            )

                    if row_text_parts:
                        rows_data.append({
                            "page": page_num,
                            "table_idx": table_idx,
                            "row_idx": row_idx,
                            "content": "\n".join(row_text_parts)
                        })

    return rows_data


def make_behealthy_sob_chunks(file_path):
    rows = extract_sob_table_rows(file_path)
    chunks = []

    for r in rows:
        chunks.append({
            "chunk_id": f"{file_path.stem}_p{r['page']}_table{r['table_idx']}_row{r['row_idx']}",
            "text": r["content"],
            "metadata": {
                "insurer": INSURER,
                "doc_type": "benefit_summary",
                "source": file_path.name,
                "page": r["page"],
                "is_latest": IS_LATEST,
                "language": "en",
                "plan": None,
                "document_group": "guide",
                "chunk_type": "benefit_table_row",
                "region_scope": "global",
                "korea_applicability": "conditional"
            }
        })

    return chunks



# 3. UHC Global Program Guide
def clean_program_text(text):
    text = clean_text(text)

    text = text.replace("UHCG-MSSNDS-UHCSR-1214", "")
    text = text.replace("￾", "")
    text = text.replace("�", "-")

    bullet_chars = ["", "", "▪", "●", "•", "◦", "·"]
    for b in bullet_chars:
        text = text.replace(b, "-")

    text = re.sub(r"-\s*\n\s*", "- ", text)
    text = re.sub(r"\n\s*-\s*", "\n- ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


SECTION_TITLES = [
    "GLOBAL EMERGENCY SERVICES",
    "PROGRAM GUIDELINES",
    "MEDICAL & SECURITY ASSISTANCE AND EVACUATION",
    "PROGRAM DESCRIPTION",
    "How To Use UnitedHealthcare Global Assistance Services",
    "MEDICAL ASSISTANCE SERVICES",
    "MEDICAL EVACUATION & REPATRIATION SERVICES",
    "WORLDWIDE DESTINATION INTELLIGENCE",
    "SECURITY AND POLITICAL EVACUATION SERVICES",
    "NATURAL DISASTER EVACUATION SERVICES",
    "TRAVEL ASSISTANCE SERVICES",
    "PROGRAM DEFINITIONS",
    "CONDITIONS AND LIMITATIONS",
]


def split_program_sections(text):
    pattern = "|".join([re.escape(title) for title in SECTION_TITLES])
    parts = re.split(f"({pattern})", text)

    sections = []

    for i in range(1, len(parts), 2):
        section_title = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""

        if len(content) >= 30:
            sections.append({
                "section_title": section_title,
                "content": content
            })

    return sections


def make_program_guide_chunks(file_path):
    pages = load_pdf_pages(file_path)
    full_text = "\n".join([clean_program_text(p["text"]) for p in pages])
    sections = split_program_sections(full_text)

    chunks = []

    for idx, section in enumerate(sections, start=1):
        section_title = section["section_title"]
        content = section["content"]

        chunks.append({
            "chunk_id": f"{file_path.stem}_section_{idx}",
            "text": f"{section_title}\n{content}",
            "metadata": {
                "insurer": INSURER,
                "doc_type": "program_guide",
                "source": file_path.name,
                "page": None,
                "is_latest": IS_LATEST,
                "language": "en",
                "plan": None,
                "document_group": "guide",
                "chunk_type": "section",
                "section_title": section_title,
                "region_scope": "global",
                "korea_applicability": "conditional"
            }
        })

    return chunks



# 4. Business Travel FAQ
def clean_faq_text(text):
    text = clean_text(text)

    text = text.replace("Business Travel Insurance | FAQs", "")
    text = text.replace(
        "Frequently asked questions about your business travel insurance plan",
        ""
    )
    text = text.replace("continued", "")

    text = re.sub(r"©.*", "", text)
    text = re.sub(r"\d{2}/\d{2}\s+MBR-BT-\d+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def split_faq_qa(text):
    question_pattern = r"(?=^[A-Z][^\n?]+\?)"
    blocks = re.split(question_pattern, text, flags=re.MULTILINE)

    qa_list = []

    for block in blocks:
        block = block.strip()

        if "?" not in block:
            continue

        question, answer = block.split("?", 1)
        question = question.strip() + "?"
        answer = answer.strip()

        if len(question) < 10 or len(answer) < 20:
            continue

        qa_list.append({
            "question": question,
            "answer": answer
        })

    return qa_list


def make_business_travel_faq_chunks(file_path):
    pages = load_pdf_pages(file_path)
    full_text = "\n".join([clean_faq_text(p["text"]) for p in pages])
    qa_list = split_faq_qa(full_text)

    chunks = []

    for idx, qa in enumerate(qa_list, start=1):
        chunks.append({
            "chunk_id": f"{file_path.stem}_faq_{idx}",
            "text": f"Question: {qa['question']}\nAnswer: {qa['answer']}",
            "metadata": {
                "insurer": INSURER,
                "doc_type": "faq",
                "source": file_path.name,
                "page": None,
                "is_latest": IS_LATEST,
                "language": "en",
                "plan": None,
                "document_group": "guide",
                "chunk_type": "faq_qa",
                "question": qa["question"],
                "region_scope": "global",
                "korea_applicability": "conditional"
            }
        })

    return chunks



def get_target_file(keyword):
    target_file = None

    for f in guide_files:
        if keyword in f.name:
            target_file = f

    return target_file


def save_chunks(chunks, file_name):
    output_path = OUTPUT_DIR / file_name

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(file_name, "저장 완료:", output_path)
    print("chunk 개수:", len(chunks))

    return output_path

import sys
sys.path.append(str(BASE_DIR))
from utils.ingest_to_db import ingest

def main():
    global guide_files

    guide_files = list(GUIDE_DIR.glob("*.pdf"))

    print("guide PDF 개수:", len(guide_files))
    for f in guide_files:
        print("-", f.name)

    all_guide_chunks = []

    target_file = get_target_file("Welcome Guide")
    welcome_chunks = make_welcome_guide_chunks(target_file)
    save_chunks(welcome_chunks, "welcome_guide_chunks.json")
    all_guide_chunks.extend(welcome_chunks)

    target_file = get_target_file("BeHealthy SOB")
    if target_file is None:
        target_file = get_target_file("SAL-EU")
    sob_chunks = make_behealthy_sob_chunks(target_file)
    save_chunks(sob_chunks, "behealthy_sob_chunks.json")
    all_guide_chunks.extend(sob_chunks)

    target_file = get_target_file("Program Guide")
    program_chunks = make_program_guide_chunks(target_file)
    save_chunks(program_chunks, "program_guide_chunks.json")
    all_guide_chunks.extend(program_chunks)

    target_file = get_target_file("Business Travel Member FAQs")
    if target_file is None:
        target_file = get_target_file("MBR-BT")
    faq_chunks = make_business_travel_faq_chunks(target_file)
    save_chunks(faq_chunks, "business_travel_faq_chunks.json")
    all_guide_chunks.extend(faq_chunks)

    save_chunks(all_guide_chunks, "uhc_guide_all_chunks.json")
    ingest(all_guide_chunks)


if __name__ == "__main__":
    main()