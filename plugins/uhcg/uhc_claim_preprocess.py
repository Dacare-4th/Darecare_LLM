from pathlib import Path
import json
import re
import fitz

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLAIM_DIR = BASE_DIR / "data" / "uhc" / "claim"
OUTPUT_DIR = BASE_DIR / "data" / "output" / "uhcg" / "uhc_claim_form"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INSURER = "UnitedHealthcare Global"
DOCUMENT_TYPE = "claim_form"
IS_LATEST = True


def load_pdf_pages_ordered(file_path):
    doc = fitz.open(str(file_path))
    pages = []

    for page_idx, page in enumerate(doc):
        blocks = page.get_text("blocks")

        text_blocks = []
        for block in blocks:
            x0, y0, x1, y1, text, *_ = block

            if text.strip():
                text_blocks.append({
                    "x0": x0,
                    "y0": y0,
                    "text": text.strip()
                })

        text_blocks = sorted(
            text_blocks,
            key=lambda b: (round(b["y0"], 1), round(b["x0"], 1))
        )

        page_text = "\n".join(block["text"] for block in text_blocks)

        pages.append({
            "page": page_idx + 1,
            "text": page_text
        })

    return pages


def clean_claim_text(text):
    text = text.replace("\x0c", " ")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"continued", "", text, flags=re.IGNORECASE)
    return text.strip()


def load_full_text(file_path):
    pages = load_pdf_pages_ordered(file_path)
    return "\n".join([p["text"] for p in pages])


def split_claim_sections(text):
    matches = re.split(r"(section\s*[123].*?)", text, flags=re.IGNORECASE)

    sections = []

    for i in range(1, len(matches), 2):
        title = matches[i]
        content = matches[i + 1] if i + 1 < len(matches) else ""

        sections.append({
            "title": title.strip(),
            "content": content.strip()
        })

    return sections


def get_section_number(title):
    match = re.search(r"section\s*([123])", title, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 99


def make_claim_section_chunks(file_path):
    full_text = load_full_text(file_path)
    sections = split_claim_sections(full_text)
    sections = sorted(sections, key=lambda x: get_section_number(x["title"]))

    chunks = []

    for section in sections:
        title = section["title"].strip()
        content = clean_claim_text(section["content"])

        if len(content) < 20:
            continue

        section_num = get_section_number(title)

        chunks.append({
            "chunk_id": f"{file_path.stem}_section_{section_num}",
            "text": f"{title}\n{content}",
            "metadata": {
                "insurer": INSURER,
                "doc_type": DOCUMENT_TYPE,
                "source": file_path.name,
                "page": None,
                "is_latest": IS_LATEST,
                "language": "en",
                "plan": None,
                "source_category": "claim",
                "section_number": section_num,
                "section_title": title,
                "chunk_type": "claim_form_section",
                "region_scope": "global",
                "korea_applicability": "conditional",
                "claim_deadline_days": 365,
                "requires_invoice_or_receipt": True
            }
        })

    return chunks


def main():
    claim_files = list(CLAIM_DIR.glob("*.pdf"))

    all_claim_chunks = []

    for file_path in claim_files:
        all_claim_chunks.extend(make_claim_section_chunks(file_path))

    output_path = OUTPUT_DIR / "uhc_claim_section_chunks.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_claim_chunks, f, ensure_ascii=False, indent=2)

    print("claim PDF 개수:", len(claim_files))
    print("claim chunk 개수:", len(all_claim_chunks))
    print("저장 경로:", output_path)


if __name__ == "__main__":
    main()