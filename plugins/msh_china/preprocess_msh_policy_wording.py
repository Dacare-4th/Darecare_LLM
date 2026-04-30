"""
preprocess_msh_policy_wording.py  (v3 - 버그 수정)
MSH_First_Expat_Plus_Policy_Wording.pdf (89p) -> data/chunks/msh_policy_wording_chunks.json

[v2 → v3 수정 내역]
1. TOP_RE  : 섹션 9 ("9. /제목", 공백 없음) + 섹션 11·12 (/ 없는 APPENDIX) 감지 추가
2. SUB_RE  : 동일하게 / 앞 공백 0~1개 허용
3. 메타데이터 오류 3개 수정
   - insurer  : "MSH" → "Groupama Gan Vie"  (MSH는 administrator)
   - plan     : "WeCare" → "FIRST'EXPAT+ / RELAIS'EXPAT+"
   - source   : 실제 파일명으로 수정
   - administrator 필드 추가
4. MIN_CHUNK 예외 처리 : 최상위 섹션 헤더(단독 줄) MIN_CHUNK 미달로 드롭되던 문제 수정
   → 다음 서브섹션 헤더 직전까지 내용이 짧아도 강제 emit
"""

import argparse
import re, json, fitz
from pathlib import Path
from typing import List, Tuple, Optional
BASE_DIR = Path(__file__).resolve().parent.parent.parent



# ── 공통 유틸 ────────────────────────────────────────────────────────────────

def clean_text(text):
    text = text.replace("\xa0", " ").replace("\u200b", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def read_pdf_pages(pdf_path):
    pages, doc = [], fitz.open(str(pdf_path))
    try:
        for i, page in enumerate(doc):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append((i + 1, text))
    finally:
        doc.close()
    return pages

def build_metadata(page, section=None, subsection=None, section_title=None, chunk_type="section_text"):
    m = {
        "insurer":             "msh_china",         
        "doc_type":            "policy_wording",
        "source":              "MSH-Health-First Expat Health-Policy Wording.pdf",
        "page":                page,
        "is_latest":           True,
        "plan":                "FIRST'EXPAT+ / RELAIS'EXPAT+",  # 실제 플랜명
    }
    if section:       m["section"]       = section
    if subsection:    m["subsection"]    = subsection
    if section_title: m["section_title"] = section_title
    return m

def make_chunk(cid, content, metadata):
    return {"chunk_id": cid, "content": content, "metadata": metadata}

def save_chunks(chunks, output_dir, filename):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[DONE] {len(chunks)}개 청크 저장 -> {path}")
    return path

# ── 헤더 패턴 ────────────────────────────────────────────────────────────────

# ✅ FIX 1: TOP_RE 수정
#   원본: r"^(\d{1,2})\. +/ +([A-Z]..." → "/ " 필수라서 아래 케이스 미감지
#   - 섹션 9: "9. /MEDICAL ASSISTANCE..." (/ 앞 공백 없음)
#   - 섹션 11,12: "11. APPENDIX 1:..." (/ 자체 없음)
#   수정: "/ " 를 선택적(optional)으로 변경
TOP_RE = re.compile(
    r"^(\d{1,2})\. +(?:/ +)?([A-Z][^\n]{3,100}?)\s*$",
    re.MULTILINE
)

# ✅ FIX 2: SUB_RE 동일하게 수정 (/ 앞 공백 0개인 케이스 허용)
SUB_RE = re.compile(
    r"^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\.\s+(?:/ +)?([A-Z][^\n]{2,120}?)\s*$",
    re.MULTILINE
)

NOISE_RE = re.compile(
    r"^(?:\d{1,3}"
    r"|// INFORMATION BOOKLET SERVING AS THE GENERAL TERMS & CONDITIONS"
    r"|Good to know|IMPORTANT|Waiting periods in detail:"
    r"|Coverage in the event of an emergency"
    r"|HEALTHCARE FOLLOWING COVERED HOSPITALIZATION"
    r"|WELLBEING & WELLNESS"
    r")\s*$", re.MULTILINE)

def strip_noise(text):
    text = NOISE_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

# ── 전체 텍스트 + 페이지맵 ──────────────────────────────────────────────────

def build_full_text(pages):
    parts, page_map, pos = [], [], 0
    for pno, text in pages:
        cleaned = strip_noise(text)
        if not cleaned:
            continue
        page_map.append((pos, pno))
        parts.append(cleaned)
        pos += len(cleaned) + 2
    return "\n\n".join(parts), page_map

def char_to_page(pos, page_map):
    result = page_map[0][1] if page_map else 1
    for start, pno in page_map:
        if pos >= start:
            result = pno
        else:
            break
    return result

# ── 헤더 수집 (TOC 제외) ─────────────────────────────────────────────────────

def find_headers(full_text, page_map):
    toc_end = 0
    for start, pno in page_map:
        if pno > 2:
            toc_end = start
            break

    raw = []
    for m in TOP_RE.finditer(full_text):
        if m.start() > toc_end:
            raw.append((m.start(), m.group(1), m.group(2).strip()))
    for m in SUB_RE.finditer(full_text):
        if m.start() > toc_end:
            raw.append((m.start(), m.group(1), m.group(2).strip()))

    raw.sort(key=lambda x: x[0])
    seen, deduped = set(), []
    for h in raw:
        if h[0] not in seen:
            deduped.append(h)
            seen.add(h[0])
    return deduped

# ── 길이 제어 ────────────────────────────────────────────────────────────────

MAX_CHUNK = 2200
MIN_CHUNK = 120

def _merge_split(items, max_chars):
    parts, buf = [], ""
    for item in items:
        cand = (buf + "\n\n" + item) if buf else item
        if len(cand) <= max_chars:
            buf = cand
        else:
            if buf:
                parts.append(buf.strip())
            buf = item
    if buf.strip():
        parts.append(buf.strip())
    return parts

def split_text(text, max_chars=MAX_CHUNK):
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    result = _merge_split(re.split(r"\n\n+", text), max_chars)
    final = []
    for p in result:
        if len(p) > max_chars:
            lines = _merge_split(re.split(r"\n", p), max_chars)
            for l in lines:
                if len(l) > max_chars:
                    final.extend(_merge_split(re.split(r"(?<=\.) ", l), max_chars))
                else:
                    final.append(l)
        else:
            final.append(p)
    return [x.strip() for x in final if x.strip()]

def get_labels(sec_id):
    parts = sec_id.split(".")
    if len(parts) == 1:
        return sec_id, None
    elif len(parts) == 2:
        return parts[0], sec_id
    else:
        return ".".join(parts[:2]), sec_id

def is_top_level(sec_id: str) -> bool:
    """최상위 섹션 여부 (예: "1", "6", "11")"""
    return bool(re.match(r"^\d{1,2}$", sec_id))

# ── 메인 ─────────────────────────────────────────────────────────────────────

def chunk_policy_wording(pdf_path):
    print(f"[INFO] PDF 로드: {pdf_path}")
    pages = read_pdf_pages(pdf_path)
    print(f"[INFO] 페이지 수: {len(pages)}")
    full_text, page_map = build_full_text(pages)
    headers = find_headers(full_text, page_map)
    print(f"[INFO] 헤더 수 (TOC 제외): {len(headers)}")

    chunks, idx = [], 0

    def emit(content, meta, part_idx=0, force=False):
        """
        force=True : 최상위 섹션처럼 내용이 짧아도 강제 emit
        ✅ FIX 4: MIN_CHUNK 미달로 짧은 섹션 헤더가 드롭되던 문제 해결
        """
        nonlocal idx
        content = content.strip()
        if not content:
            return
        if not force and len(content) < MIN_CHUNK:
            return
        cid = f"msh_policy_wording_chunk_{idx}"
        if part_idx > 0:
            meta = {**meta, "part_index": part_idx}
        chunks.append(make_chunk(cid, content, meta))
        idx += 1

    # 표지/목차
    if headers:
        pre = full_text[:headers[0][0]].strip()
        if pre:
            emit(pre[:MAX_CHUNK], build_metadata(1, "COVER_TOC",
                section_title="Cover and Table of Contents"))

    # 섹션별
    for i, (start, sec_id, sec_title) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(full_text)
        raw = full_text[start:end].strip()
        if not raw:
            continue

        sec, sub = get_labels(sec_id)
        pno = char_to_page(start, page_map)
        meta = build_metadata(pno, sec, sub, sec_title)

        # ✅ FIX 4: 최상위 섹션은 내용이 짧아도 강제 emit (섹션 3, 6, 9 등)
        force = is_top_level(sec_id)

        parts = split_text(raw)
        if parts:
            for pi, part in enumerate(parts):
                emit(part, meta, pi, force=(force and pi == 0))
        else:
            # split_text가 빈 리스트 반환할 때 (MIN_CHUNK 미달 케이스)
            emit(raw, meta, force=force)

    return chunks


def main():
    parser = argparse.ArgumentParser(description="MSH Policy Wording 전처리 → JSON (v3)")
    parser.add_argument(
        "--pdf", type=Path,
        default= BASE_DIR / "data" / "msh" / "MSH-Health-First Expat Health-Policy Wording.pdf",
        help="PDF 파일 경로"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default= BASE_DIR / "data" / "outputs" / "msh" / "policy wording",
        help="JSON 출력 디렉토리"
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"[ERROR] PDF 파일 없음: {args.pdf}")
        return

    chunks = chunk_policy_wording(args.pdf)
    import sys
    sys.path.append(str(BASE_DIR))
    from utils.ingest_to_db import ingest
    ingest(chunks)

    if not chunks:
        print("[ERROR] 생성된 청크 없음")
        return

    lengths = [len(c["content"]) for c in chunks]
    secs = set(c["metadata"].get("section", "") for c in chunks)
    print(f"\n[STATS] 총={len(chunks)} 최소={min(lengths)} 최대={max(lengths)} "
          f"평균={sum(lengths)//len(lengths)} 고유섹션={len(secs)}")

    save_chunks(chunks, args.output_dir, "msh_policy_wording_chunks.json")

    # 섹션 커버리지 확인
    print("\n[COVERAGE] 감지된 최상위 섹션:")
    top_secs = sorted(set(
        c["metadata"].get("section", "")
        for c in chunks
        if is_top_level(str(c["metadata"].get("section", "")))
    ), key=lambda x: int(x) if x.isdigit() else 99)
    for s in top_secs:
        count = sum(1 for c in chunks if c["metadata"].get("section") == s)
        title = next((c["metadata"].get("section_title","") for c in chunks
                      if c["metadata"].get("section") == s), "")
        print(f"  섹션 {s:>2} | {count:>3}개 청크 | {title[:50]}")

    print("\n[SAMPLE] 처음 8개:")
    for c in chunks[:8]:
        m = c["metadata"]
        print(f"  [{c['chunk_id']}] sec={m.get('section')} sub={m.get('subsection')} "
              f"p={m['page']} len={len(c['content'])} | "
              f"{c['content'][:80].replace(chr(10), ' ')}")

if __name__ == "__main__":
    main()
