# NHIS 전처리 — 웹 크롤링 + PDF 파싱 → 텍스트 청킹 + 표 JSON 추출
# 벡터DB 저장은 하지 않음. 결과는 data/nhis/processed/ 에 JSON으로 저장.
#
# 사용법: python -m plugins.nhis.ingest
#
# 출력 파일:
#   data/nhis/processed/web_{topic}.json
#   data/nhis/processed/pdf_{year}_{doc_type}.json

import io
import json
import os
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────

# 여기랑 밑에 save_json() 함수 바꾸면 됨!!!
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR   = BASE_DIR / "data/output/nhis"

CURRENT_YEAR = str(date.today().year)
INSURER      = "nhis"
IS_LATEST    = True

# pandas.read_html() 에서 사용할 HTML 파서
# 진단 결과에 따라 "lxml" 또는 "html5lib" 으로 변경
# 설치: pip install lxml  또는  pip install html5lib
FLAVOR = "lxml"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)

# ──────────────────────────────────────────
# PDF 소스 목록
# ──────────────────────────────────────────
PDF_SOURCES = [
    {
        "path":      "./data/nhis/booklet/2024_Booklet_for_the_Introduction_of_National_Health_Insurance_System.pdf",
        "year":      "2024",
        "language":  "en",
        "is_latest": True,
        "doc_type":  "booklet",
        "topic":     "booklet",
        "audience":  "foreigner_only",
    },
]

# 좌·우 병렬 레이아웃 페이지 등록
# dict 값: 헤더를 명시할 경우 list, 자동 감지로 충분하면 None
# pdfplumber 는 배경색 헤더 행을 누락하는 경우가 있어 명시 지정이 안전함
SIDE_BY_SIDE_PAGES = {
    77: ["Service time", "Rating", "General", "Dementia Unit"],
    # 추가 페이지: page_num: [헤더 리스트] or None
}

# ──────────────────────────────────────────
# 크롤링 대상 웹페이지 목록
# ──────────────────────────────────────────
WEB_SOURCES = [
    # 자격/가입
    {"url": "https://www.nhis.or.kr/english/wbheaa02400m01.do",  "topic": "population_coverage",   "language": "en", "audience": "foreigner_only"},
    {"url": "https://www.nhis.or.kr/english/wbheaa02900m01.do",  "topic": "eligibility_foreigners", "language": "en", "audience": "foreigner_only"},
    {"url": "https://www.nhis.or.kr/lm/lmxsrv/law/lawDetail.do?SEQ=41&LAWGROUP=1", "topic": "eligibility_regulation", "language": "ko", "audience": "foreigner_only"},
    # 보험료
    {"url": "https://www.nhis.or.kr/english/wbheaa02500m01.do",  "topic": "contribution_rate",      "language": "en", "audience": "foreigner_only"},
    {"url": "https://easylaw.go.kr/CSP/CnpClsMain.laf?popMenu=ov&csmSeq=1063&ccfNo=4&cciNo=1&cnpClsNo=1", "topic": "contribution_employee", "language": "ko", "audience": "foreigner_only"},
    # contribution_regional 제거 — 지역가입자 점수제 산정 방식은 외국인에게 적용되지 않음
    # (외국인은 전년도 직장가입자 평균 보험료 기준으로 부과)
    # 급여 범위
    {"url": "https://www.nhis.or.kr/english/wbheaa02600m01.do",  "topic": "benefits",               "language": "en", "audience": "foreigner_only"},
    # 본인부담금
    {"url": "https://www.hira.or.kr/dummy.do?pgmid=HIRAA030056020100", "topic": "copay_standard",   "language": "ko", "audience": "foreigner_only"},
    {"url": "https://www.hira.or.kr/dummy.do?pgmid=HIRAA030056020110", "topic": "copay_outpatient", "language": "ko", "audience": "foreigner_only"},
    # 고객지원
    {"url": "https://www.nhis.or.kr/english/wbheaa02800m01.do",  "topic": "customer_support",       "language": "en", "audience": "foreigner_only"},
    # private_claim_docs 제거 — 국내 민간 손해보험 내국인 대상 페이지로 외국인 챗봇 범위 외
]


# ──────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]

# 여기 수정하면 됨!!!
def save_json(data, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  저장: {path}")


# ──────────────────────────────────────────
# PDF 표 전처리 헬퍼 함수
# ──────────────────────────────────────────

_BULLET_MAP = {"●": "Y", "•": "Y", "○": "Y(pilot)"}


def _clean_cell(cell) -> str:
    """셀 원시값 → 문자열 정제 (줄바꿈 제거, 공백 정리)."""
    if cell is None:
        return ""
    return str(cell).replace("\n", " ").strip()


def _map_bullet(text: str) -> str:
    """불릿 문자(●/○)를 의미있는 텍스트로 변환."""
    return _BULLET_MAP.get(text.strip(), text)


def _ffill_cols(table: list) -> list:
    """각 열의 None·빈값을 위쪽 값으로 채운다 (rowspan 병합 셀 처리).

    pdfplumber 는 rowspan 병합 셀을 첫 행에만 값을 채우고
    나머지는 None 으로 반환한다. forward-fill 로 복원한다.
    """
    if not table:
        return table
    n_cols = max(len(row) for row in table)
    result = [list(row) + [""] * (n_cols - len(row)) for row in table]
    for col in range(n_cols):
        last = ""
        for row in result:
            val = str(row[col]).strip() if row[col] is not None else ""
            if val:
                last = val
                row[col] = val
            else:
                row[col] = last
    return result


def _detect_header_rows(table: list) -> int:
    """헤더가 몇 줄인지 탐지한다 (최대 2줄).

    첫 번째 행에 빈 셀이 30% 이상이고
    두 번째 행에 숫자가 없으면 2줄 헤더로 판단한다.
    """
    if not table or len(table) < 3:
        return 1
    first  = [str(c or "").strip() for c in table[0]]
    second = [str(c or "").strip() for c in table[1]]
    empty_ratio    = sum(1 for c in first if not c) / max(len(first), 1)
    second_has_num = any(re.search(r"\d{2,}", c) for c in second)
    if empty_ratio >= 0.3 and not second_has_num:
        return 2
    return 1


def _build_headers(filled_table: list, n_hdr: int) -> list:
    """헤더 행을 합쳐 단일 헤더 리스트를 만든다.

    2줄 헤더의 경우 상위·하위 헤더를 ' > ' 로 연결한다.
    예) 'Per household(KRW)' + 'Self-employed'
        → 'Per household(KRW) > Self-employed'
    """
    if n_hdr == 1:
        return [_clean_cell(c) for c in filled_table[0]]
    hdr_filled = _ffill_cols(filled_table[:n_hdr])
    n_cols = len(hdr_filled[0])
    headers = []
    for col in range(n_cols):
        parts = []
        for row in hdr_filled:
            val = _clean_cell(row[col]) if col < len(row) else ""
            if val and val not in parts:
                parts.append(val)
        headers.append(" > ".join(parts) if len(parts) > 1 else (parts[0] if parts else ""))
    return headers


def _is_side_by_side(table: list) -> bool:
    """헤더 첫 행이 좌·우로 동일하게 반복되면 병렬 레이아웃으로 판단."""
    if not table or len(table[0]) < 6:
        return False
    n = len(table[0])
    half = n // 2
    first = [str(c or "").strip() for c in table[0]]
    return first[:half] == first[half:]


def _table_to_md(raw_table: list, forced_headers: list = None) -> str:
    """pdfplumber 원시 표 → 마크다운 테이블 문자열.

    Args:
        raw_table:      pdfplumber 가 반환한 원시 표 (list of list)
        forced_headers: 헤더 명시 지정 (pdfplumber 가 배경색 헤더를 누락할 때).
                        지정 시 모든 행을 데이터 행으로 처리한다.
    """
    if not raw_table or len(raw_table) < 1:
        return ""
    filled = _ffill_cols(raw_table)

    if forced_headers:
        headers   = forced_headers
        data_rows = filled
    else:
        n_hdr     = _detect_header_rows(filled)
        headers   = _build_headers(filled, n_hdr)
        data_rows = filled[n_hdr:]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in data_rows:
        cleaned = [_map_bullet(_clean_cell(c)) for c in row]
        if any(c.strip() for c in cleaned):
            lines.append("| " + " | ".join(cleaned) + " |")

    return "\n".join(lines) if len(lines) > 2 else ""


def _page_table_mds(page, page_num: int) -> list[str]:
    """페이지에서 마크다운 테이블 문자열 목록을 반환.

    SIDE_BY_SIDE_PAGES 에 헤더가 명시된 페이지는 forced_headers 로 전달해
    pdfplumber 누락 헤더를 보완한다.
    """
    raw_tables = page.extract_tables() or []
    mds = []
    forced_headers = SIDE_BY_SIDE_PAGES.get(page_num)

    for table in raw_tables:
        if not table or len(table) < 2:
            continue
        if page_num in SIDE_BY_SIDE_PAGES or _is_side_by_side(table):
            half = len(table[0]) // 2
            for half_tbl in (
                [row[:half] for row in table],
                [row[half:] for row in table],
            ):
                md = _table_to_md(half_tbl, forced_headers=forced_headers)
                if md:
                    mds.append(md)
        else:
            md = _table_to_md(table)
            if md:
                mds.append(md)
    return mds


# ──────────────────────────────────────────
# 웹 크롤링
# ──────────────────────────────────────────

def fetch_page(url: str) -> tuple[str, BeautifulSoup] | tuple[None, None]:
    """URL 에서 HTML 원문과 BeautifulSoup 객체를 함께 반환한다.
    표 추출(pandas)에는 원문 HTML이, 텍스트 추출에는 soup 가 필요하다."""
    try:
        res = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
            },
            timeout=15,
        )
        res.raise_for_status()
        return res.text, BeautifulSoup(res.text, "html.parser")
    except requests.RequestException as e:
        print(f"  [오류] 크롤링 실패: {e}")
        return None, None


def extract_web_text(soup: BeautifulSoup) -> str:
    """네비게이션·푸터·스크립트 등 노이즈 태그를 제거한 뒤 본문 텍스트를 반환한다."""
    for tag in soup(["nav", "footer", "header", "script", "style", "noscript"]):
        tag.decompose()
    lines = [l for l in soup.get_text(separator="\n", strip=True).splitlines() if l.strip()]
    return "\n".join(lines)


def extract_web_tables(html_text: str) -> list[dict]:
    """pandas.read_html() 로 HTML 표를 추출한다.

    BeautifulSoup 방식과 달리 rowspan/colspan 을 자동으로 처리한다.
    MultiIndex 컬럼(2중 헤더)은 ' > ' 로 이어붙여 단일 문자열로 평탄화한다.

    반환 형식:
        [{"table_index": int, "headers": list, "rows": list[list]}, ...]
    """
    try:
        dfs = pd.read_html(io.StringIO(html_text), flavor=FLAVOR)
    except ValueError:
        # 표가 없는 페이지는 빈 리스트 반환
        return []
    except Exception as e:
        print(f"  [표 추출 오류] {type(e).__name__}: {e}")
        return []

    results = []
    for i, df in enumerate(dfs):
        df = df.fillna("")
        # 2중 헤더(MultiIndex) → "상위 > 하위" 형태로 평탄화
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" > ".join(str(c) for c in col).strip() for col in df.columns]
        results.append({
            "table_index": i,
            "headers":     list(df.columns),
            "rows":        df.values.tolist(),
        })
    return results


def ingest_web() -> list[dict]:
    """WEB_SOURCES 의 모든 URL 을 크롤링하여 청크와 표 JSON 을 추출하고 저장한다."""
    print("\n[웹] 크롤링 시작")
    all_results = []

    for source in WEB_SOURCES:
        topic    = source["topic"]
        language = source["language"]
        url      = source["url"]
        audience = source.get("audience", "all")

        print(f"\n  topic: {topic}")
        print(f"  url:   {url}")

        html_text, soup = fetch_page(url)
        if soup is None:
            continue

        raw_text = extract_web_text(soup)
        raw_chunks = chunk_text(raw_text)
        tables = extract_web_tables(html_text)

        print(f"  → 청크 {len(raw_chunks)}개 / 표 {len(tables)}개")

        # 청크를 공통 메타데이터 형식으로 포장
        chunks = []
        for idx, text in enumerate(raw_chunks, start=1):
            chunks.append({
                "chunk_id": f"nhis_web_{topic}_{idx}",
                "text": text,
                "metadata": {
                    "insurer":     INSURER,
                    "doc_type":    "web_page",
                    "source":      url,
                    "source_type": "web",
                    "topic":       topic,
                    "page":        None,
                    "is_latest":   IS_LATEST,
                    "language":    language,
                    "plan":        None,
                    "year":        CURRENT_YEAR,
                    "chunk_type":  "page_text",
                    "audience":    audience,  # 이 레포의 모든 웹 소스는 "foreigner_only"
                },
            })

        result = {
            "metadata": {
                "insurer":     INSURER,
                "source_type": "web",
                "topic":       topic,
                "url":         url,
                "language":    language,
                "year":        CURRENT_YEAR,
            },
            "chunks": chunks,
            "tables": tables,
        }
        save_json(result, f"web_{topic}.json")
        all_results.append(result)
        time.sleep(1)  # 서버 부하 방지

    print("\n[웹] 완료")
    return all_results


# ──────────────────────────────────────────
# PDF 전처리 (B안: 표 분리)
# ──────────────────────────────────────────

def ingest_pdf() -> list[dict]:
    """PDF_SOURCES 의 모든 PDF 를 파싱하여 텍스트 청크와 표 마크다운을 추출하고 저장한다.

    출력 형식 (B안 — 표 분리):
        chunks: 텍스트만 포함
        tables: [{"table_index": int, "page": int, "markdown": str}, ...]
    """
    print("\n[PDF] 전처리 시작")
    all_results = []

    for source in PDF_SOURCES:
        path      = source["path"]
        year      = source["year"]
        language  = source["language"]
        is_latest = source["is_latest"]
        doc_type  = source["doc_type"]
        topic     = source["topic"]
        audience  = source["audience"]
        filename  = Path(path).name

        print(f"\n  doc_type: {doc_type} / year: {year}")
        print(f"  path:     {path}")

        if not Path(path).exists():
            print(f"  [오류] 파일을 찾을 수 없습니다: {path}")
            continue

        all_chunks = []
        all_tables = []

        with pdfplumber.open(path) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                raw_text  = (page.extract_text() or "").strip()
                table_mds = _page_table_mds(page, page_num)

                # 텍스트 청킹
                if raw_text:
                    for idx, text_val in enumerate(chunk_text(raw_text), start=1):
                        all_chunks.append({
                            "chunk_id": f"nhis_pdf_{year}_p{page_num}_{idx}",
                            "text": text_val,
                            "metadata": {
                                "insurer":     INSURER,
                                "doc_type":    doc_type,
                                "source":      filename,
                                "source_type": "pdf",
                                "topic":       topic,
                                "page":        page_num,
                                "is_latest":   is_latest,
                                "language":    language,
                                "plan":        None,
                                "year":        year,
                                "chunk_type":  "page_text",
                                "audience":    audience,
                            },
                        })

                # 표 별도 저장 (마크다운)
                for t_idx, md in enumerate(table_mds):
                    all_tables.append({
                        "table_index": t_idx,
                        "page":        page_num,
                        "markdown":    md,
                    })

        print(f"  → 총 {total_pages}페이지 / 청크 {len(all_chunks)}개 / 표 {len(all_tables)}개")

        result = {
            "metadata": {
                "insurer":     INSURER,
                "source_type": "pdf",
                "doc_type":    doc_type,
                "source":      filename,
                "language":    language,
                "year":        year,
                "is_latest":   is_latest,
            },
            "chunks": all_chunks,
            "tables": all_tables,
        }
        save_json(result, f"pdf_{year}_{doc_type}.json")
        all_results.append(result)

    print("\n[PDF] 완료")
    return all_results


# ──────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.ingest_to_db import ingest


def run():

    print("=" * 50)
    print("NHIS 전처리 시작 (웹 + PDF)")
    print(f"출력 위치: {OUTPUT_DIR}")
    print("=" * 50)

    web_chunks = [c for r in ingest_web() for c in r["chunks"]]
    pdf_chunks = [c for r in ingest_pdf() for c in r["chunks"]]
    ingest(web_chunks + pdf_chunks)

    output_files = list(Path(OUTPUT_DIR).glob("*.json"))
    print("\n" + "=" * 50)
    print(f"완료: {len(output_files)}개 파일 생성됨")
    for f in sorted(output_files):
        size = f.stat().st_size // 1024
        print(f"  {f.name} ({size}KB)")
    print("=" * 50)


if __name__ == "__main__":
    run()
