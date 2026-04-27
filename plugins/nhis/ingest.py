# 원래 import 해야 했던 함수들 가져오기

# 공용 전처리 함수 — PDF 로딩, 청킹, Chroma 저장 (각 ingest.py가 import해서 사용)
import os
import uuid

# Mac 환경 오류 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

VECTORDB_PATH = "./vectordb"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)


# ──────────────────────────────────────────
# BAAI/bge-m3 임베딩 함수 (모든 보험사 공통)
# ──────────────────────────────────────────
class BGEM3EmbeddingFunction(EmbeddingFunction):
    """
    BAAI/bge-m3 다국어 임베딩 함수.
    모든 보험사 ingest.py에서 공통으로 사용.
    처음 실행 시 모델 다운로드 (~2GB), 이후 캐시 사용.
    """
    def __init__(self):
        print("BAAI/bge-m3 모델 로딩 중...")
        self.model = SentenceTransformer("BAAI/bge-m3")
        print("모델 로딩 완료")

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(
            input,
            normalize_embeddings=True,  # cosine similarity용 정규화
            show_progress_bar=False,
        )
        return embeddings.tolist()


def get_embedding_function() -> BGEM3EmbeddingFunction:
    """임베딩 함수 인스턴스 반환 (각 ingest.py에서 호출)"""
    return BGEM3EmbeddingFunction()


# ──────────────────────────────────────────
# 공통 유틸 함수
# ──────────────────────────────────────────
def load_pdf(data_dir: str) -> list[tuple[str, int, str]]:
    """
    디렉토리 안의 모든 PDF를 로딩.
    반환: [(파일명, 페이지번호, 텍스트), ...]
    """
    pages = []
    if not os.path.exists(data_dir):
        print(f"[경고] 데이터 디렉토리 없음: {data_dir}")
        return pages

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"[경고] PDF 파일 없음: {data_dir}")
        return pages

    for filename in pdf_files:
        print(f"  PDF 로딩: {filename}")
        loader = PyPDFLoader(os.path.join(data_dir, filename))
        for i, page in enumerate(loader.load()):
            pages.append((filename, i + 1, page.page_content))

    return pages


def chunk_text(text: str) -> list[str]:
    """텍스트를 청크 리스트로 분할 (빈 청크 제거)"""
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]


def save_to_collection(
    collection_name: str,
    chunks: list[str],
    metadatas: list[dict],
    embedding_fn: BGEM3EmbeddingFunction = None,
):
    """
    청크와 메타데이터를 Chroma 컬렉션에 저장.
    embedding_fn 지정 시 임베딩을 직접 계산 후 Chroma에 전달 (segfault 방지).
    """
    client = chromadb.PersistentClient(path=VECTORDB_PATH)
    # 임베딩 함수는 Chroma에 넘기지 않고 직접 계산 후 저장
    collection = client.get_or_create_collection(name=collection_name)

    BATCH_SIZE = 32  # Mac segfault 방지용 소형 배치
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks   = chunks[i:i + BATCH_SIZE]
        batch_meta     = metadatas[i:i + BATCH_SIZE]
        batch_ids      = [str(uuid.uuid4()) for _ in batch_chunks]

        if embedding_fn:
            # 임베딩 직접 계산 후 Chroma에 전달 (Chroma가 호출하지 않음)
            batch_embeddings = embedding_fn.model.encode(
                batch_chunks,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=16,          # 모델 내부 배치도 작게
            ).tolist()
            collection.add(
                ids=batch_ids,
                documents=batch_chunks,
                metadatas=batch_meta,
                embeddings=batch_embeddings,
            )
        else:
            collection.add(
                ids=batch_ids,
                documents=batch_chunks,
                metadatas=batch_meta,
            )

        print(f"  저장 진행: {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")



# NHIS 웹 크롤링 + PDF → 청킹 → DocumentMetadata 태깅 → nhis 컬렉션 저장
# 사용법: python -m plugins.nhis.ingest
#
# 데이터 소스
#   [웹] nhis.or.kr 영문 공식 페이지 (자격, 보험료, 급여범위, 고객지원)
#   [웹] hira.or.kr 본인부담률 기준 (한국어)
#   [PDF] data/nhis/ 디렉토리 안의 모든 PDF (NHIS 연간 보고서 등)

import os
import re
import time
from datetime import date

import requests
from bs4 import BeautifulSoup

# from utils.ingest_utils import (
#     chunk_text,
#     get_embedding_function,
#     load_pdf,
#     save_to_collection,
# )

COLLECTION_NAME = "nhis"
DATA_DIR = "./data/nhis"
CURRENT_YEAR = str(date.today().year)

# ──────────────────────────────────────────
# 크롤링 대상 웹페이지 목록
# ──────────────────────────────────────────
WEB_SOURCES = [
    {
        "url": "https://www.nhis.or.kr/english/wbheaa02900m01.do",
        "topic": "eligibility",        # 외국인 가입 자격
        "language": "en",
    },
    {
        "url": "https://www.nhis.or.kr/english/wbheaa02500m01.do",
        "topic": "contribution",       # 보험료율
        "language": "en",
    },
    {
        "url": "https://www.nhis.or.kr/english/wbheaa02600m01.do",
        "topic": "benefits",           # 급여 범위
        "language": "en",
    },
    {
        "url": "https://www.nhis.or.kr/english/wbheaa02800m01.do",
        "topic": "customer_support",   # 외국인 고객 지원
        "language": "en",
    },
    {
        "url": "https://www.hira.or.kr/dummy.do?pgmid=HIRAA030056020100",
        "topic": "copay",              # 본인부담률 기준
        "language": "ko",
    },
    {
        "url": "https://www.hira.or.kr/dummy.do?pgmid=HIRAA030056020110",
        "topic": "copay_outpatient",   # 외래진료 본인부담률
        "language": "ko",
    },
]


# ──────────────────────────────────────────
# 웹 크롤링
# ──────────────────────────────────────────
def fetch_html(url: str) -> str:
    """HTML 페이지에서 본문 텍스트만 추출 (nav, footer, script 제거)"""
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
    except requests.RequestException as e:
        print(f"  [오류] 크롤링 실패 ({url}): {e}")
        return ""

    soup = BeautifulSoup(res.text, "html.parser")

    # 노이즈 태그 제거
    for tag in soup(["nav", "footer", "header", "script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # 연속 빈 줄 정리
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def ingest_web() -> tuple[list, list]:
    """웹페이지 크롤링 후 청킹 + 메타데이터 반환"""
    all_chunks = []
    all_metadatas = []

    print("\n[웹] 크롤링 시작")
    for source in WEB_SOURCES:
        print(f"  크롤링: {source['topic']} ({source['url']})")
        text = fetch_html(source["url"])

        if not text:
            print(f"  [건너뜀] 텍스트 없음: {source['topic']}")
            continue

        chunks = chunk_text(text)
        print(f"  → {len(chunks)}개 청크 생성")

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({
                "insurer":      "nhis",
                "source_type":  "web",
                "file_name":    source["topic"],   # 토픽명을 파일명 자리에 사용
                "page":         0,                  # 웹은 페이지 번호 없음
                "year":         CURRENT_YEAR,
                "plan":         "",                 # NHIS는 플랜 구분 없음
                "language":     source["language"],
                "url":          source["url"],
                "topic":        source["topic"],
            })

        time.sleep(1)  # 서버 부하 방지

    print(f"[웹] 총 {len(all_chunks)}개 청크 완료\n")
    return all_chunks, all_metadatas


# ──────────────────────────────────────────
# PDF 처리
# ──────────────────────────────────────────
def _extract_year_from_filename(filename: str) -> str:
    """파일명에서 연도 추출 (예: nhis_2024_report.pdf → '2024')"""
    match = re.search(r"(20\d{2})", filename)
    return match.group(1) if match else CURRENT_YEAR


def ingest_pdf() -> tuple[list, list]:
    """data/nhis/ 의 PDF 파일 로딩 후 청킹 + 메타데이터 반환"""
    all_chunks = []
    all_metadatas = []

    print("[PDF] 처리 시작")
    pages = load_pdf(DATA_DIR)

    if not pages:
        print("[PDF] 처리할 파일 없음. data/nhis/ 에 PDF를 넣어주세요.\n")
        return all_chunks, all_metadatas

    for filename, page_num, text in pages:
        if not text.strip():
            continue

        chunks = chunk_text(text)
        year = _extract_year_from_filename(filename)

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({
                "insurer":      "nhis",
                "source_type":  "pdf",
                "file_name":    filename,
                "page":         page_num,
                "year":         year,
                "plan":         "",
                "language":     "en",           # NHIS 공식 PDF는 영문 기준
                "url":          "",
                "topic":        "annual_report",
            })

    print(f"[PDF] 총 {len(all_chunks)}개 청크 완료\n")
    return all_chunks, all_metadatas


# ──────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────
def run():
    print("=" * 50)
    print("NHIS ingest 시작")
    print("=" * 50)

    # 임베딩 함수 로딩 (BAAI/bge-m3)
    embedding_fn = get_embedding_function()

    # 웹 + PDF 데이터 수집
    web_chunks, web_metas = ingest_web()
    pdf_chunks, pdf_metas = ingest_pdf()

    all_chunks = web_chunks + pdf_chunks
    all_metas = web_metas + pdf_metas

    if not all_chunks:
        print("[오류] 저장할 데이터가 없습니다.")
        return

    print(f"총 {len(all_chunks)}개 청크를 '{COLLECTION_NAME}' 컬렉션에 저장 중...")
    save_to_collection(COLLECTION_NAME, all_chunks, all_metas, embedding_fn)

    print("=" * 50)
    print(f"NHIS ingest 완료: {len(web_chunks)}개 (웹) + {len(pdf_chunks)}개 (PDF)")
    print("=" * 50)


if __name__ == "__main__":
    run()
