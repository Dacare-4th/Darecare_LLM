# NHIS 웹 크롤링 → 청킹 → DocumentMetadata 태깅 → nhis 컬렉션 저장
import requests
from bs4 import BeautifulSoup
from utils.ingest_utils import chunk_text, save_to_collection
from utils.schemas import DocumentMetadata
from datetime import date

COLLECTION_NAME = "nhis"

# 크롤링 대상 URL 목록
NHIS_SOURCES = [
    {"url": "https://www.nhis.or.kr/english/wbheaa02900m01.do", "topic": "eligibility",   "language": "en"},
    {"url": "https://www.nhis.or.kr/english/wbheaa02500m01.do", "topic": "contribution",  "language": "en"},
    {"url": "https://www.nhis.or.kr/english/wbheaa02600m01.do", "topic": "benefits",      "language": "en"},
    {"url": "https://www.hira.or.kr/dummy.do?pgmid=HIRAA030056020100", "topic": "copay",  "language": "ko"},
]


def fetch_html(url: str) -> str:
    res = requests.get(url, headers={"Accept-Language": "ko-KR,ko"}, timeout=10)
    soup = BeautifulSoup(res.text, "html.parser")
    for tag in soup(["nav", "footer", "script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def run():
    today = str(date.today())
    all_chunks = []
    all_metadatas = []

    for source in NHIS_SOURCES:
        print(f"크롤링 중: {source['url']}")
        text = fetch_html(source["url"])

        for chunk in chunk_text(text):
            all_chunks.append(chunk)
            all_metadatas.append(DocumentMetadata(
                insurer="nhis",
                source_type="web",
                file_name=source["topic"],
                page=0,
                year=today[:4],
                plan="",
                language=source["language"],
            ))

    save_to_collection(COLLECTION_NAME, all_chunks, all_metadatas)
    print(f"[nhis] {len(all_chunks)}개 청크 저장 완료")


if __name__ == "__main__":
    run()
