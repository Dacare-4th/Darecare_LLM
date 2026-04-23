# uhcg PDF → 청킹 → DocumentMetadata 태깅 → uhcg 컬렉션 저장
from utils.ingest_utils import load_pdf, chunk_text, save_to_collection
from utils.schemas import DocumentMetadata

DATA_DIR = "./data/uhcg"
COLLECTION_NAME = "uhcg"


def run():
    # 1. PDF 로딩
    pages = load_pdf(DATA_DIR)

    # 2. 청킹 + 메타데이터 태깅
    chunks = []
    metadatas = []
    for page_num, text in pages:
        for chunk in chunk_text(text):
            chunks.append(chunk)
            metadatas.append(DocumentMetadata(
                insurer="uhcg",
                source_type="pdf",
                file_name="",       # TODO: 파일명 채우기
                page=page_num,
                year="",            # TODO: 연도 채우기
                plan="",            # TODO: 플랜명 채우기
                language="en",
            ))

    # 3. Chroma 저장
    save_to_collection(COLLECTION_NAME, chunks, metadatas)
    print(f"[uhcg] {len(chunks)}개 청크 저장 완료")


if __name__ == "__main__":
    run()
