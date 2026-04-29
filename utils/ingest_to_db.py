"""
ingest_to_db.py
청크 리스트 → BGE-M3 임베딩 → ChromaDB 저장 (보험사별 컬렉션 분리)

사용법 (각 전처리 코드에서 import):
    from utils.ingest_to_db import ingest

    chunks = chunk_policy_wording(pdf_path)
    ingest(chunks)

보험사별 컬렉션 매핑:
    msh_china  →  msh_china_plans
    cigna      →  cigna_plans
    uhcg       →  uhcg_plans
    tricare    →  tricare_plans
    nhis       →  nhis_plans
"""

# FIXED LIKE BELOW
# 보험사별 컬렉션 매핑:
#     msh_china  →  msh_china_plans
#     cigna      →  cigna_plans
#     uhcg       →  uhcg_plans
#     tricare    →  tricare_plans
#     nhis       →  nhis_plans



import time
from typing import List
from pathlib import Path
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings



# ── 설정 ──────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent.parent
CHROMA_PATH = str(BASE_DIR / "vectordb")
BATCH_SIZE  = 32
BGE_MODEL   = "BAAI/bge-m3"

INSURER_TO_COLLECTION = {
    "msh_china": "msh_china_plans",
    "cigna"    : "cigna_plans",
    "uhcg"     : "uhcg_plans",
    "tricare"  : "tricare_plans",
    "nhis"     : "nhis_plans",
}


# ── BGE-M3 임베딩 ─────────────────────────────────────────────────────────────

def load_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=BGE_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def embed_texts(model, texts):
    return model.embed_documents(texts)


# ── 컬렉션명 결정 ─────────────────────────────────────────────────────────────

def resolve_collection_name(chunks: list) -> str:
    insurer = chunks[0]["metadata"].get("insurer", "")
    if not insurer:
        raise ValueError("메타데이터에 insurer 필드가 없습니다.")

    collection = INSURER_TO_COLLECTION.get(insurer)
    if not collection:
        raise ValueError(
            f"알 수 없는 insurer: '{insurer}'\n"
            f"INSURER_TO_COLLECTION에 추가해주세요.\n"
            f"현재 등록된 보험사: {list(INSURER_TO_COLLECTION.keys())}"
        )
    return collection


# ── ChromaDB 컬렉션 ───────────────────────────────────────────────────────────

def get_collection(client: chromadb.PersistentClient, collection_name: str):
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ── 메타데이터 정제 ───────────────────────────────────────────────────────────

def sanitize_metadata(meta: dict) -> dict:
    clean = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


# ── 업로드 ────────────────────────────────────────────────────────────────────

def ingest(chunks: list) -> None:
    """
    청크 리스트를 받아 ChromaDB에 업로드.

    Args:
        chunks: [{"chunk_id": str, "content"|"text": str, "metadata": dict}, ...]
    """
    # FIXED ABOVE 위에 수정, content, text 둘다 설명


    print(f"[INFO] 청크 수: {len(chunks)}")

    collection_name = resolve_collection_name(chunks)
    print(f"[INFO] 대상 컬렉션: {collection_name}")

    model  = load_model()
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col    = get_collection(client, collection_name)

    # 중복 방지
    existing_ids = set(col.get(include=[])["ids"])
    new_chunks   = [c for c in chunks if c["chunk_id"] not in existing_ids]
    print(f"[INFO] 신규: {len(new_chunks)}개 / 스킵: {len(chunks)-len(new_chunks)}개")

    if not new_chunks:
        print("[DONE] 업로드할 청크 없음.")
        return

    # 배치 업로드
    total, uploaded, t0 = len(new_chunks), 0, time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = new_chunks[i : i + BATCH_SIZE]
        
        # FIXED BELOW (content, text) 둘다 되게
        texts = [c.get("content") or c.get("text", "") for c in batch]
        col.add(
            ids        = [c["chunk_id"] for c in batch],
            documents  = texts,
            embeddings = embed_texts(model, texts),
            metadatas  = [sanitize_metadata(c["metadata"]) for c in batch],
        )

        uploaded += len(batch)
        elapsed   = time.time() - t0
        eta       = (total - uploaded) / (uploaded / elapsed) if uploaded else 0
        print(f"  [{uploaded}/{total}] {elapsed:.1f}s 경과 | ETA {eta:.0f}s")

    print(f"\n[DONE] {uploaded}개 업로드 완료 → {CHROMA_PATH}{collection_name}")
    print(f"[INFO] 컬렉션 총 문서 수: {col.count()}")
