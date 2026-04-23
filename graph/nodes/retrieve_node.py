# 보험사 컬렉션에서 hybrid search + rerank 실행
import chromadb
from utils.schemas import InsuranceState


def retrieve(state: InsuranceState) -> dict:
    client = chromadb.PersistentClient(path="./vectordb")
    collection = client.get_collection(state["insurer"])

    # TODO: BM25 + Dense hybrid search + RRF + Rerank 구현
    results = collection.query(
        query_texts=[state["user_message"]],
        n_results=5,
    )
    return {"retrieved_docs": results["documents"][0]}
