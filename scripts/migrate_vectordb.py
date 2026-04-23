# 벡터DB 스키마 변경 시 컬렉션 재생성 및 마이그레이션
# 사용법: python scripts/migrate_vectordb.py
# ⚠️  실행 전 반드시 vectordb/ 백업할 것
import chromadb
import sys

VECTORDB_PATH = "./vectordb"
COLLECTIONS = ["uhcg", "cigna", "tricare", "msh_china", "nhis"]


def migrate():
    client = chromadb.PersistentClient(path=VECTORDB_PATH)

    print("현재 컬렉션 목록:")
    for col in client.list_collections():
        count = client.get_collection(col.name).count()
        print(f"  - {col.name}: {count}개 청크")

    confirm = input("\n모든 컬렉션을 삭제하고 재생성하시겠습니까? (yes/no): ")
    if confirm.lower() \!= "yes":
        print("취소되었습니다.")
        sys.exit(0)

    for name in COLLECTIONS:
        try:
            client.delete_collection(name)
            print(f"[삭제] {name}")
        except Exception:
            pass

    print("\n컬렉션 삭제 완료. 이제 ingest_all.py를 다시 실행해주세요.")
    print("예: python scripts/ingest_all.py uhcg cigna tricare msh_china nhis")


if __name__ == "__main__":
    migrate()
