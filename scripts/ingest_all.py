# 인자로 보험사 지정 시 해당 보험사(들)만, 미지정 시 에러 메시지
# 사용법: python scripts/ingest_all.py uhcg cigna
#         python scripts/ingest_all.py nhis
import sys

AVAILABLE = ["uhcg", "cigna", "tricare", "msh_china", "nhis"]

def main():
    targets = sys.argv[1:]

    if not targets:
        print("오류: 보험사를 지정해주세요.")
        print(f"사용법: python scripts/ingest_all.py <보험사> [보험사...]")
        print(f"사용 가능한 보험사: {', '.join(AVAILABLE)}")
        sys.exit(1)

    invalid = [t for t in targets if t not in AVAILABLE]
    if invalid:
        print(f"오류: 알 수 없는 보험사 → {', '.join(invalid)}")
        print(f"사용 가능한 보험사: {', '.join(AVAILABLE)}")
        sys.exit(1)

    for insurer in targets:
        print(f"\n{'='*40}")
        print(f"[{insurer}] ingest 시작")
        print(f"{'='*40}")

        if insurer == "uhcg":
            from plugins.uhcg.ingest import run
        elif insurer == "cigna":
            from plugins.cigna.ingest import run
        elif insurer == "tricare":
            from plugins.tricare.ingest import run
        elif insurer == "msh_china":
            from plugins.msh_china.ingest import run
        elif insurer == "nhis":
            from plugins.nhis.ingest import run

        run()
        print(f"[{insurer}] 완료")


if __name__ == "__main__":
    main()
