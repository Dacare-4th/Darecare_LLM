# 정보 부족 또는 사용자에게 재질문 생성
from utils.schemas import InsuranceState


def clarify(state: InsuranceState) -> dict:
    # TODO: missing_slots 기반으로 재질문 생성
    return {"answer": ""}
