# 사용자 입력 → intent + slot 추출
from utils.schemas import InsuranceState
from utils.safety import check_blocked


def analyze(state: InsuranceState) -> dict:
    # 1. 안전 필터 먼저 실행
    blocked = check_blocked(state["user_message"])
    if blocked:
        return {"answer": blocked, "intent": "blocked"}

    # 2. LLM으로 intent, slot 추출
    # TODO: 구현
    return {}
