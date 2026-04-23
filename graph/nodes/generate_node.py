# 검색 결과 → 최종 답변 생성 + 언어 자동 감지
from utils.schemas import InsuranceState


def generate(state: InsuranceState) -> dict:
    # TODO: retrieved_docs + slots 기반으로 LLM 호출
    # TODO: 응답 언어 = 사용자 입력 언어 자동 감지
    return {"answer": ""}
