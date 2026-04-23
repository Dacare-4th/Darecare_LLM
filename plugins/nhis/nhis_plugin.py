# NHISPlugin — 자격 확인 질의, 급여 범위 안내 구현
from plugins.base import InsurancePlugin
from utils.schemas import InsuranceState, AnalysisResult

SYSTEM_PROMPT = """
You are an expert assistant for Korean National Health Insurance Service (NHIS).
Answer eligibility, coverage, and copayment questions based on official NHIS documents.
"""

PLANS = []  # NHIS는 플랜 구분 없음


class NHISPlugin(InsurancePlugin):

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    @property
    def plans(self) -> list[str]:
        return PLANS

    def analyze(self, state: InsuranceState) -> AnalysisResult:
        # TODO: NHIS 관련 intent 분류 (자격확인 / 급여범위 / 본인부담금 / 민간보험 연계)
        return AnalysisResult(
            intent="nhis_query",
            slots={},
            language=state.get("language", "en"),
            missing_slots=[],
            confidence=0.0,
        )
