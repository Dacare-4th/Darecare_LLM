# CignaPlugin — 플랜 목록, 시스템 프롬프트, 슬롯 분석 구현
from plugins.base import InsurancePlugin
from utils.schemas import InsuranceState, AnalysisResult

SYSTEM_PROMPT = """
You are an expert assistant for CignaPlugin insurance benefits.
Answer only based on the provided documents.
"""

PLANS = []  # TODO: 플랜 목록 채우기


class CignaPlugin(InsurancePlugin):

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    @property
    def plans(self) -> list[str]:
        return PLANS

    def analyze(self, state: InsuranceState) -> AnalysisResult:
        # TODO: intent, slot 추출 구현
        return AnalysisResult(
            intent="benefit_query",
            slots={},
            language=state.get("language", "en"),
            missing_slots=[],
            confidence=0.0,
        )
