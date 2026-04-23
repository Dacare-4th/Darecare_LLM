# InsurancePlugin ABC — 모든 플러그인이 구현해야 할 인터페이스
from abc import ABC, abstractmethod
from utils.schemas import InsuranceState, AnalysisResult


class InsurancePlugin(ABC):

    @abstractmethod
    def analyze(self, state: InsuranceState) -> AnalysisResult:
        """사용자 입력에서 intent, slot 추출"""
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """이 보험사에 특화된 시스템 프롬프트"""
        ...

    @property
    @abstractmethod
    def plans(self) -> list[str]:
        """이 보험사가 제공하는 플랜 목록"""
        ...
