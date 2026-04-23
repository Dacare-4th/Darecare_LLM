# TypedDict 전부 — InsuranceState, AnalysisResult, RetrieveResult, DocumentMetadata
from typing import TypedDict


class InsuranceState(TypedDict):
    """LangGraph 전역 상태 — 모든 노드가 읽고 쓰는 공유 데이터"""
    session_id: str
    user_message: str
    insurer: str            # "uhcg" | "cigna" | "tricare" | "msh_china" | "nhis"
    intent: str             # "benefit_query" | "comparison" | "nhis" | "currency" | "blocked"
    slots: dict             # {"plan": "Gold", "treatment": "입원", "region": "Seoul"}
    missing_slots: list     # 아직 확인되지 않은 슬롯 목록
    language: str           # "en" | "ko" | "zh" | "ja" | "es" | "fr" | "de"
    retrieved_docs: list    # retrieve_node 결과
    answer: str             # 최종 응답


class AnalysisResult(TypedDict):
    """analyze_node → retrieve_node 전달 데이터"""
    intent: str
    slots: dict
    language: str
    missing_slots: list
    confidence: float


class RetrieveResult(TypedDict):
    """retrieve_node → generate_node 전달 데이터"""
    documents: list
    scores: list
    strategy: str           # "similarity" | "hybrid"


class DocumentMetadata(TypedDict):
    """모든 ingest.py가 반드시 따라야 하는 Chroma 메타데이터 표준"""
    insurer: str            # "uhcg" | "cigna" | "tricare" | "msh_china" | "nhis"
    source_type: str        # "pdf" | "web"
    file_name: str          # PDF 파일명 또는 웹 토픽명
    page: int               # PDF 페이지 번호 (웹은 0)
    year: str               # 문서 연도 (예: "2024")
    plan: str               # 플랜명 (없으면 "")
    language: str           # "en" | "ko"
