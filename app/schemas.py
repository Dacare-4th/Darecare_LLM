from typing import Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: int
    session_id: str
    insurer: str
    message: str

    # insurer == "compare"일 때 Django에서 넘겨줄 비교 기준
    comparison_criteria: list[str] = Field(default_factory=list)


class Source(BaseModel):
    document_name: str = ""
    page: int | None = None
    section: str = ""


class ClaimForm(BaseModel):
    claim_form_path: str
    claim_form_name: str
    claim_form_ext: str


class CompareTable(BaseModel):
    header: list[str] = Field(default_factory=list)
    body: list[list[str]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    claim_form: list[ClaimForm] = Field(default_factory=list)
    compare_table: CompareTable | None = None
    related_questions: list[str] = Field(default_factory=list)