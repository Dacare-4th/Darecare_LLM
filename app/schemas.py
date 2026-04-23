# Pydantic 모델 — Django↔FastAPI 요청/응답 형식 정의
from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    user_message: str
    insurer: str      # "uhcg" | "cigna" | "tricare" | "msh_china" | "nhis"
    language: str     # "en" | "ko" | "zh" | ...


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: str       # "benefit_query" | "comparison" | "nhis" | "currency"
    language: str
