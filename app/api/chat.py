# POST /chat — LangGraph 호출 후 응답 반환
from fastapi import APIRouter
from app.schemas import ChatRequest, ChatResponse
from graph.builder import build

router = APIRouter()
graph = build()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = graph.invoke({
        "session_id": request.session_id,
        "user_message": request.user_message,
        "insurer": request.insurer,
        "language": request.language,
    })
    return ChatResponse(
        session_id=request.session_id,
        answer=result["answer"],
        intent=result["intent"],
        language=result["language"],
    )
