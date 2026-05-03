from fastapi import APIRouter
from app.schemas import ChatRequest, ChatResponse
from graph.builder import build
from fastapi.responses import FileResponse
import os

router = APIRouter()
graph = build()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    invoke_state = {
        "user_id": request.user_id,
        "session_id": request.session_id,
        "user_message": request.message,
        "comparison_criteria": request.comparison_criteria,
    }
    if request.insurer:
        invoke_state["insurer"] = request.insurer

    result = graph.invoke(
        invoke_state,
        config={
            "configurable": {
                "thread_id": request.session_id
            }
        }
    )

    return ChatResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        claim_form=result.get("claim_form", []),
        compare_table=result.get("compare_table") or None,
        related_questions=result.get("related_questions", []),
    )

@router.get("/download/{insurer}/{filename}")
async def download_file(filename: str, insurer: str):
    root_directory = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
    file_path = os.path.join(root_directory, 'data', _normalize_insurer(insurer.lower()), 'claim_forms', filename)
    return FileResponse(path=file_path, filename=filename)

def _normalize_insurer(insurer: str) -> str:
    """
    보험사명을 시스템 내부 코드로 정규화한다.
    """
    insurer = (insurer or "").lower().strip()

    aliases = {
        "uhc": "uhcg",
        "uhcg": "uhcg",
        "unitedhealth": "uhcg",
        "cigna": "cigna",
        "tricare": "tricare",
        "msh china": "msh",
        "msh_china": "msh",
        "nhis": "nhis"
    }

    return aliases.get(insurer, insurer)