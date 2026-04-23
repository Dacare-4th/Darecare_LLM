# GET /health — 서버 상태 확인 (배포용 헬스체크)
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}
