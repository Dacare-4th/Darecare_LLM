# FastAPI 앱 생성, 라우터 등록, CORS 설정
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat, health

from dotenv import load_dotenv
load_dotenv()  # ← 이걸 추가해야 .env 파일이 로드됨!

app = FastAPI(title="Dacare LLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 Django 서버 주소로 제한
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(health.router)
