# Dacare LLM

외국인 대상 보험 혜택 안내 챗봇 — FastAPI + LangGraph + Chroma

## 지원 보험사
- UHCG
- Cigna
- Tricare
- MSH China
- NHIS (국민건강보험)

## 파일구조

```
Dacare_LLM/
│
├── .github/
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
│
├── app/
│   ├── main.py                     # FastAPI 앱 생성, 라우터 등록, CORS 설정
│   ├── schemas.py                  # Pydantic 모델 — Django↔FastAPI 요청/응답 형식 정의
│   └── api/
│       ├── chat.py                 # POST /chat — LangGraph 호출 후 응답 반환
│       └── health.py               # GET  /health — 서버 상태 확인 (배포용 헬스체크)
│
├── graph/
│   ├── builder.py                  # 노드 연결, 조건부 엣지, SqliteSaver 설정 (그래프 조립)
│   └── nodes/
│       ├── analyze_node.py         # 사용자 입력 → intent + slot 추출
│       ├── retrieve_node.py        # 보험사 컬렉션에서 hybrid search + rerank 실행
│       ├── generate_node.py        # 검색 결과 → 최종 답변 생성 + 언어 자동 감지
│       ├── clarify_node.py         # 정보 부족 또는 사용자에게 재질문 생성
│       ├── compare_node.py         # 복수 보험사 동시 검색 후 비교표 생성
│       └── nhis_node.py            # NHIS 적용 후 민간보험 잔여 청구금액 계산
│
├── plugins/
│   ├── base.py                     # InsurancePlugin ABC — 모든 플러그인이 구현해야 할 인터페이스
│   ├── uhcg/
│   │   ├── uhcg_plugin.py          # UHCGPlugin — 플랜 목록, 시스템 프롬프트, 슬롯 분석 구현
│   │   └── ingest.py               # UHCG PDF → 청킹 → DocumentMetadata 태깅 → uhcg 컬렉션 저장
│   ├── cigna/
│   │   ├── cigna_plugin.py         # CignaPlugin — 플랜 목록, 시스템 프롬프트, 슬롯 분석 구현
│   │   └── ingest.py               # Cigna PDF → 청킹 → DocumentMetadata 태깅 → cigna 컬렉션 저장
│   ├── tricare/
│   │   ├── tricare_plugin.py       # TricarePlugin — 플랜 목록, 시스템 프롬프트, 슬롯 분석 구현
│   │   └── ingest.py               # Tricare PDF → 청킹 → DocumentMetadata 태깅 → tricare 컬렉션 저장
│   ├── msh_china/
│   │   ├── msh_china_plugin.py     # MSHChinaPlugin — 플랜 목록, 시스템 프롬프트, 슬롯 분석 구현
│   │   └── ingest.py               # MSH PDF → 청킹 → DocumentMetadata 태깅 → msh_china 컬렉션 저장
│   └── nhis/
│       ├── nhis_plugin.py          # NHISPlugin — 자격 확인 질의, 급여 범위 안내 구현
│       └── ingest.py               # NHIS 웹 크롤링 → 청킹 → DocumentMetadata 태깅 → nhis 컬렉션 저장
│
├── utils/
│   ├── schemas.py                  # TypedDict 전부 — InsuranceState, AnalysisResult, RetrieveResult, DocumentMetadata
│   ├── ingest_utils.py             # 공용 전처리 함수 — PDF 로딩, 청킹, Chroma 저장 (각 ingest.py가 import해서 사용)
│   ├── currency.py                 # frankfurter.app 호출 — KRW/USD/EUR 등 환율 변환
│   ├── comparison.py               # 복수 보험사 결과 병합 + 비교표 포맷팅
│   ├── language.py                 # 입력 텍스트 언어 감지 (7개 언어)
│   └── safety.py                   # PII 차단, RECOMMENDATION_KEYWORDS, COMPARISON_KEYWORDS, check_blocked()
│
├── vectordb/                       # 단일 Chroma 저장소 — 보험사별 컬렉션(uhcg, cigna, tricare, msh_china, nhis)으로 구분 (git 제외)
│
├── data/                           # 원본 PDF 보관 (git 제외)
│   ├── uhcg/
│   ├── cigna/
│   ├── tricare/
│   ├── msh_china/
│   └── nhis/
│
├── scripts/
│   ├── ingest_all.py               # 인자로 보험사 지정 시 해당 보험사(들)만, 미지정 시 에러 메세지 (예: python ingest_all.py uhcg cigna)
│   └── migrate_vectordb.py         # 벡터DB 스키마 변경 시 컬렉션 재생성 및 마이그레이션
│
├── notebooks/                      # 실험 및 분석용 Jupyter 노트북
├── finetuning/                     # 파인튜닝 데이터셋 및 학습 스크립트
├── evaluation/                     # RAG 자동 평가 (eval_runner.py, eval_dataset.json)
├── docs/                           # 프로젝트 문서 (API 명세, 온보딩 가이드)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 시작하기

```bash
# 환경변수 설정
cp .env.example .env

# 패키지 설치
pip install -r requirements.txt

# PDF 데이터 전처리 (보험사 지정 필수)
python scripts/ingest_all.py uhcg cigna

# 서버 실행
uvicorn app.main:app --reload
```

## 문서
- [API 명세](docs/api_spec.md)
- [새 보험사 추가 가이드](docs/onboarding_guide.md)
