# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# plugins/uhcg/ingest.py
# 역할 : UHCG 전처리 JSON → uhcg_plans 컬렉션 저장
#
# 데이터 소스:
#   - data/output/uhcg/uhcg_guide/*.json   (가이드 문서)
#   - data/uhc/claim_forms/*.pdf           (청구서 양식, 직접 등록)
#
# 사용법: python scripts/ingest_all.py uhcg
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import json
from pathlib import Path

from utils.ingest_to_db import ingest

BASE_DIR        = Path(__file__).resolve().parent.parent.parent
GUIDE_OUTPUT    = BASE_DIR / "data" / "output" / "uhcg" / "uhcg_guide"
CLAIM_FORMS_DIR = BASE_DIR / "data" / "uhc" / "claim_forms"


# ──────────────────────────────────────────────────────────────
# 청구서 양식 청크 (PDF 직접 등록 — Cigna 방식)
# claim_node가 doc_type=="claim_form" 으로 필터링해 다운로드 링크 제공
# ──────────────────────────────────────────────────────────────

_CLAIM_FORM_CHUNKS: list[dict] = [
    {
        "chunk_id": "uhcg_claim_form_expatriate",
        "text": (
            "UHCG Expatriate Insurance Claim Form. "
            "Use this form to submit medical expense reimbursement claims for "
            "UnitedHealthcare Global expatriate insurance plans. "
            "Complete all sections and attach original receipts, itemized bills, "
            "and supporting medical documentation."
        ),
        "metadata": {
            "insurer":     "uhcg",
            "doc_type":    "claim_form",
            "source":      "MBR-C-26437_UHCG_Expatriate_Insurance_Claim_Form_200914_Editable.pdf",
            "file_name":   "MBR-C-26437_UHCG_Expatriate_Insurance_Claim_Form_200914_Editable.pdf",
            "source_type": "form",
            "language":    "en",
            "plan":        "",
            "year":        "2020",
        },
    },
    {
        "chunk_id": "uhcg_claim_form_dental",
        "text": (
            "UHCG Dental Only Member Claim Form. "
            "Use this form to submit dental expense reimbursement claims for "
            "UnitedHealthcare Global dental coverage. "
            "Include itemized dental treatment records and receipts."
        ),
        "metadata": {
            "insurer":     "uhcg",
            "doc_type":    "claim_form",
            "source":      "MBR-EXP-1694500-CF Dental Only Member Claim Form_230201_HRPrint.pdf",
            "file_name":   "MBR-EXP-1694500-CF Dental Only Member Claim Form_230201_HRPrint.pdf",
            "source_type": "form",
            "language":    "en",
            "plan":        "",
            "year":        "2023",
        },
    },
]


# ──────────────────────────────────────────────────────────────
# 가이드 JSON 로드
# ──────────────────────────────────────────────────────────────

def _load_guide_chunks() -> list[dict]:
    """data/output/uhcg/uhcg_guide/*.json 에서 모든 청크를 로드한다."""
    chunks: list[dict] = []

    if not GUIDE_OUTPUT.exists():
        print(f"[uhcg] 가이드 출력 디렉토리 없음: {GUIDE_OUTPUT}")
        return chunks

    for json_file in sorted(GUIDE_OUTPUT.glob("*.json")):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                chunks.extend(data)
                print(f"  [{json_file.name}] {len(data)}개 청크 로드")
            else:
                print(f"  [{json_file.name}] 예상치 못한 형식 — 스킵")
        except Exception as e:
            print(f"  [{json_file.name}] 로드 실패: {e}")

    return chunks


# ──────────────────────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────────────────────

def run() -> None:
    print("\n[uhcg] 인제스트 시작")

    # 1. 가이드 청크 로드
    print("\n[uhcg] 가이드 JSON 로드 중...")
    guide_chunks = _load_guide_chunks()
    print(f"  → 가이드 청크 합계: {len(guide_chunks)}개")

    # 2. 청구서 양식 청크 추가
    print(f"\n[uhcg] 청구서 양식 청크: {len(_CLAIM_FORM_CHUNKS)}개")
    for c in _CLAIM_FORM_CHUNKS:
        pdf_path = CLAIM_FORMS_DIR / c["metadata"]["file_name"]
        exists   = "✓" if pdf_path.exists() else "✗ (PDF 없음 — 링크만 등록)"
        print(f"  {exists} {c['metadata']['file_name']}")

    # 3. 전체 합산 후 인제스트
    all_chunks = guide_chunks + _CLAIM_FORM_CHUNKS

    if not all_chunks:
        print("[uhcg] 인제스트할 청크 없음 — 종료")
        return

    print(f"\n[uhcg] 총 {len(all_chunks)}개 청크 → uhcg_plans 컬렉션 저장 시작")
    ingest(all_chunks)
    print("\n[uhcg] 인제스트 완료")


if __name__ == "__main__":
    run()
