#!/usr/bin/env python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# scripts/test_within.py
# 역할 : within_node (파이프라인 ①: 보험 내 플랜 비교) 단독 실행 테스트
#
# 사용법:
#   python scripts/test_within.py
#
# 확인 항목:
#   - answer            : LLM이 생성한 자연어 요약
#   - compare_table     : {"header": [...], "body": [[...]]} 구조 확인
#   - related_questions : 3개 고정 확인
#   - sources           : 출처 메타데이터 목록 확인
#   - retrieved_docs    : 검색된 문서 수 확인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import os
import pprint
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.nodes.within_node import within
from utils.schemas import initial_state


def run_test(session_id: str, user_message: str, insurer: str, slots: dict) -> None:
    print("=" * 60)
    print(f"[테스트] {user_message}")
    print(f"  insurer : {insurer} | slots : {slots}")
    print("=" * 60)

    state = initial_state(session_id=session_id, user_message=user_message)
    state.update({
        "language": "ko",
        "insurer" : insurer,
        "slots"   : slots,
    })

    result = within(state)

    print("\n[answer]")
    print(result.get("answer", "(없음)"))

    ct = result.get("compare_table", {})
    print(f"\n[compare_table]")
    print(f"  header     : {ct.get('header', [])}")
    print(f"  body 행 수 : {len(ct.get('body', []))}")
    if ct.get("body"):
        print("  body 첫 행 :", ct["body"][0])

    rqs = result.get("related_questions", [])
    print(f"\n[related_questions] ({len(rqs)}개)")
    for i, q in enumerate(rqs, 1):
        print(f"  {i}. {q}")

    sources = result.get("sources", [])
    print(f"\n[sources] ({len(sources)}개)")
    for i, s in enumerate(sources, 1):
        print(f"  [{i}]")
        print(f"    document_name : {s.get('document_name') or '(없음)'}")
        print(f"    page          : {s.get('page') if s.get('page') is not None else '(없음)'}")
        print(f"    section       : {s.get('section') or '(없음)'}")

    print(f"\n[retrieved_docs 수] : {len(result.get('retrieved_docs', []))}개")
    print("=" * 60 + "\n")


def main() -> None:
    # ── UHCG 테스트 ──────────────────────────────────────────
    run_test(
        session_id   = "within-uhcg-001",
        user_message = "UHCG Core 1 플랜과 Core 2 플랜 보장 범위 비교해줘.",
        insurer      = "uhcg",
        slots        = {"plans": ["Core 1", "Core 2"]},
    )

    # ── TRICARE 테스트 ────────────────────────────────────────
    run_test(
        session_id   = "within-tricare-001",
        user_message = "TRICARE Prime과 TRICARE Select 입원 보장 비교해줘.",
        insurer      = "tricare",
        slots        = {"plans": ["TRICARE Prime", "TRICARE Select"]},
    )

    # ── MSH China 테스트 ──────────────────────────────────────
    run_test(
        session_id   = "within-msh-001",
        user_message = "MSH China 보험 플랜들의 외래 진료 보장을 비교해줘.",
        insurer      = "msh_china",
        slots        = {},
    )


if __name__ == "__main__":
    main()
