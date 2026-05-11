# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# graph/builder.py
# 역할 : LangGraph 그래프를 조립한다 (노드 등록 + 조건부 엣지 연결)
#
# 그래프 흐름 요약:
#
#   [START]
#     ↓
#   analyze  ← 모든 요청의 진입점 (언어감지 + Intent Router)
#     ↓ route_after_analyze()
#     ├─ "within_compare"  → within    → END  (① 보험 내 비교)
#     ├─ "cross_compare"   → compare   → END  (② 보험사 비교)
#     ├─ "calculation"     → calculate → END  (③ 계산)
#     ├─ "procedure"       → procedure → END  (④ 절차 안내)
#     ├─ "nhis"            → nhis               (⑤ NHIS)
#     │                        ↓ route_after_nhis()
#     │                        ├─ "claim"  → claim → END  (민간보험 연계)
#     │                        └─ default  → END
#     ├─ "claim"           → claim     → END  (⑥ 청구 + 양식)
#     ├─ "general_query"   → general   → END  (⑦ 일반 coverage 질문)
#     ├─ "clarify"         → clarify   → END  (재질문)
#     └─ "blocked"         → END              (안전 필터 차단)
#
# 멀티턴 지원:
#   SqliteSaver 체크포인터로 세션별 상태를 유지한다.
#   session_id 를 thread_id 로 사용해 대화 이력을 관리한다.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

import sqlite3

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from utils.schemas import InsuranceState, Intent

# ── 노드 임포트 ────────────────────────────────────────────────
from graph.nodes.analyze_node    import analyze
from graph.nodes.within_node     import within
from graph.nodes.compare_node    import compare
from graph.nodes.calculate_node  import calculate
from graph.nodes.procedure_node  import procedure
from graph.nodes.nhis_node       import nhis
from graph.nodes.claim_node      import claim
from graph.nodes.clarify_node    import clarify
from graph.nodes.general_node    import general


# ──────────────────────────────────────────────────────────────
# 라우팅 함수 — 조건부 엣지에서 호출되는 순수 함수
# 반환값은 반드시 add_conditional_edges 의 mapping 키와 일치해야 함
# ──────────────────────────────────────────────────────────────

def route_after_analyze(state: InsuranceState) -> str:
    """
    analyze 노드 실행 후 다음 노드를 결정한다.

    state["intent"] 값을 읽어 해당 파이프라인 노드 이름을 반환한다.
    알 수 없는 intent 는 "clarify" 로 fallback 한다.

    Returns:
        다음 노드 이름 문자열 (mapping 키와 일치)
    """
    intent = state.get("intent", Intent.CLARIFY)

    routing_map = {
        Intent.WITHIN_COMPARE : "within",
        Intent.CROSS_COMPARE  : "compare",
        Intent.CALCULATION    : "calculate",
        Intent.PROCEDURE      : "procedure",
        Intent.NHIS           : "nhis",
        Intent.CLAIM          : "claim",
        Intent.GENERAL_QUERY  : "general",
        Intent.CLARIFY        : "clarify",
        Intent.BLOCKED        : END,       # 안전 필터 차단 → 즉시 종료
        Intent.RECOMMENDATION : END,       # 추천/법적·의학적 판단 요청 → 즉시 종료
    }

    return routing_map.get(intent, "clarify")


def route_after_nhis(state: InsuranceState) -> str:
    """
    nhis 노드 실행 후 다음 노드를 결정한다.

    nhis_node 내부에서 민간보험 청구 의도를 감지한 경우
    intent 를 "claim" 으로 업데이트하고 claim_node 로 이동한다.
    그 외에는 END.

    Returns:
        "claim" 또는 END
    """
    intent = state.get("intent", "")
    if intent == Intent.CLAIM:
        return "claim"
    return END


# ──────────────────────────────────────────────────────────────
# 그래프 조립 함수
# ──────────────────────────────────────────────────────────────

def build() -> "CompiledGraph":
    """
    LangGraph 그래프를 조립하고 컴파일한 인스턴스를 반환한다.

    SqliteSaver 를 체크포인터로 사용해 멀티턴 대화를 지원한다.
    graph.invoke() 호출 시 config={"configurable": {"thread_id": session_id}}
    를 전달해야 세션별 상태가 분리된다.

    Returns:
        컴파일된 LangGraph CompiledGraph 인스턴스
    """
    workflow = StateGraph(InsuranceState)

    # ── 노드 등록 ──────────────────────────────────────────────
    # 각 노드 이름은 route_after_analyze() 의 반환값과 일치해야 함
    workflow.add_node("analyze",   analyze)    # 진입점: 언어감지 + Intent Router
    workflow.add_node("within",    within)     # ① 보험 내 비교
    workflow.add_node("compare",   compare)    # ② 보험사 비교
    workflow.add_node("calculate", calculate)  # ③ 계산
    workflow.add_node("procedure", procedure)  # ④ 절차 안내
    workflow.add_node("nhis",      nhis)       # ⑤ NHIS
    workflow.add_node("claim",     claim)      # ⑥ 청구 + 양식
    workflow.add_node("general",   general)    # ⑦ 일반 coverage 질문
    workflow.add_node("clarify",   clarify)    # 재질문

    # ── 진입점 설정 ────────────────────────────────────────────
    workflow.set_entry_point("analyze")

    # ── 조건부 엣지: analyze → 각 파이프라인 ──────────────────
    workflow.add_conditional_edges(
        source  = "analyze",
        path    = route_after_analyze,
        path_map = {
            "within"    : "within",
            "compare"   : "compare",
            "calculate" : "calculate",
            "procedure" : "procedure",
            "nhis"      : "nhis",
            "claim"     : "claim",
            "general"   : "general",
            "clarify"   : "clarify",
            END         : END,
        },
    )

    # ── 조건부 엣지: nhis → claim 또는 END ────────────────────
    # nhis_node 가 민간보험 청구 의도를 감지하면 claim_node 로 이동
    workflow.add_conditional_edges(
        source   = "nhis",
        path     = route_after_nhis,
        path_map = {
            "claim": "claim",
            END    : END,
        },
    )

    # ── 단순 엣지: 각 파이프라인 → END ────────────────────────
    for node_name in ["within", "compare", "calculate", "procedure",
                      "claim", "general", "clarify"]:
        workflow.add_edge(node_name, END)

    # ── 체크포인터 설정 (멀티턴 상태 유지) ────────────────────
    # thread_id = session_id 로 대화 세션을 분리한다.
    # 체크포인트 DB 경로는 환경변수로 변경 가능하도록 설정 권장.
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    memory = SqliteSaver(conn)

    return workflow.compile(checkpointer=memory)
