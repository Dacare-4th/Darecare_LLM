# 노드 연결, 조건부 엣지, SqliteSaver 설정 (그래프 조립)
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from utils.schemas import InsuranceState
from graph.nodes.analyze_node import analyze
from graph.nodes.retrieve_node import retrieve
from graph.nodes.generate_node import generate
from graph.nodes.clarify_node import clarify
from graph.nodes.compare_node import compare
from graph.nodes.nhis_node import nhis


def build():
    workflow = StateGraph(InsuranceState)

    workflow.add_node("analyze", analyze)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("clarify", clarify)
    workflow.add_node("compare", compare)
    workflow.add_node("nhis", nhis)

    # TODO: 조건부 엣지 연결
    workflow.set_entry_point("analyze")

    memory = SqliteSaver.from_conn_string("checkpoints.db")
    return workflow.compile(checkpointer=memory)
