"""Builds the LangGraph agent graph.

    START -> supervisor --retrieve--> retrieval_agent -> response_agent -> END
                        \\-chitchat---------------------> response_agent

Day 2 adds: input/output guards, research agent (RLM), tool agent, validator.
"""

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.response import response_agent
from app.graph.nodes.retrieval import retrieval_agent
from app.graph.nodes.supervisor import supervisor
from app.graph.state import AgentState, UserContext


def _route(state: AgentState) -> str:
    return "chitchat" if state.get("plan", {}).get("route") == "chitchat" else "retrieve"


def build_graph(checkpointer=None):
    g = StateGraph(AgentState, context_schema=UserContext)
    g.add_node("supervisor", supervisor)
    g.add_node("retrieval_agent", retrieval_agent)
    g.add_node("response_agent", response_agent)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", _route, {"retrieve": "retrieval_agent", "chitchat": "response_agent"})
    g.add_edge("retrieval_agent", "response_agent")
    g.add_edge("response_agent", END)
    return g.compile(checkpointer=checkpointer, name="combank_assistant")
