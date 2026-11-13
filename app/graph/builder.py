"""Builds the LangGraph agent graph.

    START -> supervisor --retrieve--> retrieval_agent ---------------------------------> response_agent -> END
                        --research--> research_planner =Send×N=> research_worker ⟲ => research_aggregator ->
                        --chitchat--------------------------------------------------------> response_agent

research_worker can send work to new research_worker tasks (recursion); research_aggregator is
deferred and continues to the answer only when every selected document is covered.

Day 2 adds: input/output guards, tool agent, validator.
"""

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.research import fan_out, research_aggregator, research_planner, research_worker
from app.graph.nodes.response import response_agent
from app.graph.nodes.retrieval import retrieval_agent
from app.graph.nodes.supervisor import supervisor
from app.graph.state import AgentState, UserContext

ROUTES = {"retrieve": "retrieval_agent", "research": "research_planner", "chitchat": "response_agent"}


def _route(state: AgentState) -> str:
    route = state.get("plan", {}).get("route")
    return route if route in ROUTES else "retrieve"


def build_graph(checkpointer=None):
    g = StateGraph(AgentState, context_schema=UserContext)
    g.add_node("supervisor", supervisor)
    g.add_node("retrieval_agent", retrieval_agent)
    g.add_node("research_planner", research_planner)
    g.add_node("research_worker", research_worker, destinations=("research_worker",))
    g.add_node("research_aggregator", research_aggregator, defer=True, destinations=("response_agent",))
    g.add_node("response_agent", response_agent)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", _route, ROUTES)
    g.add_edge("retrieval_agent", "response_agent")
    g.add_conditional_edges("research_planner", fan_out, ["research_worker", "response_agent"])
    g.add_edge("research_worker", "research_aggregator")
    g.add_edge("response_agent", END)
    return g.compile(checkpointer=checkpointer, name="combank_assistant")
