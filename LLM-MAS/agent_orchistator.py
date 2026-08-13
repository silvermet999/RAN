import asyncio
import time
from typing import Annotated, Any, Dict, List, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, add_messages
import json
import sys
import os
from langgraph.types import CachePolicy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agents


memory = MemorySaver()

# ollama pull llama3.1:8b
llm = ChatOllama(
    model="llama3",
    base_url="http://localhost:11434",
    temperature=0,
    max_tokens=500, 
    num_gpu = 2
)

# rag_agent = build_rag_agent()
agent_orchi_instruction = """
    You are a strict routing and assistance agent.

    Your job is to:
    - Classify the user query
    - Either respond directly (INTENT) or route to another agent

    ---

    Classify the user query into EXACTLY ONE category:

    1. INTENT → vague, exploratory, or “what is” questions
    2. PROPOSAL → proposal, schema, nodes, relationships, graph construction
    3. RAG → querying or retrieving from an existing knowledge graph, might include retrieval using math and statistical tools.

    ---

    Routing Rules (STRICT):

    - propose / schema / nodes / relationships → PROPOSAL
    - retrieve / query / search / existing data / math tools → RAG
    - vague / unclear / guidance needed → INTENT
    - if unsure → ask a clarification question ONLY

    ---

    Response Rules (MANDATORY):

    - If INTENT:
    → Answer the question of the user
    → Do NOT mention routing
    → Do NOT be verbose

    - If PROPOSAL:
    → Respond EXACTLY: "Routing to proposal agent"

    - If RAG:
    → Respond EXACTLY: "Routing to RAG agent"

    - If unsure:
    → Respond EXACTLY: "Need clarification: <your question>"

    ---

    DO NOT:
    - Answer technical schema or retrieval questions
    - Deviate from the formats above

"""



class MessageState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    history: List[str]
    user_input: str
    agent_route: str
    json_keys: List[str]
    uploaded_file: Any
    schema: dict
    approved: str
    generation: str
    query: str
    tool_results: Any
    start_node: str
    end_node: str


def agent_orchi_node(state: MessageState):
    messages = [SystemMessage(content=agent_orchi_instruction)] + state["messages"]
    start = time.time()
    # if asyncio.get_event_loop().is_running():
    #     response = await llm.ainvoke(messages)  # Continue with existing loop
    # else:
    response = llm.invoke(messages)
    
    print("invoked orchistrator, time:", (time.time() - start)/60)
    pending_schema=state["schema"]
    json_keys = state.get("json_keys")
    tool_results = state["tool_results"]
    start_node = state.get("start_node")

    #history = state["history"]
    return { #"history": history,
        "messages": [AIMessage(content=response.content)], "json_keys": json_keys, "schema": pending_schema, 
        "tool_results": tool_results, "start_node": start_node
        }

def routing_agent_node(state: MessageState):
    agent = state["messages"][-1].content
    # if "intent" in agent:
    #     agent_route = "intent"
    if "proposal" in agent:
        agent_route = "proposal"
    elif "RAG" in agent:
        agent_route = "RAG"
    else:
        agent_route= "end"

    result = {"agent_route": agent_route}

    return result



# def routing_proposal_node(state: MessageState):
#     last_msg = state["messages"][-1]

#     # Default condition
#     proposal_cond = "not approved"

#     if state.get("agent_route") == "proposal":
#         if isinstance(last_msg, HumanMessage):
#             user_input = last_msg.content.lower()

#             if "approve" in user_input or "yes" in user_input:
#                 proposal_cond = "approved"

#     return {"proposal_cond": proposal_cond}


# def approve_schema_agent(state: MessageState):
#     user_input = state["messages"][0]["content"]
#     if "approve" in user_input.lower():
#         return "approved"
#     else:
#         return "not_approved"


def build_orchistrator_agent():
    graph = StateGraph(MessageState)
    graph.add_node("orchi_node", agent_orchi_node, cache_policy=CachePolicy())
    #graph.add_node("intent_node", user_intent_node, cache_policy=CachePolicy())
    graph.add_node("proposal_node", proposal_agent_node, cache_policy=CachePolicy())
    graph.add_node("rag_agent_node", graphrag_agent, cache_policy=CachePolicy())
    graph.add_node("routing_agent_node", routing_agent_node, cache_policy=CachePolicy())
    # graph.add_node("save_schema", save_proposal_to_json)

    graph.add_edge(START, "orchi_node")
    graph.add_edge("orchi_node", "routing_agent_node")
    graph.add_conditional_edges(
        "routing_agent_node",
        lambda state: state["agent_route"],
        {
            "proposal": "proposal_node",
            "RAG": "rag_agent_node",
            "end": END
        },
    )
    
    graph.add_edge("routing_agent_node", END)
    return graph.compile(checkpointer=memory)

