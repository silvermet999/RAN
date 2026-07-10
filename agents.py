import asyncio
import io
import json
import re
import time
from typing import Annotated, Any, Dict, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph, add_messages
from langchain_core.runnables import RunnableSequence
from langgraph.types import CachePolicy

memory = MemorySaver()

llm = ChatOllama(
    model="llama3",
    base_url="http://localhost:11434",
    temperature=0.7,
)

main_instruction = PromptTemplate(template="""
    You are an expert in cybersecurity, specifically network intrusion detection. Assist a cybersecurity analyst in identifying network attacks.
    Answer the following questions to determine if there is an attack:
        - "Are there any UEs showing abnormal downlink block error rates right now?"
        - "Which base stations have the largest gap between requested and granted PRBs?"
        - "Are there any UEs whose ul_ta value changed without a corresponding handover event?"
        - "Which slices are experiencing buffer bloat across multiple UEs simultaneously?"
        - "Are there any timestamps where ta_attach_diverge = 1 outside of known DRX windows?"
        - "Which UEs have ul_bler spiking while their dl_bler remains normal?"
        
    Strict constraint: 
        - Analyse the analyst's overall goal before answering the question.
        - Use ONLY the dataset provided, the feature list is available in the file input along with the description of each feature.
    __INPUT__
    analyst goal: {analyst_goal}
    feature list: {feature_list}

    __OUTPUT__
    Answer:

    """,
    input_variables=["analyst_goal", "feature_list"]
)


report_prompt = PromptTemplate(
    template="""
    You are an assistant to the analyst agent. According to the analyst agent's answer, follow the incident template attached to report a structured incident report.

    Strict constraints:
        - Follow the report structure provided. Do not use another one.
        - Do not speculate beyond the input provided. 
        - After finishing the report, state your confidence level.
    
    __INPUT__
    Analyst agent answer : {main_agent}
    Report template: {report_template}
    
    
    __OUPUT__
    Structured report in .txt file: 
    """,
    input_variables=["main_agent", "report_template"]
)



class MessageState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: str
    feature_list: str
    report_template: str


def agent_node(state: MessageState):
    user_goal = state["messages"][0].content
    feature_list = state.get("feature_list")
    report_template = state.get("report_template")

    analyst_output = main_instruction | llm | (lambda x: x.content)
    report = report_prompt | llm | (lambda x: x.content)

    start = time.time()
    analyst = analyst_output.invoke({"analyst_goal": user_goal, "feature_list": feature_list})
    print("analyst: ", (time.time() - start) / 60)
    start = time.time()
    reporter = report.invoke({"main_agent": analyst_output, "report_template": report_template})
    print("report: ", (time.time() - start) / 60)

    state["report"] = reporter
    return {"messages": [AIMessage(content=analyst)], "report": [AIMessage(content=reporter)]}

def build_agent():
    workflow = StateGraph(MessageState)
    workflow.add_node("agent", agent_node, cache_policy=CachePolicy())
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    return workflow.compile(checkpointer=memory)
# agent = build_agent()
# async def result():
#    result_proposal = await agent.ainvoke(
#                        {"messages": "Are there any UEs showing abnormal downlink block error rates right now?"},
#                        config={"configurable": {"thread_id": "session2"}}
#                    )
#    return result_proposal
# asyncio.run(result())

# async def run_conversation_proposal(agent, user_goal):
#     inputs = {"user_goal": user_goal}
#     async for output in agent.astream(inputs):
#         for key, value in output.items():
#             pprint(f"Finished running: {key}:")
#     pprint(value["proposed_schema"])
#     json_schema_output = value['proposed_schema']["json_schema"]
#     return json_schema_output