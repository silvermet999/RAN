import asyncio
import io
import json
import re
import time
from typing import Annotated, Any, Dict, List

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph, add_messages
from langchain_core.runnables import RunnableSequence
from langgraph.types import CachePolicy
import getpass
import os
from langchain_anthropic import ChatAnthropic
from typing_extensions import Annotated

memory = MemorySaver()

llm = ChatOllama(
    model="CyberCrew/notmythos-8b:latest",
    base_url="http://localhost:11434",
    temperature=0.7,
)


# if "ANTHROPIC_API_KEY" not in os.environ:
#     os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter your Anthropic API key: ")
#
#
# llm_claude = ChatAnthropic(model="claude-haiku-4-5-20251001")


# Answer the following questions to determine if there is an attack:
#         - "Are there any UEs showing abnormal downlink block error rates right now?"
#         - "Which base stations have the largest gap between requested and granted PRBs?"
#         - "Are there any UEs whose ul_ta value changed without a corresponding handover event?"
#         - "Which slices are experiencing buffer bloat across multiple UEs simultaneously?"
#         - "Are there any timestamps where ta_attach_diverge = 1 outside of known DRX windows?"
#         - "Which UEs have ul_bler spiking while their dl_bler remains normal?"

main_instruction = PromptTemplate(template="""
    You are an expert in cybersecurity, specifically network intrusion detection. Assist a cybersecurity analyst in identifying network attacks from the dataset provided.
    
    __INPUT__
    analyst goal: {analyst_goal}
    dataset: {dataset}

    __OUTPUT__
    Answer:

    """,
    input_variables=["analyst_goal", "dataset"]
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
    dataset: str
    report: str


def analyst_node(state: MessageState):
    user_goal = state["messages"][-1].content
    # feature_list = state.get("feature_list")
    dataset_csv = state.get("dataset")

    if dataset_csv:
        dataset = pd.read_csv(io.StringIO(dataset_csv))
        dataset_str = (
            f"Shape: {dataset.shape}\n\n"
            f"Columns: {list(dataset.columns)}\n\n"
            f"Preview:\n{dataset.to_markdown(index=False)}\n\n"
            f"Summary stats:\n{dataset.describe().to_markdown()}"
        )
    else:
        dataset_str = "No dataset provided."

    analyst_chain = main_instruction | llm | (lambda x: x.content)

    start = time.time()
    analyst = analyst_chain.invoke({
        "analyst_goal": user_goal,
        # "feature_list": feature_list,
        "dataset": dataset_str,
    })
    print("analyst: ", (time.time() - start) / 60)
    print("analyst output:", repr(analyst))

    return {"messages": [AIMessage(content=analyst)]}


# def reporter_node(state: MessageState):
#     report_template = state.get("report_template")
#     analyst_output = state["messages"][-1].content
#
#     report_chain = report_prompt | llm | (lambda x: x.content)
#
#     start = time.time()
#     reporter = report_chain.invoke({
#         "main_agent": analyst_output,
#         "report_template": report_template,
#     })
#
#     print("report: ", (time.time() - start) / 60)
#     print("reporter output:", repr(reporter))
#
#     return {"messages": [AIMessage(content=reporter)]}


def build_agent():
    workflow = StateGraph(MessageState)
    workflow.add_node("analyst", analyst_node)
    # workflow.add_node("reporter", reporter_node)
    workflow.add_edge(START, "analyst")
    workflow.add_edge("analyst", END)
    # workflow.add_edge("analyst", "reporter")
    # workflow.add_edge("reporter", END)
    return workflow.compile(checkpointer=memory)


with open('/home/silver/PycharmProjects/RAN2/LLM-MAS/feature_definitions.txt', 'r') as f:
    feature_list = f.read()

with open('/home/silver/PycharmProjects/RAN2/LLM-MAS/report_template.txt', 'r') as f:
    report_template = f.read()

dataset = pd.read_csv("/home/silver/PycharmProjects/RAN2/models/high_conf_attacks_epoch9.csv")
dataset_csv = dataset.to_csv(index=False)

agent = build_agent()
def result():
    result_proposal = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Analyze the provided dataset and determine whether there are any UEs showing abnormal downlink block error rates."
                )
            ],
            # "feature_list": feature_list,
            # "report_template": report_template,
            "dataset": dataset_csv,
        },
        config={"configurable": {"thread_id": "session2"}},
    )

    print("\n=== FINAL MESSAGES ===")
    for message in result_proposal["messages"]:
        print(type(message).__name__)
        print(message.content)
        print()

    return result_proposal
result()