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
    temperature=0.1,
    num_ctx=8192,
)


# if "ANTHROPIC_API_KEY" not in os.environ:
#     os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter your Anthropic API key: ")
#
# llm_claude = ChatAnthropic(model="claude-haiku-4-5-20251001")



main_instruction = PromptTemplate(template="""
    You are an expert in identifying and classifying traffic in RAN datasets.

    Strict grounding rule: only refer to column names, values, and facts that
    literally appear in the "Dataset traffic sample" section below. Never invent
    or assume the existence of a column, field, or value that is not shown
    there. If the data needed to answer is not present, say so explicitly
    instead of guessing.

    Reference rules (use only what is relevant to the question below):
    - If the question is about the type of traffic, map each value to its attack:
        * 0 -> Constant bitrate traffic
        * 1 -> Poisson traffic (30 pkt/s of 125 bytes per UE)
        * 2 -> Poisson traffic (10 pkt/s of 125 bytes per UE)
    - If the question is about a new traffic pattern, refer to the OOD score in the dataset.
        * If the OOD score is below 0.4, the attack is new (OOD-like).
        * If the OOD score is over 0.8, the attack is known (ID-like).

    Dataset traffic sample (includes precomputed ground-truth facts — use these
    numbers directly, do not try to count or estimate frequencies yourself
    from the row sample):
    {dataset}

    __QUESTION TO ANSWER__
    {question}

    Answer the question above directly and specifically. Do not restate the rules;
    apply them to answer only what was asked.

    Answer:
    """,
    input_variables=["question", "dataset"])


report_prompt = PromptTemplate(
    template="""
    You are an assistant to the analyst agent. Using the analyst agent's answer, 
    write a structured report
    following the report template provided.

    Strict constraints:
        - Follow the report structure provided. Do not use another one.
        - Do not speculate beyond the input provided.
        - After finishing the report, state your confidence level.

    __INPUT__
    Analyst agent answer: {main_agent}

    Report template: {report_template}

    __OUTPUT__
    Structured report:
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


def _find_column(columns, keywords):
    lowered = {c: c.lower() for c in columns}
    for kw in keywords:
        for col, low in lowered.items():
            if kw in low:
                return col
    return None


TRAFFIC_LABELS = {
    0: "eMBB: Constant bitrate traffic",
    1: "mMTC: Poisson traffic (30 pkt/s of 125 bytes per UE)",
    2: "URLLC: Poisson traffic (10 pkt/s of 125 bytes per UE)",
}



def _compute_grounding_facts(dataset: pd.DataFrame) -> str:
    facts = []

    traffic_col = _find_column(dataset.columns, ["label"])
    if traffic_col is not None:
        counts = dataset[traffic_col].value_counts(dropna=False)
        total = int(counts.sum())
        lines = [f"Column used for traffic type: '{traffic_col}'", "Value counts:"]
        for value, count in counts.items():
            pct = 100 * count / total
            label = TRAFFIC_LABELS.get(value, "unknown/unmapped value")
            lines.append(f"  - value={value} ({label}): {count} rows ({pct:.1f}%)")
        most_common_value = counts.idxmax()
        lines.append(
            f"Most common value: {most_common_value} "
            f"-> {TRAFFIC_LABELS.get(most_common_value, 'unknown/unmapped value')}"
        )
        facts.append("\n".join(lines))
    else:
        facts.append(
            "No column matching label was found; "
            "cannot compute malware-type frequency."
        )

    ood_col = _find_column(dataset.columns, ["ood_score"])
    if ood_col is not None:
        ood = dataset[ood_col]
        below = int((ood < 0.4).sum())
        above = int((ood > 0.8).sum())
        facts.append(
            f"OOD score column: '{ood_col}'. "
            f"min={ood.min():.3f}, max={ood.max():.3f}, mean={ood.mean():.3f}. "
            f"Rows with OOD < 0.4 (new/OOD-like): {below}. "
            f"Rows with OOD > 0.8 (known/ID-like): {above}."
        )

    return "\n\n".join(facts)


_TRAFFIC_TYPE_QUESTION_RE = re.compile(
    r"(common|most\s+frequent|which|what)\s+.*(type|kind)s?\s+of\s+(traffic|attack)", re.IGNORECASE
)

_OOD_QUESTION_RE = re.compile(
    r"(known|novel|new|unknown|ood|out.of.distribution|id.like|ood.like)", re.IGNORECASE
)


def analyst_node(state: MessageState):
    user_goal = state["messages"][-1].content
    print("USER GOAL", user_goal)
    feature_list = state.get("feature_list", "")
    dataset_csv = state.get("dataset")

    dataset = None
    grounding_facts = None
    if dataset_csv:
        dataset = pd.read_csv(io.StringIO(dataset_csv))
        grounding_facts = _compute_grounding_facts(dataset)
        dataset_str = (
            f"Shape: {dataset.shape}\n\n"
            f"Columns: {list(dataset.columns)}\n\n"
            f"Precomputed facts (treat these as ground truth, do not recompute or "
            f"override them from the row sample below):\n{grounding_facts}\n\n"
            f"Row sample (first 20 rows, for context only):\n"
            f"{dataset.head(20).to_markdown(index=False)}"
        )
    else:
        dataset_str = "No dataset provided."

    if dataset is not None and _TRAFFIC_TYPE_QUESTION_RE.search(user_goal):
        traffic_col = _find_column(dataset.columns, ["label"])
        if traffic_col is not None:
            counts = dataset[traffic_col].value_counts(dropna=False)
            most_common_value = counts.idxmax()
            label = TRAFFIC_LABELS.get(most_common_value, f"unmapped value {most_common_value}")
            total = int(counts.sum())
            pct = 100 * counts[most_common_value] / total
            answer = (
                f"The most common traffic type in the dataset is: {label} "
                f"(value={most_common_value} in column '{traffic_col}'), "
                f"appearing in {int(counts[most_common_value])} of {total} rows ({pct:.1f}%).\n\n"
                f"Full breakdown:\n" + "\n".join(
                    f"  - {TRAFFIC_LABELS.get(v, f'unmapped value {v}')} (value={v}): "
                    f"{int(c)} rows ({100 * c / total:.1f}%)"
                    for v, c in counts.items()
                )
            )
            return {"messages": [AIMessage(content=answer)]}

    if dataset is not None and _OOD_QUESTION_RE.search(user_goal):
        ood_col = _find_column(dataset.columns, ["ood_score"])
        if ood_col is not None:
            ood = dataset[ood_col]
            total = int(ood.shape[0])
            below = int((ood < 0.4).sum())
            above = int((ood > 0.8).sum())
            ambiguous = total - below - above

            mean_score = float(ood.mean())
            if below > above and below > ambiguous:
                verdict = "The dataset is dominated by OOD-like (novel/unknown) samples."
            elif above > below and above > ambiguous:
                verdict = "The dataset is dominated by ID-like (known) samples."
            else:
                verdict = (
                    "The dataset does not have a single dominant category; scores are "
                    "mixed across OOD-like, ID-like, and ambiguous ranges."
                )

            answer = (
                f"OOD score column used: '{ood_col}'. Mean OOD score: {mean_score:.3f}.\n\n"
                f"- OOD-like (score < 0.4, novel/unknown attack): {below} of {total} rows "
                f"({100 * below / total:.1f}%)\n"
                f"- ID-like (score > 0.8, known attack): {above} of {total} rows "
                f"({100 * above / total:.1f}%)\n"
                f"- Ambiguous (0.4 <= score <= 0.8, no confident call): {ambiguous} of {total} rows "
                f"({100 * ambiguous / total:.1f}%)\n\n"
                f"{verdict}"
            )
            return {"messages": [AIMessage(content=answer)]}


    analyst_chain = main_instruction | llm | (lambda x: x.content)

    formatted_prompt = main_instruction.format(
        question=user_goal, feature_list=feature_list, dataset=dataset_str
    )

    approx_tokens = len(formatted_prompt) // 4
    configured_ctx = getattr(llm, "num_ctx", None) or 2048
    if approx_tokens > configured_ctx * 0.8:
        print(
            f"WARNING: prompt is ~{approx_tokens} tokens, close to or over "
            f"num_ctx={configured_ctx}. Ollama will silently truncate context "
            f"if this is exceeded, which can produce hallucinated output with "
            f"no error. Consider raising num_ctx or trimming the row sample."
        )

    start = time.time()
    analyst = analyst_chain.invoke({
        "question": user_goal,
        "dataset": dataset_str,
    })
    print("analyst: ", (time.time() - start) / 60)
    print("analyst output:", repr(analyst))

    if dataset is not None:
        real_cols_lower = {c.lower() for c in dataset.columns}
        mentioned_unknown = [
            w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_ ]{2,}", analyst)
            if w.strip().lower() not in real_cols_lower and w.strip().lower() in
            {"anomaly", "abnormal"}
        ]
        if mentioned_unknown:
            print("WARNING: model output may reference fabricated fields:", mentioned_unknown)

    return {"messages": [AIMessage(content=analyst)]}


def reporter_node(state: MessageState):
    report_template = state.get("report_template", "")
    dataset_csv = state.get("dataset")
    analyst_output = state["messages"][-1].content

    if dataset_csv:
        dataset = pd.read_csv(io.StringIO(dataset_csv))
    else:
        dataset = None

    report_chain = report_prompt | llm | (lambda x: x.content)

    formatted_prompt = report_prompt.format(
        main_agent=analyst_output,
        report_template=report_template,
    )

    start = time.time()
    reporter = report_chain.invoke({
        "main_agent": analyst_output,
        "report_template": report_template,
    })

    print("report: ", (time.time() - start) / 60)

    return {"messages": [AIMessage(content=reporter)], "report": reporter}


def build_agent():
    workflow = StateGraph(MessageState)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("reporter", reporter_node)
    workflow.add_edge(START, "analyst")
    workflow.add_edge("analyst", "reporter")
    workflow.add_edge("reporter", END)
    return workflow.compile(checkpointer=memory)


# with open('/home/silver/PycharmProjects/RAN2/LLM-MAS/feature_definitions.txt', 'r') as f:
#     feature_list = f.read()
#
# with open('/home/silver/PycharmProjects/RAN2/LLM-MAS/report_template.txt', 'r') as f:
#     report_template = f.read()
#
# dataset = pd.read_csv("/home/silver/PycharmProjects/RAN2/models/high_conf_attacks.csv")
# dataset_csv = dataset.to_csv(index=False)
#
# agent = build_agent()
# def result():
#     result_proposal = agent.invoke(
#         {
#             "messages": [
#                 HumanMessage(
#                     content="What type of traffic is common in the dataset?"
#                 )
#             ],
#             "feature_list": feature_list,
#             "report_template": report_template,
#             "dataset": dataset_csv,
#         },
#         config={"configurable": {"thread_id": "session1"}},
#     )
#
#     print("\n=== FINAL MESSAGES ===")
#     for message in result_proposal["messages"]:
#         print(type(message).__name__)
#         print(message.content)
#         print()
#
#     return result_proposal
# result()