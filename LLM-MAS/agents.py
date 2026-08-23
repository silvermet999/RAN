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
    # Ollama defaults to a 2048-token context window regardless of what the
    # underlying model architecture supports, unless this is set explicitly.
    # Without it, larger prompts (e.g. more dataset rows) get silently
    # truncated rather than erroring - which looks exactly like hallucination,
    # since the model may lose either the grounding rules/facts (if truncated
    # from the front) or the question itself (if truncated from the back).
    # Check `ollama show CyberCrew/notmythos-8b:latest` for the model's actual
    # max supported context and set this at or below that ceiling.
    num_ctx=8192,
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

# main_instruction = PromptTemplate(template="""
#     You are an expert in cybersecurity, specifically network intrusion detection. Assist a cybersecurity analyst in identifying network attacks from the dataset provided.
#
#     __INPUT__
#     analyst goal: {analyst_goal}
#     dataset: {dataset}
#
#     __OUTPUT__
#     Answer:
#
#     """,
#     input_variables=["analyst_goal", "dataset"]
# )
main_instruction = PromptTemplate(template="""
    You are an expert in identifying and classifying malware in RAN datasets.

    Strict grounding rule: only refer to column names, values, and facts that
    literally appear in the "Dataset attack sample" section below. Never invent
    or assume the existence of a column, field, or value that is not shown
    there. If the data needed to answer is not present, say so explicitly
    instead of guessing.

    Reference rules (use only what is relevant to the question below):
    - If the question is about the type of malware, map each value to its attack:
        * 0 -> Constant bitrate traffic
        * 1 -> Poisson traffic (30 pkt/s of 125 bytes per UE)
        * 2 -> Poisson traffic (10 pkt/s of 125 bytes per UE)
    - If the question is about a new attack, refer to the OOD score in the dataset.
        * If the OOD score is below 0.4, the attack is new (OOD-like).
        * If the OOD score is over 0.8, the attack is known (ID-like).
    - If the question is about the meaning of a feature, use this feature list:
    {feature_list}

    Dataset attack sample (includes precomputed ground-truth facts — use these
    numbers directly, do not try to count or estimate frequencies yourself
    from the row sample):
    {dataset}

    __QUESTION TO ANSWER__
    {question}

    Answer the question above directly and specifically. Do not restate the rules;
    apply them to answer only what was asked.

    Answer:
    """,
    input_variables=["question", "feature_list", "dataset"])


report_prompt = PromptTemplate(
    template="""
    You are an assistant to the analyst agent. Using the analyst agent's answer
    and the precomputed MITRE mapping below, write a structured incident report
    following the report template provided.

    Strict constraints:
        - Follow the report structure provided. Do not use another one.
        - Do not speculate beyond the input provided.
        - MITRE ATT&CK/FiGHT technique IDs, tactic names, and technique names
          MUST come only from the "Precomputed MITRE mapping" section below.
          Do not invent, guess, or recall from memory any technique ID, tactic,
          or framework name that is not explicitly listed there.
        - {scope_disclaimer}
        - After finishing the report, state your confidence level.

    __INPUT__
    Analyst agent answer: {main_agent}

    Precomputed MITRE mapping (ground truth — narrate around this, do not
    contradict or extend it):
    {mitre_mapping}

    Report template: {report_template}

    __OUTPUT__
    Structured report:
    """,
    input_variables=["main_agent", "mitre_mapping", "report_template", "scope_disclaimer"]
)



class MessageState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: str
    feature_list: str
    report_template: str
    dataset: str
    report: str


def _find_column(columns, keywords):
    """Best-effort match of a column name against a list of keyword substrings."""
    lowered = {c: c.lower() for c in columns}
    for kw in keywords:
        for col, low in lowered.items():
            if kw in low:
                return col
    return None


ATTACK_LABELS = {
    0: "Constant bitrate traffic",
    1: "Poisson traffic (30 pkt/s of 125 bytes per UE)",
    2: "Poisson traffic (10 pkt/s of 125 bytes per UE)",
}

# Deterministic mapping from what this dataset can actually evidence
# (traffic-generation pattern / volumetric behavior) to MITRE techniques.
# This is intentionally narrow: the dataset has no auth, process, or
# cross-host telemetry, so it cannot support any tactic other than Impact
# (denial of service). The LLM must not be allowed to pick technique IDs
# itself - it has no reliable memorized mapping and will fabricate plausible
# ones. Extend this table if the dataset gains richer telemetry.
MITRE_TECHNIQUE_MAPPING = {
    0: {
        "tactic": "N/A",
        "technique_id": None,
        "technique_name": None,
        "framework": None,
        "rationale": (
            "Constant bitrate traffic matches baseline/expected behavior; "
            "no anomalous pattern to map to a technique."
        ),
    },
    1: {
        "tactic": "Impact (TA0040)",
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "framework": "MITRE ATT&CK Enterprise; see also MITRE FiGHT (5G-specific) "
                      "addendum for RAN/gNB resource exhaustion",
        "rationale": (
            "Elevated Poisson-rate traffic (30 pkt/s per UE) is consistent with "
            "a volumetric flood pattern aimed at exhausting network/RAN "
            "bandwidth or scheduling resources."
        ),
    },
    2: {
        "tactic": "Impact (TA0040)",
        "technique_id": "T1499",
        "technique_name": "Endpoint Denial of Service",
        "framework": "MITRE ATT&CK Enterprise; see also MITRE FiGHT (5G-specific) "
                      "addendum for RAN/gNB resource exhaustion",
        "rationale": (
            "Lower-rate but sustained Poisson traffic (10 pkt/s per UE) is "
            "consistent with resource-exhaustion behavior targeting endpoint "
            "processing/connection capacity rather than raw bandwidth."
        ),
    },
}

MITRE_SCOPE_DISCLAIMER = (
    "SCOPE LIMITATION: This dataset contains only per-UE traffic-generation "
    "pattern labels and an OOD novelty score. It contains no authentication, "
    "process, payload, or cross-host telemetry. Therefore only the Impact "
    "tactic (denial-of-service style techniques) can be evidenced from this "
    "data. Any other MITRE ATT&CK/FiGHT tactic (Initial Access, Persistence, "
    "Lateral Movement, Credential Access, Exfiltration, Command and Control, "
    "etc.) CANNOT be assessed from this data and must not be claimed or "
    "implied in the report."
)


def _compute_grounding_facts(dataset: pd.DataFrame) -> str:
    """Compute exact aggregate facts in pandas so the LLM only has to explain
    them, rather than infer counts/most-common-value from a raw row preview
    or describe() output (which it cannot reliably do and will hallucinate)."""
    facts = []

    attack_col = _find_column(dataset.columns, ["attack", "malware", "label", "class", "type"])
    if attack_col is not None:
        counts = dataset[attack_col].value_counts(dropna=False)
        total = int(counts.sum())
        lines = [f"Column used for malware type: '{attack_col}'", "Value counts:"]
        for value, count in counts.items():
            pct = 100 * count / total
            label = ATTACK_LABELS.get(value, "unknown/unmapped value")
            lines.append(f"  - value={value} ({label}): {count} rows ({pct:.1f}%)")
        most_common_value = counts.idxmax()
        lines.append(
            f"Most common value: {most_common_value} "
            f"-> {ATTACK_LABELS.get(most_common_value, 'unknown/unmapped value')}"
        )
        facts.append("\n".join(lines))
    else:
        facts.append(
            "No column matching attack/malware/label/class/type was found; "
            "cannot compute malware-type frequency."
        )

    ood_col = _find_column(dataset.columns, ["ood"])
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


def _compute_mitre_mapping(dataset: pd.DataFrame) -> str:
    """Compute the MITRE ATT&CK/FiGHT technique mapping deterministically from
    the attack-type value counts, rather than letting the LLM choose technique
    IDs itself (which it cannot reliably recall and will fabricate)."""
    attack_col = _find_column(dataset.columns, ["attack", "malware", "label", "class", "type"])
    if attack_col is None:
        return (
            "No attack-type column found in the dataset; no MITRE technique "
            "mapping can be computed."
        )

    counts = dataset[attack_col].value_counts(dropna=False)
    total = int(counts.sum())
    lines = [f"Attack-type column used: '{attack_col}'", ""]
    for value, count in counts.items():
        pct = 100 * count / total
        mapping = MITRE_TECHNIQUE_MAPPING.get(value)
        if mapping is None:
            lines.append(
                f"- value={value}: {count} rows ({pct:.1f}%) — "
                f"no MITRE mapping defined for this value; do not invent one."
            )
            continue
        if mapping["technique_id"] is None:
            lines.append(
                f"- value={value} ({ATTACK_LABELS.get(value, 'unknown')}): "
                f"{count} rows ({pct:.1f}%) — {mapping['rationale']}"
            )
        else:
            lines.append(
                f"- value={value} ({ATTACK_LABELS.get(value, 'unknown')}): "
                f"{count} rows ({pct:.1f}%) -> {mapping['tactic']}, "
                f"{mapping['technique_id']} ({mapping['technique_name']}). "
                f"Framework: {mapping['framework']}. Rationale: {mapping['rationale']}"
            )
    return "\n".join(lines)



_MALWARE_TYPE_QUESTION_RE = re.compile(
    r"(common|most\s+frequent|which|what)\s+.*(type|kind)s?\s+of\s+malware", re.IGNORECASE
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
            f"Row sample (first 10 rows, for context only):\n"
            f"{dataset.head(10).to_markdown(index=False)}"
        )
    else:
        dataset_str = "No dataset provided."

    # Deterministic short-circuit: for the "what type of malware is common"
    # question pattern, answer directly from the pandas-computed facts instead
    # of asking the LLM to summarize/interpret raw rows. Small local models
    # have shown a tendency to invent plausible-sounding but nonexistent
    # column names (e.g. "Oblivion", "Malware Type") instead of using the
    # actual injected data, so for this well-defined question we skip the LLM
    # entirely and guarantee a grounded answer.
    if dataset is not None and _MALWARE_TYPE_QUESTION_RE.search(user_goal):
        attack_col = _find_column(dataset.columns, ["attack", "malware", "label", "class", "type"])
        if attack_col is not None:
            counts = dataset[attack_col].value_counts(dropna=False)
            most_common_value = counts.idxmax()
            label = ATTACK_LABELS.get(most_common_value, f"unmapped value {most_common_value}")
            total = int(counts.sum())
            pct = 100 * counts[most_common_value] / total
            answer = (
                f"The most common malware type in the dataset is: {label} "
                f"(value={most_common_value} in column '{attack_col}'), "
                f"appearing in {int(counts[most_common_value])} of {total} rows ({pct:.1f}%).\n\n"
                f"Full breakdown:\n" + "\n".join(
                    f"  - {ATTACK_LABELS.get(v, f'unmapped value {v}')} (value={v}): "
                    f"{int(c)} rows ({100 * c / total:.1f}%)"
                    for v, c in counts.items()
                )
            )
            print("analyst output (deterministic, no LLM call):", repr(answer))
            return {"messages": [AIMessage(content=answer)]}

    analyst_chain = main_instruction | llm | (lambda x: x.content)

    # Log the exact formatted prompt so it's possible to verify the model is
    # actually receiving the real dataset facts (vs. a template-substitution
    # bug producing an empty/garbled prompt, which would also look like
    # "the model is making things up").
    formatted_prompt = main_instruction.format(
        question=user_goal, feature_list=feature_list, dataset=dataset_str
    )
    print("=== FORMATTED PROMPT SENT TO LLM ===")
    print(formatted_prompt)
    print("=== END FORMATTED PROMPT ===")

    # Rough token estimate (chars/4 is a common approximation for English
    # text) to flag when the prompt is approaching or exceeding num_ctx.
    # This is approximate, not exact - if you need precision, tokenize with
    # the model's actual tokenizer instead.
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
        "feature_list": feature_list,
        "dataset": dataset_str,
    })
    print("analyst: ", (time.time() - start) / 60)
    print("analyst output:", repr(analyst))

    # Sanity check: flag (don't silently trust) any answer that references a
    # column name not present in the dataset - a strong signal of fabrication.
    if dataset is not None:
        real_cols_lower = {c.lower() for c in dataset.columns}
        mentioned_unknown = [
            w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_ ]{2,}", analyst)
            if w.strip().lower() not in real_cols_lower and w.strip().lower() in
            {"oblivion", "infection", "malware type"}  # extend as needed
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
        mitre_mapping = _compute_mitre_mapping(dataset)
    else:
        dataset = None
        mitre_mapping = "No dataset provided; no MITRE mapping can be computed."

    report_chain = report_prompt | llm | (lambda x: x.content)

    formatted_prompt = report_prompt.format(
        main_agent=analyst_output,
        mitre_mapping=mitre_mapping,
        report_template=report_template,
        scope_disclaimer=MITRE_SCOPE_DISCLAIMER,
    )
    print("=== FORMATTED REPORT PROMPT SENT TO LLM ===")
    print(formatted_prompt)
    print("=== END FORMATTED REPORT PROMPT ===")

    start = time.time()
    reporter = report_chain.invoke({
        "main_agent": analyst_output,
        "mitre_mapping": mitre_mapping,
        "report_template": report_template,
        "scope_disclaimer": MITRE_SCOPE_DISCLAIMER,
    })

    print("report: ", (time.time() - start) / 60)
    print("reporter output:", repr(reporter))

    # Sanity check: flag any MITRE technique ID mentioned in the report that
    # wasn't in the precomputed mapping we supplied - a strong signal the LLM
    # invented one instead of using the ground truth it was given.
    allowed_ids = {
        m["technique_id"] for m in MITRE_TECHNIQUE_MAPPING.values() if m["technique_id"]
    }
    mentioned_ids = set(re.findall(r"T\d{4}(?:\.\d{3})?", reporter))
    unauthorized_ids = mentioned_ids - allowed_ids
    if unauthorized_ids:
        print(
            "WARNING: reporter output references MITRE technique IDs not in "
            "the precomputed mapping (likely fabricated):", unauthorized_ids
        )

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
#                     content="What type of malware is common in the dataset?"
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