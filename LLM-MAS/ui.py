import asyncio
import io
import json
import time
import pandas as pd
import streamlit as st
from agents import build_agent
from langchain_core.messages import AIMessage, HumanMessage


# @st.cache_resource
# def get_agent():
#     return build_agent()


if "agent_state" not in st.session_state:
    st.session_state["agent_state"] = {
        "messages": [],
        "report_template": None,
        "dataset": None
    }

if "last_processed" not in st.session_state:
    st.session_state["last_processed"] = -1

chat_placeholder = st.empty()


def display_chat():
    with chat_placeholder.container():
        for msg in st.session_state["agent_state"]["messages"]:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.write(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.write(msg.content)
        st.session_state["last_processed"] = len(st.session_state["agent_state"]["messages"])


def process_input(user_input):

    with open('/home/silver/PycharmProjects/RAN2/LLM-MAS/report_template.txt', 'r') as f:
        st.session_state["agent_state"]["report_template"] = f.read()

    dataset = pd.read_csv("/home/silver/PycharmProjects/RAN2/models/high_conf_attacks.csv")
    st.session_state["agent_state"]["dataset"] = dataset.to_csv(index=False)

    st.session_state["agent_state"]["messages"].append(HumanMessage(content=user_input))

    with st.spinner("Thinking..."):
        start = time.time()
        result = build_agent().invoke({"messages": st.session_state["agent_state"]["messages"],
                                     "report_template": st.session_state["agent_state"]["report_template"],
                                     "dataset": st.session_state["agent_state"]["dataset"],
                                     },
                                    config={"configurable": {"thread_id": "session2"}})
        print("result: ", (time.time() - start) / 60)

    st.session_state["agent_state"]["messages"] = result["messages"]
    st.session_state["agent_state"]["report"] = result.get("report")


def render_page():
    display_chat()

    report_text = st.session_state["agent_state"].get("report")
    if report_text is not None:
        st.text(report_text)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Accept"):
                st.session_state["approved"] = True

        with col2:
            if st.button("Reject"):
                st.session_state["approved"] = False

    if st.session_state.get("approved") is True and report_text is not None:
        with open("report.txt", "w") as f:
            f.write(report_text)

        st.download_button(
            label="Download report.txt",
            data=report_text,
            file_name="report.txt",
            mime="text/plain",
        )

    with st.form("input_form", clear_on_submit=True):
        user_input = st.text_input("input question")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        process_input(user_input)
        st.rerun()


render_page()