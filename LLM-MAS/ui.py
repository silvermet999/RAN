import asyncio
import io
import json
import time
import pandas as pd
import streamlit as st
from agents import build_agent
from langchain_core.messages import AIMessage, HumanMessage


@st.cache_resource
def get_agent():
    return build_agent()


# if "messages" not in st.session_state:
#     st.session_state["messages"] = []
if "agent_state" not in st.session_state:
    st.session_state["agent_state"] = {
        "messages": [],
        "feature_list": None,
        "report_template": None
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


def conversation_function(submitted, user_input):
    if submitted and user_input:

        with open('/home/silver/PycharmProjects/RAN2/LLM-MAS/feature_definitions.txt', 'r') as f:
            st.session_state["agent_state"]["feature_list"] = f.read()

        with open('/home/silver/PycharmProjects/RAN2/LLM-MAS/report_template.txt', 'r') as f:
            st.session_state["agent_state"]["report_template"] = f.read()

        st.session_state["agent_state"]["messages"].append(HumanMessage(content=user_input))
        print("ui", st.session_state["agent_state"]["messages"])

        display_chat()

        with st.spinner("Thinking..."):
            start = time.time()
            result = get_agent().invoke({"messages": st.session_state["agent_state"]["messages"],
                                         "feature_list": st.session_state["agent_state"]["feature_list"],
                                         "report_template": st.session_state["agent_state"]["report_template"]
                                         },
                                        config={"configurable": {"thread_id": "session"}})
            print("result: ", (time.time() - start) / 60)
        latest_msg_ai = result["messages"][-1]
        if isinstance(latest_msg_ai, AIMessage):
            st.session_state["agent_state"]["messages"].append(latest_msg_ai)
            print(st.session_state["agent_state"]["messages"])
        else:
            st.session_state["agent_state"]["messages"].append(
                AIMessage(content="(No response generated for this turn.)")
            )

        display_chat()

        try:
            report_template = result.get("report_template")
            if report_template is not None and isinstance(report_template[0], AIMessage):
                st.session_state["agent_state"]["report_template"] = report_template[0].content
                with open("final_report.txt", "w", encoding="utf-8") as f:
                    f.write(st.session_state["agent_state"]["report_template"])
            else:
                st.session_state["agent_state"]["report_template"] = None
        except Exception as e:
            print("Failed to process report_template:", e)

    display_chat()


with st.form("input_form", clear_on_submit=True):
    user_input = st.text_input("input question")
    # uploaded_file = st.file_uploader("Ask any question")
    submitted = st.form_submit_button("Send")

if __name__ == "__main__":
    conversation_function(submitted, user_input)