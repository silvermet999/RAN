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
        "feature_list" : None,
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
        # if uploaded_file:
        #     uploaded_file.seek(0)
        #     content = uploaded_file.read().decode("utf-8")
        #
        #     if st.button("-----SAVE-----"):
        #         st.session_state["new_version"] = True
        #         with open(uploaded_file.name, "w", encoding="utf-8") as f:
        #             f.write(content)
        #
        #     else:
        #         st.session_state["new_version"] = False
            
            # allowed_keys = set()
            # sample_record = None
            # for record in content:
            #     if isinstance(record, str):
            #         allowed_keys.update(record.keys())
            #         if sample_record is None:
            #             sample_record = record

        #     st.session_state["agent_state"]["feature_list"] =
        # else:
        with open('feature_definitions.txt', 'r') as f:
            print("def")
            st.session_state["agent_state"]["feature_list"] = f.read()

        with open('report_template.txt', 'r') as f:
            print("template")
            st.session_state["agent_state"]["report_template"] = f.read()

        st.session_state["agent_state"]["messages"].append(HumanMessage(content=user_input))
        print("ui", st.session_state["agent_state"]["messages"])

        # current_input = HumanMessage(content=user_input)
        # print(current_input)
        # st.session_state["agent_state"] = {
        #         "history": st.session_state["agent_state"]["history"],
        #         "messages": current_input,
        #         "schema": st.session_state["agent_state"]["schema"]
        #     }
        display_chat()
            
        with st.spinner("Thinking..."):
            start = time.time()
            result = get_agent().invoke({"messages": st.session_state["agent_state"]["messages"],
                                                    "feature_list": st.session_state["agent_state"]["feature_list"],
                                                    "report_template": st.session_state["agent_state"]["report_template"],
                                                    },
                                    config={"configurable":{"thread_id": "session"}})
            print("result: ", (time.time() - start)/60)

        latest_msg_ai = result["messages"][-1] 
        if isinstance(latest_msg_ai, AIMessage):
            st.session_state["agent_state"]["messages"].append(latest_msg_ai)

        
        # st.session_state["agent_state"]["history"] = st.session_state["agent_state"]["messages"]
        # st.session_state["agent_state"]["history"].extend(result["messages"])

        if result["report_template"] is not None:
            if isinstance(result["report_template"][0], AIMessage):
                st.session_state["agent_state"]["report_template"] = result["report_template"][0].content
                with open("final_report.txt", "w", encoding="utf-8") as f:
                    f.write(st.session_state["agent_state"]["report_template"])
            else :
                st.session_state["agent_state"]["report_template"] = None
            
    display_chat()
    

with st.form("input_form", clear_on_submit=True):
    user_input = st.text_input("input question")
    # uploaded_file = st.file_uploader("If proposal, try limiting the file size to five jsonlines for faster process. If retrieval, upload the full file here and in Neo4j import folder (README instructions)", type=["json", "jsonl"])
    submitted = st.form_submit_button("Send")

if __name__ == "__main__":
    conversation_function(submitted, user_input)