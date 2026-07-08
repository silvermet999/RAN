import asyncio
import io
import json
import time
import pandas as pd
import streamlit as st
from agents import build_agent
from langchain_core.messages import AIMessage, HumanMessage


@st.cache_resource
def get_orchistrator_agent():
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
        for msg in st.session_state["orch_state"]["messages"]:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.write(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.write(msg.content)
        st.session_state["last_processed"] = len(st.session_state["orch_state"]["messages"])



def conversation_function(submitted, user_input, uploaded_file, start_node):
    if submitted and user_input:
        if uploaded_file:
            uploaded_file.seek(0)
            content = uploaded_file.read().decode("utf-8")
            try:
                data = json.loads(content)
                if not isinstance(data, list):
                    data = [data]
                
            except json.JSONDecodeError:
                uploaded_file.seek(0)
                data = [json.loads(line.decode("utf-8").strip()) for line in uploaded_file if line.strip()]
            
            flatten_data = [flatten_json(item) for item in data]
            # st.info("The JSON should be flat to use RAG, click on SAVE to save the flat version. PREVIEW")
            # st.write(flatten_data[:1][:1])
        
        # st.write(flatten_data[:1])
        
            if st.button("-----SAVE-----"):
                st.session_state["new_version"] = True
                save_flat_json(uploaded_file.name)
            else:
                st.session_state["new_version"] = False
            
            allowed_keys = set()
            sample_record = None
            for record in flatten_data:
                if isinstance(record, dict):
                    allowed_keys.update(record.keys())
                    if sample_record is None:
                        sample_record = record

            st.session_state["orch_state"]["json_keys"] = sorted(list(allowed_keys))[:20]
        else:
            st.session_state["orch_state"]["json_keys"] = ["pcap_cnt", "src_ip",
                                                           "dest_ip", "alert_category"]


        
        st.session_state["orch_state"]["start_node"] = start_node
        st.session_state["orch_state"]["messages"].append(HumanMessage(content=user_input))
        print("ui", st.session_state["orch_state"]["messages"])

        # current_input = HumanMessage(content=user_input)
        # print(current_input)
        # st.session_state["orch_state"] = {
        #         "history": st.session_state["orch_state"]["history"],
        #         "messages": current_input,
        #         "schema": st.session_state["orch_state"]["schema"]
        #     }
        display_chat()
            
        with st.spinner("Thinking..."):
            start = time.time()
            result = get_orchistrator_agent().invoke({"messages": st.session_state["orch_state"]["messages"],
                                                    "json_keys": st.session_state["orch_state"]["json_keys"],
                                                    "schema": st.session_state["orch_state"]["schema"],
                                                    "tool_results": st.session_state["orch_state"]["tool_results"],
                                                    "start_node": st.session_state["orch_state"]["start_node"]},
                                    config={"configurable":{"thread_id": "session"}})
            print("orchi time:", (time.time() - start)/60)
            if result["tool_results"] is not None:
                raw = result["tool_results"].content[0].text
                data = []
                try:
                    rag_result = json.loads(raw)

                except json.JSONDecodeError:

                    for line in raw.strip().split("\n"):
                        parts = line.split()

                        if len(parts) == 3:
                            id_val, score, node_name = parts
                        elif len(parts) == 4:
                            id_val, _, score, node_name = parts
                        else:
                            continue

                        data.append([id_val, score, node_name])

                    rag_result = pd.DataFrame(data)
                    st.dataframe(rag_result.head(5))
                st.write(rag_result)
                st.session_state["orch_state"]["tool_results"] = raw
            else:
                st.session_state["orch_state"]["tool_results"] = None

        # ai_msg = result["messages"]
        # st.write(ai_msg.content)

        latest_msg_ai = result["messages"][-1] 
        if isinstance(latest_msg_ai, AIMessage):
            st.session_state["orch_state"]["messages"].append(latest_msg_ai)

        
        # st.session_state["orch_state"]["history"] = st.session_state["orch_state"]["messages"]
        # st.session_state["orch_state"]["history"].extend(result["messages"])

        if result["schema"] is not None:
            if isinstance(result["schema"][0], AIMessage):
                st.session_state["orch_state"]["schema"] = result["schema"][0].content
            else :
                st.session_state["orch_state"]["schema"] = None


    if st.session_state["orch_state"]["schema"] is not None:
        clean_schema = json.loads(st.session_state["orch_state"]["schema"])
        st.subheader("Proposed Schema")
        st.json(clean_schema)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Accept"):
                st.session_state["approved"] = True

        with col2:
            if st.button("❌ Reject"):
                st.session_state["approved"] = False

    if st.session_state.get("approved") is True:
        with open("schema.json", "w") as f:
            json.dump(clean_schema, f, indent=2)

        st.success("Schema saved!")

    elif st.session_state.get("approved") is False:
        st.warning("Schema rejected, defaulting to Suricata schema")
        clean_schema = {
        "packet": {
            "construction_type": "node",
            "source_file": "eve.json",
            "label": "packet",
            "uniqueKey": "pcap_cnt",
            "properties": ["packet", "packet_info_linktype", "event_type"]
        },
        "app": {
            "construction_type": "node",
            "source_file": "eve.json",
            "label": "app",
            "uniqueKey": "app_proto",
            "properties": ["tx_id", "app_version", "app_proto_ts", "app_proto_tc", "event_type"]
        },
        "alert": {
            "construction_type": "node",
            "source_file": "eve.json",
            "label": "alert",
            "uniqueKey": "timestamp",
            "properties": ["timestamp", "alert_signature", "alert_severity", "event_type"]
        },
        
        "SrcIP": {
            "construction_type": "node",
            "source_file": "eve.json",
            "label": "SrcIP",
            "uniqueKey": "src_ip",
            "properties": ["src_ip", "event_type"]
        },
        "SrcPort": {
            "construction_type": "node",
            "source_file": "eve.json",
            "label": "SrcPort",
            "uniqueKey": "src_port",
            "properties": ["src_port", "event_type"]
        },
        "DstIP": {
            "construction_type": "node",
            "source_file": "eve.json",
            "label": "DstIP",
            "uniqueKey": "dest_ip",
            "properties": ["dest_ip", "event_type"]
        },
        "DstPort": {
            "construction_type": "node",
            "source_file": "eve.json",
            "label": "DstPort",
            "uniqueKey": "dest_port",
            "properties": ["dest_port", "event_type"]
        },
        "proto": {
            "construction_type": "node",
            "source_file": "eve.json",
            "label": "proto",
            "uniqueKey": "proto",
            "properties": ["proto", "event_type"]
        },
        "category": {
            "construction_type": "node",
            "source_file": "eve.json",
            "label": "category",
            "uniqueKey": "alert_category",
            "properties": ["alert_signature", "alert_severity", "event_type"]
        },
        "payload": {
            "construction_type": "node",
            "source_file": "eve.json",
            "label": "payload",
            "uniqueKey": "tls_fingerprint",
            "properties": ["http_status", "tls_fingerprint", "smtp_helo", "email_status", "event_type"]
        },
        "Triggered_by": {
            "construction_type": "relationship",
            "source_file": "eve.json",
            "relationship_type": "Triggered_by",
            "from_node_label": "alert",
            "from_node_key": "timestamp",
            "to_node_label": "packet",
            "to_node_key": "pcap_cnt",
            "properties": ["event_type"]
        },
        "Has_app": {
            "construction_type": "relationship",
            "source_file": "eve.json",
            "relationship_type": "Has_app",
            "from_node_label": "packet",
            "from_node_key": "pcap_cnt",
            "to_node_label": "app",
            "to_node_key": "app_proto",
            "properties": ["event_type"]
        },
        "Has_payload": {
            "construction_type": "relationship",
            "source_file": "eve.json",
            "relationship_type": "Has_payload",
            "from_node_label": "app",
            "from_node_key": "app_proto",
            "to_node_label": "payload",
            "to_node_key": "tls_fingerprint",
            "properties": ["event_type"]
        },
        "Has_SrcIP": {
            "construction_type": "relationship",
            "source_file": "eve.json",
            "relationship_type": "Has_SrcIP",
            "from_node_label": "packet",
            "from_node_key": "pcap_cnt",
            "to_node_label": "SrcIP",
            "to_node_key": "src_ip",
            "properties": ["event_type"]
        },
        "Has_SrcPort": {
            "construction_type": "relationship",
            "source_file": "eve.json",
            "relationship_type": "Has_SrcPort",
            "from_node_label": "packet",
            "from_node_key": "pcap_cnt",
            "to_node_label": "SrcPort",
            "to_node_key": "src_port",
            "properties": ["event_type"]
        },
        "Has_DstIP": {
            "construction_type": "relationship",
            "source_file": "eve.json",
            "relationship_type": "Has_DstIP",
            "from_node_label": "packet",
            "from_node_key": "pcap_cnt",
            "to_node_label": "DstIP",
            "to_node_key": "dest_ip",
            "properties": ["event_type"]
        },
        "Has_DstPort": {
            "construction_type": "relationship",
            "source_file": "eve.json",
            "relationship_type": "Has_DstPort",
            "from_node_label": "packet",
            "from_node_key": "pcap_cnt",
            "to_node_label": "DstPort",
            "to_node_key": "dest_port",
            "properties": ["event_type"]
        },
        "Has_proto": {
            "construction_type": "relationship",
            "source_file": "eve.json",
            "relationship_type": "Has_proto",
            "from_node_label": "packet",
            "from_node_key": "pcap_cnt",
            "to_node_label": "proto",
            "to_node_key": "proto",
            "properties": ["event_type"]
        },
        "Has_category":
        {"construction_type": "relationship",
            "source_file": "eve.json",
            "relationship_type": "Has_category",
            "from_node_label": "alert",
            "from_node_key": "timestamp",
            "to_node_label": "category",
            "to_node_key": "alert_category",
            "properties": ["event_type"]}
    }
        with open("schema.json", "w") as f:
            json.dump(clean_schema, f, indent=2)
            
    display_chat()
    

with st.form("input_form", clear_on_submit=True):
    user_input = st.text_input("For optimal performance, start by 'what is' for intent, 'propose' for schema proposal and 'in the graph' for RAG")
    uploaded_file = st.file_uploader("If proposal, try limiting the file size to five jsonlines for faster process. If retrieval, upload the full file here and in Neo4j import folder (README instructions)", type=["json", "jsonl"])
    nodeslist = ["packet", "alert", "app", "SrcIP", "SrcPort", "DstIP", "DstPort", "proto", "category", "payload"]
    start_node = st.selectbox("Select node if you have a start node for retrieval", nodeslist)
    submitted = st.form_submit_button("Send")

if __name__ == "__main__":
    conversation_function(submitted, user_input, uploaded_file, start_node)