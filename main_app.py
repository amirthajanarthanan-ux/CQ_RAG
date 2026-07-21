import streamlit as st

st.set_page_config(
    page_title="Knowledge Bot",
    layout="wide"
)
import os
import uuid
import pandas as pd
from dotenv import load_dotenv
from config import config 

from config import config
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph

# LangGraph and LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
# Updated (non-deprecated) embeddings


import config as cfg

print("Config file loaded from:")
print(cfg.__file__)

print("Config object:")
print(config)

print("Available attributes:")
print(dir(config))

# Load environment variables
load_dotenv()

# Ensure GROQ API Key is set
if not os.getenv("GROQ_API_KEY"):
    st.warning("⚠️ GROQ_API_KEY not set. Some functionality may fail.")
else:
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# --- Initialize embeddings model ---
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

embeddings = get_embeddings_model()

# --- Initialize Chroma vector store ---
@st.cache_resource
def get_vector_store(_embed_func):
    # Leading underscore avoids Streamlit caching/serialization issues
    return Chroma(
        persist_directory=config.CHROMA_PERSIST_DIRECTORY,
        embedding_function=_embed_func
    )

vectordb = get_vector_store(embeddings)

# --- Initialize ChatGroq model ---
@st.cache_resource
def get_chat_model():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=400
    )

model = get_chat_model()

# --- LangGraph node function ---
def call_model(state: MessagesState):
    system_prompt = (
        "You are an assistant for academic question-answering tasks. "
        "Use the retrieved context to answer concisely (max 3 sentences). "
        "If unsure, say 'I don’t know'."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": response}

# --- Build and compile the LangGraph workflow ---
@st.cache_resource
def get_langgraph_app():
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

app = get_langgraph_app()

# --- Streamlit UI Setup ---

st.title("Clear_Quote Bot 🚘")

# --- Session State ---
if "threads" not in st.session_state:
    st.session_state.threads = {}  # thread_id → list of messages
if "active_thread" not in st.session_state:
    st.session_state.active_thread = str(uuid.uuid4())
if st.session_state.active_thread not in st.session_state.threads:
    st.session_state.threads[st.session_state.active_thread] = []

# --- Sidebar ---
with st.sidebar:
    st.header("📂 Documents & Chats")

    # Upload PDFs
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        save_path = os.path.join(config.PDF_SOURCE_DIRECTORY, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {uploaded_file.name}")
        st.info("👉 Re-run ingestion separately to add this file to ChromaDB.")

    # Chat thread dropdown
    thread_ids = list(st.session_state.threads.keys())
    selected_thread = st.selectbox(
        "Select chat session",
        thread_ids,
        index=thread_ids.index(st.session_state.active_thread)
    )
    st.session_state.active_thread = selected_thread

    # Start new chat
    if st.button("➕ New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.threads[new_id] = []
        st.session_state.active_thread = new_id
        st.rerun()

# --- Active chat history ---
messages = st.session_state.threads[st.session_state.active_thread]

# --- Display previous messages ---
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat input ---
if prompt := st.chat_input("Ask a question..."):
    # Add user message
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Retrieve top 3 relevant docs
                docs = vectordb.similarity_search_with_score(prompt, k=3)
                _docs = pd.DataFrame(
                    [
                        (
                            prompt,
                            doc[0].page_content,
                            doc[0].metadata.get("source"),
                            doc[0].metadata.get("page"),
                            doc[1]
                        )
                        for doc in docs
                    ],
                    columns=["query", "paragraph", "document", "page_number", "relevant_score"]
                )

                current_context = "\n\n".join(_docs["paragraph"])

                # Construct HumanMessage
                current_turn_message = HumanMessage(
                    content=f"Context: {current_context}\n\nQuestion: {prompt}"
                )

                # Call LangGraph app
                result = app.invoke(
                    {"messages": [current_turn_message]},
                    config={"configurable": {"thread_id": st.session_state.active_thread}},
                )
                ai_response = result["messages"][-1].content

                # Extract metadata
                source_doc = _docs["document"][0] if not _docs.empty else "N/A"
                page_nums = _docs["page_number"].drop_duplicates().head(3).astype(str).tolist()
                page_str = ", ".join(page_nums) if page_nums else "N/A"

                final_response = (
                    f"{ai_response}\n\n**Source**: {source_doc}\n**Pages**: {page_str}"
                )

                st.markdown(final_response)
                messages.append({"role": "assistant", "content": final_response})

            except Exception as e:
                st.error(f"Error: {e}")
                messages.append(
                    {"role": "assistant", "content": "⚠️ I encountered an error. Please try again."}
                )
