import streamlit as st

# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Knowledge Bot",
    layout="wide"
)

# ---------------------------------------------------------
# Standard libraries
# ---------------------------------------------------------
import os
import uuid
import pandas as pd

# ---------------------------------------------------------
# Load environment variables FIRST
# ---------------------------------------------------------
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH)

# ---------------------------------------------------------
# Import configuration AFTER loading .env
# ---------------------------------------------------------
from config import config

# ---------------------------------------------------------
# LangChain / LangGraph imports
# ---------------------------------------------------------
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from langchain_core.messages import HumanMessage, SystemMessage


# =========================================================
# GROQ API KEY
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

if not GROQ_API_KEY:
    st.error(
        "❌ GROQ_API_KEY is missing. "
        "Please add your Groq API key to the .env file."
    )
    st.stop()


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME
    )


embeddings = get_embeddings_model()


# =========================================================
# CHROMA VECTOR DATABASE
# =========================================================

@st.cache_resource
def get_vector_store(_embed_func):
    return Chroma(
        persist_directory=config.CHROMA_PERSIST_DIRECTORY,
        embedding_function=_embed_func
    )


vectordb = get_vector_store(embeddings)


# =========================================================
# GROQ CHAT MODEL
# =========================================================

@st.cache_resource
def get_chat_model():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=config.CHAT_MODEL_NAME,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS
    )


model = get_chat_model()


# =========================================================
# LANGGRAPH MODEL NODE
# =========================================================

def call_model(state: MessagesState):

    system_prompt = (
        "You are an assistant for ClearQuote automotive "
        "knowledge-question answering. "
        "Use the retrieved context to answer the user's question. "
        "Answer concisely in a maximum of 3 sentences. "
        "If the answer is not available in the retrieved context, "
        "say 'I don't know based on the provided context.'"
    )

    messages = [
        SystemMessage(content=system_prompt)
    ] + state["messages"]

    response = model.invoke(messages)

    return {
        "messages": [response]
    }


# =========================================================
# LANGGRAPH WORKFLOW
# =========================================================

@st.cache_resource
def get_langgraph_app():

    workflow = StateGraph(
        state_schema=MessagesState
    )

    workflow.add_node(
        "model",
        call_model
    )

    workflow.add_edge(
        START,
        "model"
    )

    memory = MemorySaver()

    return workflow.compile(
        checkpointer=memory
    )


app = get_langgraph_app()


# =========================================================
# STREAMLIT UI
# =========================================================

st.title("Clear_Quote Bot 🚘")


# =========================================================
# SESSION STATE
# =========================================================

if "threads" not in st.session_state:
    st.session_state.threads = {}

if "active_thread" not in st.session_state:
    st.session_state.active_thread = str(uuid.uuid4())

if (
    st.session_state.active_thread
    not in st.session_state.threads
):
    st.session_state.threads[
        st.session_state.active_thread
    ] = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📂 Documents & Chats")

    # -----------------------------------------------------
    # PDF Upload
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file:

        save_path = os.path.join(
            config.PDF_SOURCE_DIRECTORY,
            uploaded_file.name
        )

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        st.info(
            "👉 Run the ingestion process separately "
            "to add this document to ChromaDB."
        )

    # -----------------------------------------------------
    # Chat Sessions
    # -----------------------------------------------------

    thread_ids = list(
        st.session_state.threads.keys()
    )

    selected_thread = st.selectbox(
        "Select chat session",
        thread_ids,
        index=thread_ids.index(
            st.session_state.active_thread
        )
    )

    st.session_state.active_thread = selected_thread

    # -----------------------------------------------------
    # New Chat
    # -----------------------------------------------------

    if st.button("➕ New Chat"):

        new_id = str(uuid.uuid4())

        st.session_state.threads[new_id] = []

        st.session_state.active_thread = new_id

        st.rerun()


# =========================================================
# ACTIVE CHAT HISTORY
# =========================================================

messages = st.session_state.threads[
    st.session_state.active_thread
]


# =========================================================
# DISPLAY PREVIOUS MESSAGES
# =========================================================

for message in messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

if prompt := st.chat_input(
    "Ask a question..."
):

    # -----------------------------------------------------
    # Store user message
    # -----------------------------------------------------

    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    # -----------------------------------------------------
    # Assistant response
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base..."):

            try:

                # =========================================
                # RETRIEVE TOP 3 DOCUMENTS
                # =========================================

                docs = (
                    vectordb
                    .similarity_search_with_score(
                        prompt,
                        k=3
                    )
                )

                # =========================================
                # PREPARE RETRIEVED DOCUMENT DATA
                # =========================================

                _docs = pd.DataFrame(
                    [
                        (
                            prompt,
                            doc[0].page_content,
                            doc[0].metadata.get(
                                "source"
                            ),
                            doc[0].metadata.get(
                                "page"
                            ),
                            doc[1]
                        )
                        for doc in docs
                    ],
                    columns=[
                        "query",
                        "paragraph",
                        "document",
                        "page_number",
                        "relevant_score"
                    ]
                )

                # =========================================
                # BUILD CONTEXT
                # =========================================

                current_context = (
                    "\n\n".join(
                        _docs["paragraph"]
                    )
                    if not _docs.empty
                    else ""
                )

                # =========================================
                # CREATE HUMAN MESSAGE
                # =========================================

                current_turn_message = HumanMessage(
                    content=(
                        f"Retrieved Context:\n"
                        f"{current_context}\n\n"
                        f"Question:\n"
                        f"{prompt}"
                    )
                )

                # =========================================
                # RUN LANGGRAPH
                # =========================================

                result = app.invoke(
                    {
                        "messages": [
                            current_turn_message
                        ]
                    },
                    config={
                        "configurable": {
                            "thread_id":
                                st.session_state.active_thread
                        }
                    }
                )

                # =========================================
                # GET AI RESPONSE
                # =========================================

                ai_response = (
                    result["messages"][-1].content
                )

                # =========================================
                # SOURCE INFORMATION
                # =========================================

                if not _docs.empty:

                    source_doc = (
                        _docs["document"].iloc[0]
                    )

                    page_nums = (
                        _docs["page_number"]
                        .drop_duplicates()
                        .head(3)
                        .astype(str)
                        .tolist()
                    )

                else:

                    source_doc = "N/A"
                    page_nums = []

                page_str = (
                    ", ".join(page_nums)
                    if page_nums
                    else "N/A"
                )

                # =========================================
                # FINAL RESPONSE
                # =========================================

                final_response = (
                    f"{ai_response}\n\n"
                    f"**Source:** {source_doc}\n"
                    f"**Pages:** {page_str}"
                )

                st.markdown(
                    final_response
                )

                # =========================================
                # SAVE ASSISTANT MESSAGE
                # =========================================

                messages.append(
                    {
                        "role": "assistant",
                        "content": final_response
                    }
                )

            except Exception as e:

                st.error(
                    f"❌ Error: {e}"
                )

                messages.append(
                    {
                        "role": "assistant",
                        "content":
                            "⚠️ I encountered an error. "
                            "Please try again."
                    }
                )