import os
import gc
import tempfile
import uuid
import pandas as pd
import io

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from llama_index.core import PromptTemplate
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import Document  # Needed to create a Document for CSV

import streamlit as st

# -----------------------------
# Creative Sidebar & UI Styling
# -----------------------------
st.set_page_config(page_title="RAG with Llama & Dockling", layout="wide")

# Add a colorful sidebar header with emoji
st.sidebar.markdown(
    """
    <div style="text-align: center;">
      <h2 style="color: #4B8BBE;">🤖 Llama RAG Config</h2>
      <p style="font-size: 14px;">Customize your API & upload your documents!</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Session & Utility Setup
# -----------------------------
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

@st.cache_resource
def load_llm(api_key: str):
    # Create an LLM instance using the provided OpenRouter API key.
    llm = Ollama(model="mistral", request_timeout=60.0)
    return llm

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_file(file):
    st.markdown("### 📄 File Preview")
    file_extension = os.path.splitext(file.name)[1].lower()
    try:
        if file_extension == ".csv":
            # For CSV files, use pandas to display
            file.seek(0)  # Reset stream pointer
            df = pd.read_csv(file)
            st.dataframe(df)
        else:
            # For Excel files, use pandas for preview
            file.seek(0)
            df = pd.read_excel(file)
            st.dataframe(df)
    except Exception as e:
        st.error(f"Error displaying file: {e}")

# -----------------------------
# File Uploader & Indexing Logic
# -----------------------------
with st.sidebar:
    st.markdown("---")
    st.header("📂 Upload Documents")
    # Accept Excel and CSV files
    uploaded_file = st.file_uploader("Choose a file (.xlsx, .xls, or .csv)", type=["xlsx", "xls", "csv"])

    if uploaded_file:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        file_key = f"{session_id}-{uploaded_file.name}"
        st.info("Indexing your document...")

        if file_key not in st.session_state.get('file_cache', {}):
            try:
                if file_extension == ".csv":
                    # For CSV: use pandas to read the CSV and convert to a Document.
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)
                    # Convert the CSV content to a string
                    csv_text = df.to_csv(index=False)
                    docs = [Document(text=csv_text)]
                else:
                    # For Excel files (.xlsx, .xls): use pandas to read and convert to Document
                    uploaded_file.seek(0)
                    df = pd.read_excel(uploaded_file)
                    # Convert the Excel content to a string
                    excel_text = df.to_csv(index=False)
                    docs = [Document(text=excel_text)]

                # Setup LLM & embedding model
                llm = Ollama(model="mistral", request_timeout=60.0)
                # Use default embedding model
                Settings.embed_model = None  # Let llama-index use its default embedding model
                node_parser = MarkdownNodeParser()
                index = VectorStoreIndex.from_documents(documents=docs, transformations=[node_parser], show_progress=True)

                # Create the query engine with streaming responses
                Settings.llm = llm
                query_engine = index.as_query_engine(streaming=True)

                # ====== Customize prompt template ======
                qa_prompt_tmpl_str = (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above, think step by step to answer the query in a highly precise and crisp manner focused on the final answer. If unsure, say 'I don't know!'.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                )
                qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
                query_engine.update_prompts(
                    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                )

                st.session_state.file_cache[file_key] = query_engine

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()
        else:
            query_engine = st.session_state.file_cache[file_key]

        st.success("✅ Ready to Chat!")
        display_file(uploaded_file)

# -----------------------------
# Main Chat Interface
# -----------------------------
col1, col2 = st.columns([6, 1])

with col1:
    st.markdown(
        """
        <h1 style="color: #4B8BBE; text-align: center;">🤖 RAG over Documents</h1>
        <h3 style="text-align: center;">Using Dockling & Llama-3.3-70b-instruct</h3>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.button("Clear ↺", on_click=reset_chat)

if "messages" not in st.session_state:
    reset_chat()

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input and process the query
if prompt := st.chat_input("What's your question? 💬"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        streaming_response = query_engine.query(prompt)
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response}) 