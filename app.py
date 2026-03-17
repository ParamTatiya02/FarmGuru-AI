from secret_key import sarvam_ai_key
import os
import json
import streamlit as st
from datetime import datetime
from pdf_reader import PDFReader
from rag_pipeline import RAGPipeline
from llm_client import LLMClient

# -------- INITIALIZE -------- #
os.environ['SARVAM_API_KEY'] = sarvam_ai_key

@st.cache_resource
def init_components():
    reader = PDFReader("data")
    rag = RAGPipeline()
    llm = LLMClient()
    return reader, rag, llm

reader, rag, llm = init_components()

CACHE_DIR = "cache"
PROCESSED_LOG = os.path.join(CACHE_DIR, "processed.json")

# -------- CACHE HELPERS -------- #
def load_processed_log() -> dict:
    """Load the log of already processed PDFs."""
    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_processed_log(log: dict):
    """Save updated log to disk."""
    with open(PROCESSED_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def get_or_cache_text(filename: str) -> str:
    """Read from cache if available, else extract from PDF and save."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    name = os.path.splitext(filename)[0]
    cache_path = os.path.join(CACHE_DIR, name + ".txt")

    if os.path.exists(cache_path):
        print(f"⚡ Loading cached text: '{filename}'")
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        print(f"📖 Reading PDF for first time: '{filename}'")
        text = reader.read_pdf(os.path.join("data", filename))
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"💾 Cached: '{filename}'")
        return text

# -------- LOAD / BUILD VECTOR DB -------- #
@st.cache_resource
def load_vector_db():
    os.makedirs(CACHE_DIR, exist_ok=True)
    processed_log = load_processed_log()

    # Get all PDFs in data folder
    all_pdfs = [f for f in os.listdir("data") if f.endswith(".pdf")]

    # Find only NEW unprocessed PDFs
    new_pdfs = [f for f in all_pdfs if f not in processed_log]

    if not new_pdfs:
        # ⚡ Nothing new — just load existing vectorDB
        print("⚡ All PDFs already processed, loading vectorDB...")
        return rag.load_vector_db()

    # 🔨 Process only new PDFs
    print(f"🔨 Found {len(new_pdfs)} new PDF(s) to process: {new_pdfs}")
    all_new_chunks = []

    for filename in new_pdfs:
        print(f"\n📂 Processing: {filename}")
        text = get_or_cache_text(filename)
        chunks = rag.create_chunks(text, source=filename)
        all_new_chunks.extend(chunks)
        processed_log[filename] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load existing vectorDB and ADD new chunks, or create fresh
    if os.path.exists("vectorDB"):
        print("📂 Adding new chunks to existing vectorDB...")
        vectorstore = rag.load_vector_db()
        vectorstore.add_documents(all_new_chunks)
        vectorstore.save_local("vectorDB")
    else:
        print("🆕 Creating fresh vectorDB...")
        vectorstore = rag.create_vector_db(all_new_chunks)

    # Save updated log
    save_processed_log(processed_log)
    print(f"✅ Done! Total PDFs processed: {len(processed_log)}")
    return vectorstore

vectorstore = load_vector_db()

# -------- SIDEBAR -------- #
with st.sidebar:
    st.title("🌱 FarmGuru")
    st.markdown("Ask questions about farming in **English, Hindi or Marathi**")
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Powered by Sarvam AI")

# -------- MAIN UI -------- #
st.title("🌱 FarmGuru Chatbot")

# Initialize chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "source" in message:
            st.caption(f"📚 Source: {message['source']}")

# -------- HANDLE INPUT -------- #
user_input = st.chat_input("Ask anything about Farming...")

if user_input:

    # Show & save user message first
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.spinner("🌱 Thinking..."):

        if llm.is_farming_question(user_input):
            # -------- FARMING QUESTION → RAG -------- #
            context = rag.query(vectorstore, user_input)
            answer = llm.ask(context, user_input)

            docs = vectorstore.similarity_search(user_input, k=3)
            sources = set(doc.metadata.get("source", "unknown") for doc in docs)
            source_text = ", ".join(sources)

            with st.chat_message("assistant"):
                st.write(answer[7:])
                st.caption(f"📚 Source: {source_text}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "source": source_text
            })

        else:
            # -------- CASUAL QUESTION → DIRECT ANSWER -------- #
            answer = llm.casual_chat(user_input)

            st.chat_message("assistant").write(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
            