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

def get_or_cache_text(filename: str) -> tuple:
    """Read from cache if available, else extract from PDF and save."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    name = os.path.splitext(filename)[0]
    cache_path = os.path.join(CACHE_DIR, name + ".txt")

    if os.path.exists(cache_path):
        print(f"⚡ Loading cached text: '{filename}'")
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read(), os.path.basename(cache_path)
    else:
        print(f"📖 Reading PDF for first time: '{filename}'")
        text = reader.read_pdf(os.path.join("data", filename))
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"💾 Cached: '{filename}'")
        return text, os.path.basename(cache_path)

# -------- PROCESS UPLOADED PDF -------- #
def process_new_pdf(filename: str, progress=None) -> bool:
    try:
        if progress: progress.progress(10, text="📖 Extracting text from PDF...")
        text, cache_name = get_or_cache_text(filename)

        if progress: progress.progress(50, text="✂️ Creating chunks...")
        chunks = rag.create_chunks(text, source=cache_name)

        if progress: progress.progress(75, text="🧠 Adding to knowledge base...")
        
        # ✅ Use existing vectorDB logic instead of get_vector_db
        global vectorstore
        if os.path.exists("vectorDB"):
            vectorstore = rag.load_vector_db()
            vectorstore.add_documents(chunks)
            vectorstore.save_local("vectorDB")
        else:
            vectorstore = rag.create_vector_db(chunks)

        if progress: progress.progress(90, text="💾 Saving progress log...")
        processed_log = load_processed_log()
        processed_log[filename] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_processed_log(processed_log)

        if progress: progress.progress(100, text="✅ Done!")
        return True

    except Exception as e:
        print(f"❌ Failed to process '{filename}': {e}")
        return False

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
        text, cache_name = get_or_cache_text(filename)      # ✅ unpack tuple
        chunks = rag.create_chunks(text, source=cache_name) # ✅ use cache_name
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

    # -------- PDF UPLOAD -------- #
    st.subheader("📁 Upload Farming PDF")
    uploaded_file = st.file_uploader(
        "Upload a new farming PDF",
        type=["pdf"],
        help="PDF will be saved, cached and added to the knowledge base"
    )

    if uploaded_file is not None:
        processed_log = load_processed_log()

        if uploaded_file.name in processed_log:
            st.info(f"✅ '{uploaded_file.name}' is already in the knowledge base!")
        else:
            if st.button("➕ Add to Knowledge Base"):
                os.makedirs("data", exist_ok=True)
                save_path = os.path.join("data", uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                 # ✅ Create progress bar and pass it
                progress = st.progress(0, text="🚀 Starting...")
                success = process_new_pdf(uploaded_file.name, progress)

                if success:
                    st.success(f"✅ '{uploaded_file.name}' added to knowledge base!")
                else:
                    progress.empty()   # ✅ Hide progress bar on failure
                    st.error("❌ Failed to process. Try again.")
                
                with st.spinner(f"📖 Processing '{uploaded_file.name}'... this may take a while"):
                    success = process_new_pdf(uploaded_file.name)

                if success:
                    st.success(f"✅ '{uploaded_file.name}' added to knowledge base!")
                else:
                    st.error("❌ Failed to process. Please try again.")

    st.divider()

    # -------- KNOWLEDGE BASE -------- #
    processed_log = load_processed_log()
    if processed_log:
        st.subheader("📚 Knowledge Base")
        for pdf_name in processed_log:
            st.caption(f"✅ {pdf_name}")

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
            