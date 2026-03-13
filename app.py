from secret_key import sarvam_ai_key
import os
import streamlit as st
from pdf_reader import PDFReader
from rag_pipeline import RAGPipeline
from llm_client import LLMClient

# -------- INITIALIZE -------- #
os.environ['SARVAM_API_KEY'] = sarvam_ai_key
reader = PDFReader("data")
rag = RAGPipeline()
llm = LLMClient()

# -------- LOAD / BUILD VECTOR DB -------- #
@st.cache_resource
def load_vector_db():
    if os.path.exists("faiss_index"):
        print("📂 Loading existing FAISS index...")
        return rag.load_vector_db()
    else:
        print("🔨 Building new FAISS index...")
        filename = "आंबा फळपिक बुक.pdf"
        text = reader.read_pdf(os.path.join("data", filename))
        chunks = rag.create_chunks(text, source=filename)
        return rag.create_vector_db(chunks)

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
                st.write(answer)
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
