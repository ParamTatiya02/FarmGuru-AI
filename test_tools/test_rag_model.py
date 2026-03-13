from pdf_reader import PDFReader
from rag_pipeline import RAGPipeline
from llm_client import LLMClient
from secret_key import sarvam_ai_key
import os

# -------- INITIALIZE -------- #
reader = PDFReader("data")
rag = RAGPipeline()
os.environ['SARVAM_API_KEY'] = sarvam_ai_key
llm = LLMClient()

# -------- BUILD VECTOR DB -------- #
if os.path.exists("faiss_index"):
    print("📂 Loading existing FAISS index...")
    vectorstore = rag.load_vector_db()
else:
    print("🔨 Building new FAISS index...")
    filename = "आंबा फळपिक बुक.pdf"
    text = reader.read_pdf(os.path.join("data", filename))
    chunks = rag.create_chunks(text, source=filename)
    vectorstore = rag.create_vector_db(chunks)
    
# # -------- BUILD VECTOR DB -------- #
# if os.path.exists("faiss_index"):
#     print("📂 Loading existing FAISS index...")
#     vectorstore = rag.load_vector_db()
# else:
#     print("🔨 Building new FAISS index from PDFs...")
#     all_chunks = []

#     # ✅ Loop all PDFs — each chunk tagged with its source filename
#     for filename in os.listdir("data"):
#         if filename.endswith(".pdf"):
#             print(f"\n📄 Processing: {filename}")
#             text = reader.read_pdf(os.path.join("data", filename))
#             chunks = rag.create_chunks(text, source=filename)  # ✅ real filename as source
#             all_chunks.extend(chunks)

#     print(f"\n📦 Total chunks across all PDFs: {len(all_chunks)}")
#     vectorstore = rag.create_vector_db(all_chunks)

# -------- QUERY -------- #
question = "How to control mango pests?"
print(f"\n❓ Question: {question}")

context = rag.query(vectorstore, question)   # ✅ fixed variable name
answer = llm.ask(context, question)

print("\n💬 Answer:")
print(answer)