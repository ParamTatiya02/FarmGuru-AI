from pdf_reader import PDFReader
from rag_pipeline import RAGPipeline
from llm_client import LLMClient
from secret_key import sarvam_ai_key
import os

# initialize
reader = PDFReader("data")
rag = RAGPipeline()
os.environ['SARVAM_API_KEY'] = sarvam_ai_key
llm = LLMClient(api_key=os.environ['SARVAM_API_KEY'])

# extract text
text = reader.read_pdf("data/आंबा फळपिक बुक.pdf")

# create chunks
chunks = rag.create_chunks(text)

# create vector db
vector_db = rag.create_vector_db(chunks)

# user query
question = "How to control mango pests?"
print(question)

# retrieve context
context = rag.query(vector_db, question)

# ask LLM
answer = llm.ask(context, question)

print("\nAnswer:\n")
print(answer)