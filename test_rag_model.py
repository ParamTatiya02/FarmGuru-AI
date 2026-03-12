from pdf_reader import PDFReader
from rag_pipeline import RAGPipeline

# Step 1: extract text
reader = PDFReader("data")
text = reader.read_pdf("data/Turmeric-research.pdf")

# Step 2: create RAG pipeline
rag = RAGPipeline()

# Step 3: split text
chunks = rag.create_chunks(text)

# Step 4: create vector database
vector_db = rag.create_vector_db(chunks)

# Step 5: ask question
query = "Summarize this research paper?"
print(query)

context = rag.query(vector_db, query)

print("\nRetrieved Context:\n")
print(context)