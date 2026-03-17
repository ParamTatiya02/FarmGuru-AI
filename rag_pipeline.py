import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class RAGPipeline:

    def __init__(self, index_path: str = "vectorDB"):
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def create_chunks(self, text: str, source: str = "unknown"):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=80
        )
        chunks = splitter.create_documents(
            [text],
            metadatas=[{"source": source}]
        )
        print(f"📄 '{source}' → {len(chunks)} chunks created")
        return chunks

    def create_vector_db(self, chunks):
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local(self.index_path)
        print(f"💾 Vector DB saved to '{self.index_path}'")
        return vectorstore

    def load_vector_db(self):
        print(f"📂 Loading existing vector DB from '{self.index_path}'")
        return FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def get_or_create_vector_db(self, chunks=None):
        """Load existing DB if available, otherwise create a new one."""
        if os.path.exists(self.index_path):
            return self.load_vector_db()
        if chunks is None:
            raise ValueError("No existing index found. Please provide chunks.")
        return self.create_vector_db(chunks)

    def query(self, vectorstore, question: str) -> str:
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        sources = set(doc.metadata.get("source", "unknown") for doc in docs)
        print(f"📚 Answer sourced from: {sources}")
        return context