from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS


class RAGPipeline:

    def __init__(self):
        self.embeddings = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    def create_chunks(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_text(text)
        print(f"Total chunks created: {len(chunks)}")
        return chunks

    def create_vector_db(self, chunks):
        vectorstore = FAISS.from_texts(
            chunks,
            self.embeddings
        )
        return vectorstore

    def query(self, vectorstore, question):
        docs = vectorstore.similarity_search(
            question,
            k=3
        )
        context = "\n".join([doc.page_content for doc in docs])
        return context