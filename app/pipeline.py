from app.config import TOP_K, EMBEDDING_DIM
from app.ingestion.chunker import Chunker
from app.embeddings.embedder import Embedder
from app.retrieval.retriever import InMemoryRetriever
from app.generation.generator import Generator


class Pipeline:
    def __init__(self):
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.retriever = InMemoryRetriever(dimension=EMBEDDING_DIM)
        self.generator = Generator()
        self._indexed = False

    def index_documents(self, documents: list) -> None:
        chunks = self.chunker.chunk_documents(documents)
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed(texts)
        self.retriever.index(embeddings, chunks)
        self._indexed = True

    def query(self, question: str) -> dict:
        if not self._indexed:
            raise RuntimeError("No documents indexed. Call index_documents() first.")
        query_embedding = self.embedder.embed([question])
        retrieved = self.retriever.retrieve(query_embedding, TOP_K)
        answer = self.generator.generate(question, retrieved)
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved,
        }
