from app.config import TOP_K, TOP_N
from app.ingestion.chunker import Chunker
from app.embeddings.embedder import Embedder
from app.retrieval.retriever import QdrantHybridRetriever
from app.retrieval.reranker import Reranker
from app.generation.generator import Generator


class Pipeline:
    def __init__(self):
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.retriever = QdrantHybridRetriever()
        self.reranker = Reranker()
        self.generator = Generator()
        self._indexed = False

    def index_documents(self, documents: list) -> None:
        chunks = self.chunker.chunk_documents(documents)
        texts = [c["text"] for c in chunks]
        dense_vecs = self.embedder.embed(texts)
        sparse_vecs = self.embedder.embed_sparse(texts)
        self.retriever.index(dense_vecs, sparse_vecs, chunks)
        self._indexed = True

    def query(self, question: str) -> dict:
        if not self._indexed:
            raise RuntimeError("No documents indexed. Call index_documents() first.")
        dense = self.embedder.embed([question])[0]
        sparse = self.embedder.embed_sparse([question])[0]
        retrieved = self.retriever.retrieve(dense, sparse, TOP_N)
        reranked = self.reranker.rerank(question, retrieved)[:TOP_K]
        answer = self.generator.generate(question, reranked)
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": reranked,
        }
