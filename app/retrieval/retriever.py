from typing import List
import numpy as np
from qdrant_client import QdrantClient, models

from app.config import EMBEDDING_DIM, QDRANT_URL, QDRANT_COLLECTION, TOP_N


class QdrantHybridRetriever:
    def __init__(
        self,
        url: str = QDRANT_URL,
        collection_name: str = QDRANT_COLLECTION,
    ):
        self._client = QdrantClient(url, check_compatibility=False)
        self._collection_name = collection_name

    def index(self, dense_vecs: np.ndarray, sparse_vecs: List[dict], chunks: List[dict]) -> None:
        if self._client.collection_exists(self._collection_name):
            self._client.delete_collection(self._collection_name)
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                ),
            },
        )
        points = [
            models.PointStruct(
                id=i,
                vector={
                    "dense": dense.tolist(),
                    "sparse": models.SparseVector(
                        indices=list(sparse.keys()),
                        values=list(sparse.values()),
                    ),
                },
                payload=chunk,
            )
            for i, (dense, sparse, chunk) in enumerate(zip(dense_vecs, sparse_vecs, chunks))
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)

    def retrieve(self, dense: np.ndarray, sparse: dict, k: int = 5) -> List[dict]:
        try:
            results = self._client.query_points(
                collection_name=self._collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense.tolist(),
                        using="dense",
                        limit=TOP_N,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=list(sparse.keys()),
                            values=list(sparse.values()),
                        ),
                        using="sparse",
                        limit=TOP_N,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k,
                with_payload=True,
            )
        except Exception:
            return []
        return [
            {
                **point.payload,
                "score": point.score,
            }
            for point in results.points
        ]

    @property
    def size(self) -> int:
        try:
            collection = self._client.get_collection(self._collection_name)
            return collection.points_count
        except Exception:
            return 0
