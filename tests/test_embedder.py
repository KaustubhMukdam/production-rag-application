import numpy as np

from app.embeddings.embedder import Embedder
from app.config import EMBEDDING_DIM


def test_embedder_returns_correct_shape():
    embedder = Embedder()
    texts = ["hello world", "test sentence"]
    embeddings = embedder.embed(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, EMBEDDING_DIM)


def test_embedder_single_text():
    embedder = Embedder()
    embeddings = embedder.embed(["single text"])
    assert embeddings.shape == (1, EMBEDDING_DIM)
    assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64


def test_embedder_similar_texts_close():
    embedder = Embedder()
    similar = embedder.embed(["machine learning is interesting", "machine learning is fun"])
    different = embedder.embed(["machine learning is interesting", "pizza recipes for dinner"])
    sim_similar = np.dot(similar[0], similar[1])
    sim_different = np.dot(different[0], different[1])
    assert sim_similar > sim_different


def test_embedder_normalized_vectors():
    embedder = Embedder()
    embeddings = embedder.embed(["test one", "test two"])
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embedder_identical_texts_almost_identical():
    embedder = Embedder()
    embeddings = embedder.embed(["same text", "same text"])
    cosine_sim = np.dot(embeddings[0], embeddings[1])
    assert cosine_sim > 0.9999
