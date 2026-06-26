import numpy as np

from app.embeddings.embedder import Embedder
from app.config import EMBEDDING_DIM


def test_embedder_returns_dense_correct_shape():
    embedder = Embedder()
    texts = ["hello world", "test sentence"]
    embeddings = embedder.embed(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, EMBEDDING_DIM)
    assert embeddings.dtype == np.float32


def test_embedder_dense_single_text():
    embedder = Embedder()
    embeddings = embedder.embed(["single text"])
    assert embeddings.shape == (1, EMBEDDING_DIM)


def test_embedder_dense_normalized():
    embedder = Embedder()
    embeddings = embedder.embed(["test one", "test two"])
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embedder_sparse_returns_dict():
    embedder = Embedder()
    texts = ["hello world", "machine learning"]
    sparse = embedder.embed_sparse(texts)
    assert len(sparse) == 2
    assert isinstance(sparse[0], dict)
    assert len(sparse[0]) > 0


def test_embedder_sparse_int_keys():
    embedder = Embedder()
    sparse = embedder.embed_sparse(["hello world"])
    for key in sparse[0].keys():
        assert isinstance(key, int)


def test_embedder_sparse_consistent_with_dense():
    embedder = Embedder()
    texts = ["the cat sat on the mat"]
    dense = embedder.embed(texts)
    sparse = embedder.embed_sparse(texts)
    assert len(dense) == 1
    assert len(sparse) == 1


def test_embedder_identical_texts_similar():
    embedder = Embedder()
    embeddings = embedder.embed(["same text", "same text"])
    cosine_sim = np.dot(embeddings[0], embeddings[1])
    assert cosine_sim > 0.999


def test_embedder_different_texts_less_similar():
    embedder = Embedder()
    embeddings = embedder.embed([
        "machine learning is fascinating",
        "pizza recipes for dinner tonight",
    ])
    cosine_sim = np.dot(embeddings[0], embeddings[1])
    assert cosine_sim < 0.95
