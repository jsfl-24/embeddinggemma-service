import math

import pytest


def _get_embedding(client, text):
    response = client.post("/v1/embeddings", json={"input": text})
    assert response.status_code == 200
    return response.json()["embedding"]


def test_single_embedding_shape(client):
    body = client.post("/v1/embeddings", json={"input": "Hello world"}).json()
    assert body["object"] == "embedding"
    assert body["dimension"] == 768
    assert body["model"] == "embeddinggemma-300m"
    assert len(body["embedding"]) == 768
    assert all(isinstance(x, float) for x in body["embedding"])


def test_batch_embedding_ordering_matches_single(client):
    texts = ["Hello world", "This is another sentence.", "A third piece of text."]
    response = client.post("/v1/embeddings", json={"input": texts})
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["dimension"] == 768
    assert len(body["data"]) == 3
    assert [d["index"] for d in body["data"]] == [0, 1, 2]
    for d in body["data"]:
        assert len(d["embedding"]) == 768

    singles = [_get_embedding(client, text) for text in texts]
    for data, single in zip(body["data"], singles):
        assert data["embedding"] == pytest.approx(single, rel=1e-5)


def test_deterministic(client):
    first = _get_embedding(client, "The quick brown fox jumps over the lazy dog.")
    second = _get_embedding(client, "The quick brown fox jumps over the lazy dog.")
    assert first == pytest.approx(second, rel=1e-6)


def test_normalize(client):
    body = client.post(
        "/v1/embeddings", json={"input": "Hello world", "normalize": True}
    ).json()
    norm = math.sqrt(sum(x * x for x in body["embedding"]))
    assert norm == pytest.approx(1.0, rel=1e-3)


def test_empty_string_rejected(client):
    assert client.post("/v1/embeddings", json={"input": ""}).status_code == 400
    assert client.post("/v1/embeddings", json={"input": "   "}).status_code == 400


def test_empty_batch_rejected(client):
    assert client.post("/v1/embeddings", json={"input": []}).status_code == 400


def test_batch_size_limit(client):
    saved = client.app.state.settings
    client.app.state.settings = saved.model_copy(update={"max_batch_size": 2})
    try:
        response = client.post(
            "/v1/embeddings", json={"input": ["one", "two", "three"]}
        )
        assert response.status_code == 400
    finally:
        client.app.state.settings = saved