import httpx
import pytest

from embeddinggemma_service.client import (
    DEFAULT_EMBEDDING_SERVICE_URL,
    EmbeddingClient,
    EmbeddingClientError,
    EmbeddingServiceUnavailable,
    InvalidEmbeddingInput,
)

SINGLE_PAYLOAD = {
    "object": "embedding",
    "embedding": [0.1, -0.2, 0.3],
    "dimension": 3,
    "model": "embeddinggemma-300m",
}

BATCH_PAYLOAD = {
    "object": "list",
    "data": [
        {"index": 0, "embedding": [0.1, -0.2, 0.3]},
        {"index": 1, "embedding": [0.4, 0.5, -0.6]},
    ],
    "model": "embeddinggemma-300m",
    "dimension": 3,
}


def _client(handler) -> EmbeddingClient:
    return EmbeddingClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )


def _json_response(payload, status_code=200):
    return httpx.Response(status_code, json=payload)


def test_default_url():
    assert DEFAULT_EMBEDDING_SERVICE_URL == "http://127.0.0.1:8100"


def test_embed_single():
    client = _client(lambda request: _json_response(SINGLE_PAYLOAD))
    with client:
        embedding = client.embed("Hello world")
    assert isinstance(embedding, list)
    assert all(isinstance(value, float) for value in embedding)
    assert embedding == pytest.approx([0.1, -0.2, 0.3])


def test_embed_posts_correct_payload():
    captured = {}

    def handler(request):
        captured["json"] = request.read()
        return _json_response(SINGLE_PAYLOAD)

    client = _client(handler)
    with client:
        client.embed("Hello world", normalize=True)
    import json

    body = json.loads(captured["json"])
    assert body == {"input": "Hello world", "normalize": True}


def test_embed_batch_order():
    client = _client(lambda request: _json_response(BATCH_PAYLOAD))
    with client:
        embeddings = client.embed_batch(["first", "second"])
    assert len(embeddings) == 2
    assert embeddings[0] == pytest.approx([0.1, -0.2, 0.3])
    assert embeddings[1] == pytest.approx([0.4, 0.5, -0.6])


def test_embed_batch_mismatched_count_rejected():
    payload = {
        "object": "list",
        "data": [{"index": 0, "embedding": [0.1]}],
        "model": "m",
        "dimension": 1,
    }
    client = _client(lambda request: _json_response(payload))
    with client:
        with pytest.raises(EmbeddingClientError):
            client.embed_batch(["a", "b"])


def test_invalid_input_on_400():
    client = _client(lambda request: _json_response({"detail": "invalid request"}, 400))
    with client:
        with pytest.raises(InvalidEmbeddingInput):
            client.embed("")

    client = _client(lambda request: _json_response({"detail": "not loaded"}, 503))
    with client:
        with pytest.raises(EmbeddingServiceUnavailable):
            client.embed("Hello")


def test_unavailable_on_server_error():
    client = _client(lambda request: _json_response({"detail": "boom"}, 500))
    with client:
        with pytest.raises(EmbeddingServiceUnavailable):
            client.embed("Hello")


def test_unavailable_on_connection_error():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=httpx.Request(request.method, request.url))

    client = _client(handler)
    with client:
        with pytest.raises(EmbeddingServiceUnavailable):
            client.embed("Hello")


def test_malformed_response_rejected():
    client = _client(lambda request: _json_response("not a valid payload"))
    with client:
        with pytest.raises(EmbeddingClientError):
            client.embed("Hello")


def test_embedding_missing_field_rejected():
    client = _client(lambda request: _json_response({"object": "embedding"}))
    with client:
        with pytest.raises(EmbeddingClientError):
            client.embed("Hello")