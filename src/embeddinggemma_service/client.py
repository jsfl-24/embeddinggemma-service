import os
from typing import Protocol, Sequence

import httpx

DEFAULT_EMBEDDING_SERVICE_URL = "http://127.0.0.1:8100"


class EmbeddingClientError(Exception):
    """Base error for embedding client failures."""


class EmbeddingServiceUnavailable(EmbeddingClientError):
    """Raised when the embedding service cannot be reached or is not ready."""


class InvalidEmbeddingInput(EmbeddingClientError):
    """Raised when the embedding service rejects the supplied input."""


class Embedder(Protocol):
    """Interface between application code and text embedding.

    Application code should depend only on this shape so it is not tied
    to any particular embedding implementation.
    """

    def embed(self, text: str, *, normalize: bool = False) -> list[float]: ...

    def embed_batch(
        self, texts: Sequence[str], *, normalize: bool = False
    ) -> list[list[float]]: ...


class EmbeddingClient:
    """HTTP client for a local embedding service.

    Connects to POST /v1/embeddings and returns embeddings as plain
    Python lists of floats.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_EMBEDDING_SERVICE_URL,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        )

    @classmethod
    def from_env(cls) -> "EmbeddingClient":
        return cls(
            base_url=os.getenv(
                "EMBEDDING_SERVICE_URL", DEFAULT_EMBEDDING_SERVICE_URL
            )
        )

    def embed(self, text: str, *, normalize: bool = False) -> list[float]:
        data = self._post({"input": text, "normalize": normalize})
        embedding = data.get("embedding")
        if not isinstance(embedding, list) or not all(
            isinstance(value, (int, float)) for value in embedding
        ):
            raise EmbeddingClientError("unexpected embedding payload from service")
        return [float(value) for value in embedding]

    def embed_batch(
        self, texts: Sequence[str], *, normalize: bool = False
    ) -> list[list[float]]:
        payload_texts = list(texts)
        data = self._post({"input": payload_texts, "normalize": normalize})
        items = data.get("data")
        if not isinstance(items, list) or len(items) != len(payload_texts):
            raise EmbeddingClientError("unexpected batch payload from service")
        try:
            indexed = {item["index"]: item["embedding"] for item in items}
        except (KeyError, TypeError) as exc:
            raise EmbeddingClientError("unexpected batch payload from service") from exc
        return [
            [float(value) for value in indexed[index]]
            for index in range(len(payload_texts))
        ]

    def _post(self, payload: dict) -> dict:
        try:
            response = self._client.post("/v1/embeddings", json=payload)
        except httpx.HTTPError as exc:
            raise EmbeddingServiceUnavailable(
                f"unable to reach embedding service: {exc}"
            ) from exc
        if response.status_code == 400:
            raise InvalidEmbeddingInput("the embedding service rejected the input")
        if response.status_code >= 500:
            raise EmbeddingServiceUnavailable("embedding service is not available")
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as exc:
            raise EmbeddingClientError("invalid response from embedding service") from exc
        if not isinstance(data, dict):
            raise EmbeddingClientError("invalid response from embedding service")
        return data

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "EmbeddingClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()