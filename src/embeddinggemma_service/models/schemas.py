from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EmbeddingRequest(BaseModel):
    input: str | list[str] = Field(
        description="Text to embed, or a list of texts for batch embedding."
    )
    normalize: bool = Field(
        default=False,
        description=(
            "If true, L2-normalize the returned embedding(s). "
            "Defaults to false and preserves raw model output."
        ),
    )

    @field_validator("input")
    @classmethod
    def input_must_be_non_empty(cls, value: str | list[str]) -> str | list[str]:
        texts = value if isinstance(value, list) else [value]
        if not texts:
            raise ValueError("input list must not be empty")
        for text in texts:
            if not text.strip():
                raise ValueError("input must be a non-empty string")
        return value


class SingleEmbeddingResponse(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    dimension: int
    model: str


class EmbeddingData(BaseModel):
    index: int
    embedding: list[float]


class BatchEmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    dimension: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ModelInfoResponse(BaseModel):
    model: str
    dimension: int
    max_sequence_length: int
    runtime: Literal["litert"] = "litert"