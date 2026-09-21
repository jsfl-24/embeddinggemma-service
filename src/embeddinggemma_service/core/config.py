from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODEL_PATH = Path(
    r"D:\LLM\liteRT\embeddinggemma-300m\embeddinggemma-300M_seq2048_mixed-precision.tflite"
)
DEFAULT_TOKENIZER_PATH = Path(
    r"D:\LLM\liteRT\embeddinggemma-300m\sentencepiece.model"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8100

    embedding_model_path: Path = DEFAULT_MODEL_PATH
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH

    model_name: str = "embeddinggemma-300m"
    embedding_dimension: int = 768
    max_sequence_length: int = 2048

    max_batch_size: int = 32
    truncation_mode: Literal["truncate", "error"] = "truncate"
    num_threads: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()