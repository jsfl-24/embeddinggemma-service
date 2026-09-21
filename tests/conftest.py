from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from embeddinggemma_service.core.config import Settings
from embeddinggemma_service.main import create_app

MODEL_PATH = Path(
    r"D:\LLM\liteRT\embeddinggemma-300m\embeddinggemma-300M_seq2048_mixed-precision.tflite"
)
TOKENIZER_PATH = Path(r"D:\LLM\liteRT\embeddinggemma-300m\sentencepiece.model")

REAL_MODELS_AVAILABLE = MODEL_PATH.is_file() and TOKENIZER_PATH.is_file()


def build_settings(**overrides) -> Settings:
    kwargs = dict(
        embedding_model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        model_name="embeddinggemma-300m",
        embedding_dimension=768,
        max_sequence_length=2048,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


@pytest.fixture(scope="session")
def client():
    if not REAL_MODELS_AVAILABLE:
        pytest.skip("local model files not available")
    app = create_app(build_settings())
    with TestClient(app) as test_client:
        yield test_client