from pathlib import Path

import numpy as np
import pytest

from embeddinggemma_service.services.tokenizer import (
    InputTooLongError,
    TokenizerService,
)

TOKENIZER_PATH = Path(r"D:\LLM\liteRT\embeddinggemma-300m\sentencepiece.model")

pytestmark = pytest.mark.skipif(
    not TOKENIZER_PATH.is_file(), reason="local tokenizer not available"
)


def _service(**kwargs):
    defaults = dict(model_path=TOKENIZER_PATH, max_sequence_length=8)
    defaults.update(kwargs)
    return TokenizerService(**defaults)


def test_special_ids_match_gemma():
    sp = _service()
    assert sp.pad_id == 0
    assert sp.eos_id == 1
    assert sp.bos_id == 2


def test_build_input_ids_layout():
    sp = _service(truncation_mode="truncate")
    ids = sp.build_input_ids("Hello world")
    assert ids.shape == (1, 8)
    assert ids.dtype == np.int32
    row = ids[0].tolist()
    assert row[0] == sp.bos_id
    eos_at = row.index(sp.eos_id)
    assert any(token != sp.pad_id for token in row[1:eos_at])
    assert all(token == sp.pad_id for token in row[eos_at + 1 :])


def test_truncates_long_input():
    sp = _service(truncation_mode="truncate")
    ids = sp.build_input_ids("word " * 100)
    assert ids.shape == (1, 8)
    row = ids[0].tolist()
    assert row[0] == sp.bos_id
    assert sp.eos_id in row


def test_error_mode_rejects_long_input():
    sp = _service(truncation_mode="error")
    with pytest.raises(InputTooLongError):
        sp.build_input_ids("word " * 100)