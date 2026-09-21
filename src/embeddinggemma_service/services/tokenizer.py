from pathlib import Path
from typing import Literal

import numpy as np
import sentencepiece as spm


class InputTooLongError(ValueError):
    """Raised when strict truncation mode rejects an over-length input."""


class TokenizerService:
    def __init__(
        self,
        model_path: str | Path,
        max_sequence_length: int,
        truncation_mode: Literal["truncate", "error"] = "truncate",
    ) -> None:
        self._sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self._max_sequence_length = max_sequence_length
        self._truncation_mode = truncation_mode
        self.pad_id = int(self._sp.pad_id())
        self.bos_id = int(self._sp.bos_id())
        self.eos_id = int(self._sp.eos_id())

    def encode(self, text: str) -> list[int]:
        return self._sp.encode(text, out_type=int)

    def build_input_ids(self, text: str) -> np.ndarray:
        """Tokenize text into the fixed [1, max_sequence_length] int32 input.

        Layout: [BOS] + tokens + [EOS] + [PAD]... up to max_sequence_length.
        """
        ids = self.encode(text)
        available = self._max_sequence_length - 2
        if len(ids) > available and self._truncation_mode == "error":
            raise InputTooLongError(
                f"input exceeds the {self._max_sequence_length}-token limit"
            )
        ids = ids[:available]
        ids = [self.bos_id] + ids + [self.eos_id]
        ids.extend([self.pad_id] * (self._max_sequence_length - len(ids)))
        return np.asarray([ids], dtype=np.int32)