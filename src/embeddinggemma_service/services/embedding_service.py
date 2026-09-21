import numpy as np

from .litert_engine import LiteRTEngine
from .tokenizer import TokenizerService


class EmbeddingService:
    def __init__(
        self,
        tokenizer: TokenizerService,
        engine: LiteRTEngine,
        dimension: int,
    ) -> None:
        self._tokenizer = tokenizer
        self._engine = engine
        self._dimension = dimension

    def embed(self, text: str, normalize: bool = False) -> list[float]:
        input_ids = self._tokenizer.build_input_ids(text)
        output = self._engine.embed(input_ids)
        return self._to_embedding(output, normalize=normalize)

    def embed_batch(
        self, texts: list[str], normalize: bool = False
    ) -> list[list[float]]:
        return [
            self.embed(text, normalize=normalize)
            for text in texts
        ]

    def _to_embedding(self, output: np.ndarray, normalize: bool) -> list[float]:
        embedding = np.asarray(output).reshape(-1)
        if embedding.shape != (self._dimension,):
            raise RuntimeError(
                f"unexpected embedding dimension {len(embedding)}; "
                f"expected {self._dimension}"
            )
        if normalize:
            norm = float(np.linalg.norm(embedding))
            if norm > 0:
                embedding = embedding / norm
        return [float(x) for x in embedding.tolist()]