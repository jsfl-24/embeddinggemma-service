import argparse
import logging
import sys

from embeddinggemma_service.core.config import get_settings
from embeddinggemma_service.core.logging import configure_logging
from embeddinggemma_service.services.embedding_service import EmbeddingService
from embeddinggemma_service.services.litert_engine import LiteRTEngine
from embeddinggemma_service.services.tokenizer import TokenizerService

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(
        prog="embeddinggemma-service",
        description="Generate a text embedding using the local EmbeddingGemma model.",
    )
    parser.add_argument("text", nargs="?", help="text to embed")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize the returned embedding",
    )
    args = parser.parse_args(argv)

    if not args.text:
        parser.print_help(sys.stderr)
        return 2

    settings = get_settings()
    try:
        tokenizer = TokenizerService(
            settings.tokenizer_path,
            max_sequence_length=settings.max_sequence_length,
            truncation_mode=settings.truncation_mode,
        )
        engine = LiteRTEngine(
            settings.embedding_model_path,
            max_sequence_length=settings.max_sequence_length,
            num_threads=settings.num_threads,
        )
        service = EmbeddingService(tokenizer, engine, settings.embedding_dimension)
    except Exception:
        logger.exception("failed to load model")
        return 1

    embedding = service.embed(args.text, normalize=args.normalize)
    print(f"Model: {settings.model_name}")
    print(f"Dimension: {settings.embedding_dimension}")
    print("Embedding:")
    print(embedding)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())