import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from embeddinggemma_service.api import routes_embeddings, routes_health, routes_models
from embeddinggemma_service.core.config import Settings, get_settings
from embeddinggemma_service.core.logging import configure_logging
from embeddinggemma_service.services.embedding_service import EmbeddingService
from embeddinggemma_service.services.litert_engine import LiteRTEngine
from embeddinggemma_service.services.tokenizer import TokenizerService

logger = logging.getLogger(__name__)


def _build_services(settings: Settings) -> EmbeddingService:
    logger.info("Loading EmbeddingGemma model")
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
    logger.info("Model loaded successfully")
    logger.info("Embedding dimension: %d", settings.embedding_dimension)
    return EmbeddingService(tokenizer, engine, settings.embedding_dimension)


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.services = None
        try:
            app.state.services = _build_services(settings)
        except Exception:
            logger.exception("Failed to load model")
        yield
        app.state.services = None

    app = FastAPI(
        title="EmbeddingGemma Service",
        description=(
            "Generic local text embedding service using "
            "EmbeddingGemma 300M through LiteRT."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": "invalid request"})

    app.include_router(routes_health.router)
    app.include_router(routes_models.router)
    app.include_router(routes_embeddings.router)
    return app


app = create_app()