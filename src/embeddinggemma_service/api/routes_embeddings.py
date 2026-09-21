import logging

from fastapi import APIRouter, HTTPException, Request

from embeddinggemma_service.models.schemas import (
    BatchEmbeddingResponse,
    EmbeddingData,
    EmbeddingRequest,
    SingleEmbeddingResponse,
)
from embeddinggemma_service.services.embedding_service import EmbeddingService
from embeddinggemma_service.services.tokenizer import InputTooLongError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["embeddings"])


def _get_service(request: Request) -> EmbeddingService:
    service = request.app.state.services
    if service is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return service


@router.post(
    "/embeddings",
    response_model=SingleEmbeddingResponse | BatchEmbeddingResponse,
)
async def create_embeddings(
    payload: EmbeddingRequest, request: Request
) -> SingleEmbeddingResponse | BatchEmbeddingResponse:
    settings = request.app.state.settings
    service = _get_service(request)

    texts = payload.input if isinstance(payload.input, list) else [payload.input]
    if len(texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"batch size {len(texts)} exceeds limit {settings.max_batch_size}",
        )

    try:
        embeddings = service.embed_batch(texts, normalize=payload.normalize)
    except InputTooLongError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("embedding inference failed")
        raise HTTPException(status_code=500, detail="internal inference error")

    dimension = settings.embedding_dimension
    model = settings.model_name
    if isinstance(payload.input, str):
        return SingleEmbeddingResponse(
            embedding=embeddings[0], dimension=dimension, model=model
        )
    return BatchEmbeddingResponse(
        data=[
            EmbeddingData(index=index, embedding=embedding)
            for index, embedding in enumerate(embeddings)
        ],
        model=model,
        dimension=dimension,
    )