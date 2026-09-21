from fastapi import APIRouter, Request

from embeddinggemma_service.models.schemas import ModelInfoResponse

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models", response_model=ModelInfoResponse)
async def model_info(request: Request) -> ModelInfoResponse:
    settings = request.app.state.settings
    return ModelInfoResponse(
        model=settings.model_name,
        dimension=settings.embedding_dimension,
        max_sequence_length=settings.max_sequence_length,
    )