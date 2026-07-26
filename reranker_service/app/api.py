from fastapi import APIRouter
from reranker_service.app.reranker import Reranker
from reranker_service.app.schemas import (
    RerankRequest,
    RerankResponse,
)
from reranker_service.app.loader import load_model
from reranker_service.app.config import MODEL_PATH

session, tokenizer = load_model(MODEL_PATH)
reranker = Reranker(
    session=session,
    tokenizer=tokenizer,
)

router = APIRouter()

@router.post(
    "/rerank",
    response_model=RerankResponse,
)
def rerank(request: RerankRequest):

    scores = reranker.rerank(
        request.query,
        request.documents
    )

    return {
        "scores": scores
    }