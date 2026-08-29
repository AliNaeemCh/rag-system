from app.rag.pipeline import RAGPipeline
from app.rag.config import ResponseMode
from app.rag.chat_history import chat_history
from app.dependencies.auth import verify_credentials
from app.dependencies.rate_limiter import rate_limit
from app.core.config import settings
from app.models import RetrievedDocument

import logging
logger = logging.getLogger("app.api.routes.inference")
logger.info("Loading file...")

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from typing import Optional 
import uuid
import json
from fastapi.responses import StreamingResponse, JSONResponse

router = APIRouter()

class InferenceRequest(BaseModel):
    message: str
    mode: str

class ChatRequest(InferenceRequest):
    session_id: Optional[str] = Field(default=None)

class EvaluatorInferenceResponse(BaseModel):
    final_top_docs: list[RetrievedDocument]
    response: str

def rag_pipeline(request: Request):
    return request.app.state.pipeline

@router.post("/chat",    dependencies=[
        Depends(rate_limit)
    ])
async def chat_endpoint(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(rag_pipeline),
):
    chat_history.cleanup()  # Cleans inactive sessions data

    input_len = len(request.message)
    if input_len > settings.USER_IN_MAX_CHARS:
        return JSONResponse(status_code=400, content={"detail": "Input too long."})
    
    session_id = request.session_id or str(uuid.uuid4())

    async def event_stream():

        try:
            # session start event
            yield f"data: {json.dumps({
                'type': 'session',
                'session_id': session_id
            })}\n\n"

            stream = await pipeline.run(
                user_message=request.message,
                session_id=session_id,
                stream=True,
                response_mode=ResponseMode(request.mode)
            )

            full_response = ""

            async for chunk in stream:
                full_response += chunk

                yield f"data: {json.dumps({
                    'type': 'token',
                    'token': chunk,
                    'session_id': session_id
                })}\n\n"

            # done event
            yield f"data: {json.dumps({
                'type': 'done',
                'session_id': session_id
            })}\n\n"

        except Exception as e:
            logger.exception(f"Error in chat endpoint.")

            yield f"data: {json.dumps({
                'type': 'error',
                'message': "Something went wrong",
                'detail': str(e),
                'session_id': session_id
            })}\n\n"

            return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )

@router.post(
    "/evaluator/inference",
    response_model=EvaluatorInferenceResponse,
    dependencies=[
        Depends(verify_credentials)
    ],
)
async def evaluator_inference_endpoint(
    request: InferenceRequest,
    pipeline: RAGPipeline = Depends(rag_pipeline),
):

    try:
        response = await pipeline.run(
            user_message=request.message,
            session_id="",
            stream=False,
            eval_mode=True,
            response_mode=ResponseMode(request.mode),
        )

        return EvaluatorInferenceResponse(**response)

    except Exception as e:
        logger.exception("Error in evaluator inference endpoint.")

        return JSONResponse(
            status_code=500,
            detail="Something went wrong.",
        )