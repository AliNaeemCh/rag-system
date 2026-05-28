from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from app.rag.pipeline import RAGPipeline
from app.infra.dependencies import get_rag_pipeline
from typing import Optional 
import uuid
import logging
import json
from fastapi.responses import StreamingResponse
from app.rag.config import ResponseMode

logger = logging.getLogger("app.api.routes.chat")

logger.info("Loading file...")

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    mode: str
    session_id: Optional[str] = Field(default=None)

class ChatResponse(BaseModel):
    response: str
    session_id: str


@router.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):

    session_id = request.session_id or str(uuid.uuid4())

    def event_stream():

        try:
            # session start event
            yield f"data: {json.dumps({
                'type': 'session',
                'session_id': session_id
            })}\n\n"

            stream = pipeline.run(
                user_message=request.message,
                session_id=session_id,
                stream=True,
                response_mode=ResponseMode(request.mode)
            )

            full_response = ""

            for chunk in stream:
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
            logger.exception("Error in chat endpoint")

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