from app.dependencies.auth import verify_credentials

import logging
logger = logging.getLogger("app.api.routes.llm")
logger.info("Loading file...")

from typing import Optional, Any
from fastapi import APIRouter, Request, Depends
from pydantic import BaseModel, Field

router = APIRouter()

class LLMRequest(BaseModel):
    user_prompt: str
    model_name: Optional[str] = Field(default=None)
    output_schema: Optional[Any] = Field(default=None)
    system_prompt: Optional[str] = Field(
        default="You are a helpful assistant."
    )
    temperature: Optional[float] = Field(default=1)


@router.post("/llm", dependencies=[Depends(verify_credentials)])
async def call_llm(
    request: LLMRequest,
    req: Request,
):
    result = await req.app.state.llm_engine.generate(
        user_prompt=request.user_prompt,
        system_prompt=request.system_prompt,
        schema=request.output_schema,
        temperature=request.temperature,
    )

    return result