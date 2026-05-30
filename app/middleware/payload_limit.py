from app.core.config import settings

import logging
logger = logging.getLogger("app.middleware.payload_limit")
logger.info("Loading file...")

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse

class PayloadLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        body = await request.body()

        if len(body) > settings.PAYLOAD_LIMIT:
            return JSONResponse(
                status_code=413,
                content={"detail": "Payload too large"}
            )

        return await call_next(request)