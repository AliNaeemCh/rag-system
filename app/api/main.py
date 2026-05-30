from app.core.logger import setup_logging
setup_logging()

from app.api.routes.chat import router as chat_router
from app.core.config import settings
from app.middleware.payload_limit import PayloadLimitMiddleware


import logging
logger = logging.getLogger("app.api.main")
logger.info("Loading file...")

from fastapi import FastAPI

app = FastAPI(
    title=settings.TITLE,
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

app.add_middleware(PayloadLimitMiddleware)
app.include_router(chat_router, prefix="/api/v1")
