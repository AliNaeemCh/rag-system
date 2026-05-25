from app.core.logger import setup_logging
setup_logging()

import logging
logger = logging.getLogger("app.api.main")
logger.info("Loading file...")

from fastapi import FastAPI
from app.api.routes.chat import router as chat_router
from app.core.config import settings

app = FastAPI(title=settings.TITLE)

app.include_router(chat_router, prefix="/api/v1")
