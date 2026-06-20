from app.core.logger import setup_logging
setup_logging()

from app.api.routes.chat import router as chat_router
from app.core.config import settings
from app.middleware.payload_limit import PayloadLimitMiddleware
from app.infra.scheduling.scheduler import scheduler, start_scheduler
from app.infra.executor import executor
import app.dependencies.rag_pipeline    # Loads RAG pipeline

import logging
logger = logging.getLogger("app.api.main")
logger.info("Loading file...")

from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------------- STARTUP ----------------
    start_scheduler(scheduler)

    yield

    # ---------------- SHUTDOWN ----------------
    scheduler.shutdown(wait=True)
    executor.shutdown(wait=True)

app = FastAPI(
    title=settings.TITLE,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan
)

app.add_middleware(PayloadLimitMiddleware)
app.include_router(chat_router, prefix="/api/v1")
