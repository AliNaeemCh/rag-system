from app.core.logger import setup_logging
setup_logging()

from app.api.routes.chat import router as chat_router
from app.core.config import settings
from app.middleware.payload_limit import PayloadLimitMiddleware
from app.infra.scheduling.scheduler import scheduler, start_scheduler
from app.infra.executor import executor
from app.infra.usage_tracking.tracker import usage_tracker_db_pool
from app.infra.dependencies import build_rag_pipeline

import logging
logger = logging.getLogger("app.api.main")
logger.info("Loading file...")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------------- STARTUP ----------------
    pipeline, rag_db_pool = await build_rag_pipeline()
    app.state.pipeline = pipeline
    app.state.rag_db_pool = rag_db_pool
    start_scheduler(scheduler)

    yield

    # ---------------- SHUTDOWN ----------------
    if usage_tracker_db_pool:
        await usage_tracker_db_pool.close()
    await rag_db_pool.close()
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://d3v7a184z6qq5v.cloudfront.net"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router, prefix="/api/v1")
