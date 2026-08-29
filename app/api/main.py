from app.core.logger import setup_logging
setup_logging()

from app.api.routes.inference import router as inference_router
from app.api.routes.llm import router as llm_router
from app.core.config import settings
from app.middleware.payload_limit import PayloadLimitMiddleware
from app.infra.scheduling.scheduler import scheduler, start_scheduler
from app.infra.executor import executor
from app.infra.usage_tracking.tracker import usage_tracker_db_pool, get_usage_tracker
from app.infra.dependencies import build_rag_pipeline, create_openai_client
from app.infra.llm_engines.openai.engine import OpenAIEngine

import logging
logger = logging.getLogger("app.api.main")
logger.info("Loading file...")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------------- STARTUP ----------------
    usage_tracker = await get_usage_tracker()
    openai_client = create_openai_client(api_key=settings.OPENAI_API_KEY)
    app.state.llm_engine = OpenAIEngine(model_name=settings.DEFAULT_LLM_MODEL, client=openai_client, usage_tracker=usage_tracker)
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
app.include_router(inference_router, prefix="/api/v1")
app.include_router(llm_router, prefix="/api/v1")
