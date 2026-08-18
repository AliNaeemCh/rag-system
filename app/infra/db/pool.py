from app.core.config import settings

import logging
logger = logging.getLogger("app.infra.db.pool")
logger.info("Loading file...")

from psycopg_pool import AsyncConnectionPool


def get_usage_tracker_db_pool() -> AsyncConnectionPool:

    usage_tracker_db_pool = AsyncConnectionPool(
        min_size=1,
        max_size=settings.USAGE_TRACKER_DB_POOL_MAX_CONNS,
        conninfo=settings.USAGE_TRACKER_DB_URL,
        open=False
    )

    return usage_tracker_db_pool

def get_rag_db_pool() -> AsyncConnectionPool:

    rag_db_pool = AsyncConnectionPool(
        min_size=1,
        max_size=settings.RAG_DB_POOL_MAX_CONNS,
        conninfo=settings.RAG_DB_URL,
        open=False
    )

    return rag_db_pool
