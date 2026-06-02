from app.core.config import settings

import logging
logger = logging.getLogger("app.infra.db.pool")
logger.info("Loading file...")

from psycopg2.pool import SimpleConnectionPool

db_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=settings.DB_POOL_MAX_CONNS,
    dsn=settings.DB_URL
)

usage_tracker_db_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=settings.USAGE_TRACKER_DB_POOL_MAX_CONNS,
    dsn=settings.USAGE_TRACKER_DB_URL
)