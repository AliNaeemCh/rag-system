import logging
logger = logging.getLogger("app.infra.db.session")
logger.info("Loading file...")

from contextlib import asynccontextmanager
from psycopg_pool import AsyncConnectionPool
from psycopg import AsyncConnection
from typing import AsyncGenerator

@asynccontextmanager
async def get_connection(
    pool: AsyncConnectionPool,
) -> AsyncGenerator[AsyncConnection, None]:
    async with pool.connection() as conn:
        yield conn