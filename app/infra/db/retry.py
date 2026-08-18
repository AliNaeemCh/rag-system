import logging
logger = logging.getLogger("app.infra.db.retry")
logger.info("Loading file...")

from functools import wraps
from psycopg import OperationalError, InterfaceError

from functools import wraps
from inspect import iscoroutinefunction


def db_retry(retries=1):
    def decorator(fn):
        if iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args, **kwargs):
                for attempt in range(retries + 1):
                    try:
                        return await fn(*args, **kwargs)
                    except (OperationalError, InterfaceError):
                        if attempt == retries:
                            raise

            return async_wrapper

        @wraps(fn)
        def sync_wrapper(*args, **kwargs):
            for attempt in range(retries + 1):
                try:
                    return fn(*args, **kwargs)
                except (OperationalError, InterfaceError):
                    if attempt == retries:
                        raise

        return sync_wrapper

    return decorator