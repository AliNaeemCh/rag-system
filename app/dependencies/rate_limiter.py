from app.core.config import settings

import logging
logger = logging.getLogger("app.dependencies.rate_limiter")
logger.info("Loading file...")

import time
from collections import defaultdict
from fastapi import Request, HTTPException

hits = defaultdict(list)

def rate_limit(request: Request):
    ip = request.client.host
    now = time.time()

    hits[ip] = [t for t in hits[ip] if now - t < settings.WINDOW]
    hits[ip].append(now)

    current_count = len(hits[ip])

    if current_count > settings.RATE_LIMIT:
        logger.warning(
            f"RATE LIMITED | ip={ip} | hits={current_count}"
        )
        raise HTTPException(status_code=429, detail="Too many requests")
