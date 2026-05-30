from app.core.config import settings

import logging
logger = logging.getLogger("app.dependencies.auth")
logger.info("Loading file...")

from fastapi import Header, HTTPException

def verify_key(x_api_key: str = Header(None)):
    if not settings.API_KEY or x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")