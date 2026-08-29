from app.core.config import settings

import logging
logger = logging.getLogger("app.dependencies.auth")
logger.info("Loading file...")

from fastapi import Header, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

security = HTTPBasic()

def verify_key(x_api_key: str = Header(None)):
    if not settings.API_KEY or x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

def verify_credentials(
    credentials: HTTPBasicCredentials = Depends(security),
):
    correct_username = secrets.compare_digest(
        credentials.username,
        settings.API_USERNAME,
    )

    correct_password = secrets.compare_digest(
        credentials.password,
        settings.API_PASSWORD,
    )

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )