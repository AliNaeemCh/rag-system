from app.infra.usage_tracking.tracker import get_usage_tracker
from app.infra.scheduling.decorators import job

import logging
logger = logging.getLogger("app.infra.tasks")
logger.info("Loading file...")

@job(cron={"hour": 0, "minute": 0})
async def delete_older_token_usage():
    usage_tracker = await get_usage_tracker()

    if usage_tracker:
        return await usage_tracker.delete_older_token_usage()