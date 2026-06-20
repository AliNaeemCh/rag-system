from app.infra.usage_tracking.tracker import usage_tracker
from app.infra.scheduling.decorators import job

import logging
logger = logging.getLogger("app.infra.tasks")
logger.info("Loading file...")

@job(cron={"hour": 0, "minute": 0}, enabled=usage_tracker is not None)
def delete_older_token_usage():
    return usage_tracker.delete_older_token_usage()