from app.infra.scheduling.registry import JOB_REGISTRY

import logging
logger = logging.getLogger("app.infra.scheduling.decorators")
logger.info("Loading file...")

def job(cron=None, interval=None):
    """
    Decorator to register scheduled jobs
    """
    def wrapper(fn):
        JOB_REGISTRY.append({
            "func": fn,
            "cron": cron,
            "interval": interval
        })
        return fn

    return wrapper