import app.infra.tasks  # Loading tasks file to populate registry
from app.infra.scheduling.registry import JOB_REGISTRY

import logging
logger = logging.getLogger("app.infra.scheduling.scheduler")
logger.info("Loading file...")

from apscheduler.schedulers.background import BackgroundScheduler
import pytz

def start_scheduler(scheduler: BackgroundScheduler):
    """
    Starts APScheduler and loads all @job decorated tasks
    """

    if scheduler.running:
        return
    for job in JOB_REGISTRY:

        # cron-based jobs
        if job["cron"]:
            scheduler.add_job(
                job["func"],
                trigger="cron",
                **job["cron"]
            )

        # interval-based jobs
        elif job["interval"]:
            scheduler.add_job(
                job["func"],
                trigger="interval",
                **job["interval"]
            )

    scheduler.start()

scheduler = BackgroundScheduler(timezone=pytz.UTC)