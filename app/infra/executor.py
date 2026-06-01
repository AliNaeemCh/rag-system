import logging
logger = logging.getLogger("app.infra.executor")
logger.info("Loading file...")

from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=10)