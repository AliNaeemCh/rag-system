import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging(
    log_file: str = "logs/app.log",
    max_bytes: int = 5 * 1024 * 1024,  # 5MB
    backup_count: int = 5,
):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Prevent duplicate handlers (important in reload servers like FastAPI)
    if root_logger.handlers:
        return

    # --------------------
    # Console logging (stdout)
    # --------------------
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    # --------------------
    # File logging (rotating)
    # --------------------
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)