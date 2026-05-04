import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from app.core.config import settings, LogLevel

COLORS = {
    LogLevel.DEBUG: "\033[36m",
    LogLevel.INFO: "\033[32m",
    LogLevel.WARNING: "\033[33m",
    LogLevel.ERROR: "\033[31m",
    LogLevel.CRITICAL: "\033[41m",
    "RESET": "\033[0m",
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname
        name = record.name

        try:
            level_enum = LogLevel(levelname)
            color = COLORS.get(level_enum, COLORS["RESET"])

            record.levelname = f"{color}{levelname}{COLORS['RESET']}"
            record.name = f"\033[37m{name}{COLORS['RESET']}"

            return super().format(record)

        finally:
            # restore original values
            record.levelname = levelname
            record.name = name


def setup_logging(
    log_level: LogLevel = settings.LOG_LEVEL,
    log_file: str = "logs/app.log",
    max_bytes: int = 5 * 1024 * 1024,   # 5 MB
    backup_count: int = 5,
):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    plain_formatter = logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S")
    colored_formatter = ColorFormatter(fmt, "%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(log_level.value)

    if root.handlers:
        return

    # Console (colored)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(colored_formatter)

    # File (plain — NO colors)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(plain_formatter)

    root.addHandler(console)
    root.addHandler(file_handler)