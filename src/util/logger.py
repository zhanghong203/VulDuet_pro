"""Thread-safe console and file logging for VulDuet experiments."""

import logging
from pathlib import Path


LOGGER_NAME = "rule_rag.v2"
DETAIL_LOGGER_NAME = "rule_rag.v2.detail"


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(threadName)s | %(message)s"
    )

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Detailed tracebacks and raw model responses are file-only so the
    # terminal stays readable during long multithreaded runs.
    detail_logger = logging.getLogger(DETAIL_LOGGER_NAME)
    detail_logger.setLevel(logging.DEBUG)
    detail_logger.handlers.clear()
    detail_logger.propagate = False
    detail_file_handler = logging.FileHandler(log_path, encoding="utf-8")
    detail_file_handler.setLevel(logging.DEBUG)
    detail_file_handler.setFormatter(formatter)
    detail_logger.addHandler(detail_file_handler)

    logger.info("Log file: %s", log_path)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def get_detail_logger() -> logging.Logger:
    return logging.getLogger(DETAIL_LOGGER_NAME)

