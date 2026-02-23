import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from pyacemaker.domain_models.logging import LoggingConfig


def setup_logger(config: LoggingConfig, project_name: str) -> logging.Logger:
    """
    Sets up the logger based on the configuration.

    Args:
        config: LoggingConfig object.
        project_name: Name of the project (logger name).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(project_name)
    logger.setLevel(config.level)

    # Remove existing handlers to avoid duplicates if re-initialized
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if config.log_file:
        log_path = Path(config.log_file)
        # Ensure directory exists
        if log_path.parent != Path():
            log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            config.log_file, maxBytes=config.max_bytes, backupCount=config.backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
