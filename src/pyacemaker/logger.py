import logging
from logging import Logger


# Simplified mock for tests or a basic setup
def get_logger(name: str = "pyacemaker") -> Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def setup_logger(config: object = None, project_name: str = "pyacemaker") -> Logger:
    """Legacy alias, just delegates to get_logger for now."""
    return get_logger(project_name)
