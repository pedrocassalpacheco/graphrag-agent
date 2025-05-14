import colorlog
import logging
from pathlib import Path
from datetime import datetime


def get_logger(name):
    """stdout logging with colors"""
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Set the level
        logger.setLevel(logging.DEBUG)

        # Create a color formatter
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s: %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        # Create console handler and add formatter
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

    return logger


def get_file_logger(name, log_file=None):
    """
    Get a logger that writes to a file.

    Args:
        name: Logger name
        log_file: Path to log file (default: logs/{name}_{timestamp}.log)

    Returns:
        Logger configured for file output
    """

    # Create default log filename with timestamp if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{name}_{timestamp}.log"
    else:
        log_file = Path(log_file)

    # Get logger instance
    logger = logging.getLogger(f"{name}_file")
    logger.setLevel(logging.INFO)  # Use a fixed level like the standard logger

    # Only add handler if not already set up
    if not logger.handlers:
        # Create file handler with standard formatter
        file_handler = logging.FileHandler(str(log_file), mode="a")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add handler and store log file reference
        logger.addHandler(file_handler)
        logger.log_file = str(log_file)
        logger.propagate = False

    return logger


# Configure root logger to suppress third-party logs
def configure_logging():
    # Set root logger to WARNING to suppress INFO logs from third-party libraries
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    # Specific configuration for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Add more libraries as needed


# Call this at application startup
configure_logging()
