# Add to utils/logging_config.py
import colorlog
import logging


def get_logger(name):
    """Return a logger with colored output."""
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
