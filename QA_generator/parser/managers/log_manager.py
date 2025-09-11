from pathlib import Path
from loguru import logger


class LogManager:
    """Singleton class for managing logging configuration using Loguru."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._setup_logging()
            LogManager._initialized = True

    def _setup_logging(self):
        """Set up logging configuration using loguru."""
        from sys import stdout

        logger.remove()

        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / "pdf_converter.log"

        # Console (clean, colorized)
        logger.add(
            stdout,
            format="<green>{time:DD-MM-YYYY HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level="INFO",
            colorize=True,
            enqueue=False,  # important to prevent double console logging
        )

        # File (detailed)
        logger.add(
            log_file,
            format="{time:DD-MM-YYYY HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            enqueue=True,  # safe for multiprocessing, but ONLY for file
        )

        self.logger = logger
        self.logger.info("Logging initialized")

    def get_logger(self):
        """Return the Loguru logger."""
        return logger

    def set_log_level(self, level: str):
        """
        Set the logging level.

        Args:
            level (str): Logging level as a string (e.g., "DEBUG", "INFO")
        """
        logger.remove()  # Clear current handlers
        self._setup_logging()  # Re-setup logging with possibly new level
        logger.info(f"Log level set to {level}")

    def add_success_level(self):
        """Add custom SUCCESS level to loguru."""
        try:
            logger.level("SUCCESS", no=25, color="<green>", icon="✓")
        except TypeError:
            pass

    def success(self, message):
        """Log a success message."""
        self.add_success_level()
        logger.opt(depth=1).log("SUCCESS", message)
