import os
from datetime import datetime
from pathlib import Path
from loguru import logger
import sys


class LogManager:
    """Singleton class for managing logging configuration using loguru."""
    
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
        # Remove default handler
        logger.remove()
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"answer_generator_{timestamp}.log"
        
        # Add console handler with colors
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO",
            colorize=True
        )
        
        # Add file handler
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days"
        )
        
        logger.info("Logging initialized")
        self.logger = logger
    
    def get_logger(self, name=None):
        """
        Get a logger instance.
        
        Args:
            name (str, optional): Logger name (not used in loguru, kept for compatibility).
            
        Returns:
            loguru.Logger: Logger instance
        """
        return self.logger
    
    def set_log_level(self, level):
        """
        Set the logging level.
        
        Args:
            level (str): Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR")
        """
        # Remove existing handlers and re-add with new level
        logger.remove()
        
        log_dir = Path("logs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"answer_generator_{timestamp}.log"
        
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=level,
            colorize=True
        )
        
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation="10 MB",
            retention="7 days"
        )
        
        logger.info(f"Log level set to {level}")
    
    def add_success_level(self):
        """Add custom SUCCESS level to loguru (if not already present)."""
        # Loguru doesn't have a built-in SUCCESS level, so we can add it
        try:
            logger.level("SUCCESS", no=25, color="<green>", icon="✓")
        except TypeError:
            # Level already exists
            pass
    
    def success(self, message):
        """Log a success message."""
        self.add_success_level()
        logger.opt(depth=1).log("SUCCESS", message)