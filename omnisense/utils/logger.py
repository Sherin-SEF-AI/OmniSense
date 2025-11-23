"""
Comprehensive logging system for OMNISENSE platform.

Provides structured logging with multiple handlers including file rotation,
console output, and database logging for production monitoring.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional
import colorama
from colorama import Fore, Style

# Initialize colorama for cross-platform colored output
colorama.init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color-coded log levels for console output."""

    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured log data for database insertion."""

    def format(self, record):
        # Add structured fields
        structured_data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            structured_data['exception'] = self.formatException(record.exc_info)

        # Add custom fields if present
        if hasattr(record, 'camera_id'):
            structured_data['camera_id'] = record.camera_id
        if hasattr(record, 'person_id'):
            structured_data['person_id'] = record.person_id
        if hasattr(record, 'processing_time'):
            structured_data['processing_time'] = record.processing_time

        return str(structured_data)


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a logger with multiple handlers.

    Args:
        name: Logger name (typically module name)
        log_dir: Directory for log files (default: ./logs)
        level: Logging level
        console_output: Enable console handler
        file_output: Enable file handler
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    # Create log directory if needed
    if file_output:
        if log_dir is None:
            log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

    # Console handler with colored output
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if file_output:
        file_handler = RotatingFileHandler(
            log_dir / f"{name}.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Separate error log file
        error_handler = RotatingFileHandler(
            log_dir / f"{name}_errors.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name."""
    return logging.getLogger(name)


class PerformanceLogger:
    """Context manager for logging execution time of code blocks."""

    def __init__(self, logger: logging.Logger, operation: str, log_level: int = logging.DEBUG):
        self.logger = logger
        self.operation = operation
        self.log_level = log_level
        self.start_time = None

    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.perf_counter() - self.start_time
        self.logger.log(
            self.log_level,
            f"{self.operation} completed in {elapsed:.4f}s"
        )
        return False


# Module-level logger for utilities
logger = setup_logger(__name__)
