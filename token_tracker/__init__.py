"""
Token Tracker - AI/LLM Token Usage Tracking Library

A comprehensive solution for tracking token usage, costs, and performance
metrics for AI/LLM applications.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .config import TokenTrackerConfig
from .exceptions import (
    TokenTrackerError,
    TokenLoggerError,
    ConfigurationError,
    PricingError,
    TokenCountingError,
    TelemetryError,
)
from .logger import (
    TokenUsageLogger,
    TokenUsageEntry,
    TokenSource,
    get_token_logger,
)

try:
    from .middleware import TokenUsageMiddleware
except ImportError:
    # Middleware requires additional dependencies
    TokenUsageMiddleware = None

__all__ = [
    # Core classes
    "TokenTrackerConfig",
    "TokenUsageLogger",
    "TokenUsageEntry",
    "TokenUsageMiddleware",
    "TokenSource",

    # Exceptions
    "TokenTrackerError",
    "TokenLoggerError",
    "ConfigurationError",
    "PricingError",
    "TokenCountingError",
    "TelemetryError",

    # Utility functions
    "get_token_logger",

    # Version
    "__version__",
]

def setup_middleware(app, logger=None):
    """
    Setup token tracker middleware on the given app.

    Returns:
        bool: True if successfully setup, False otherwise
    """
    try:
        config = TokenTrackerConfig.from_env()

        if not config.enabled:
            if logger:
                logger.info("Token tracker is disabled via configuration")
            return False

        # Validate config
        warnings = config.validate()
        for warning in warnings:
            if logger:
                logger.warning(f"Token tracker warning: {warning}")

        # Add middleware
        from .middleware import TokenUsageMiddleware
        app.add_middleware(TokenUsageMiddleware, config=config)

        if logger:
            logger.info(f"Token tracker middleware enabled - OTEL: {config.otel_endpoint or 'console'}, "
                       f"File: {config.file_logging_enabled}")
        return True

    except Exception as e:
        if logger:
            logger.error(f"Failed to setup token tracker: {e}")
        return False
