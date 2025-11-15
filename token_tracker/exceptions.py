"""
Custom exceptions for token tracker module
"""

class TokenTrackerError(Exception):
    """Base exception for token tracker"""
    pass


class TokenLoggerError(TokenTrackerError):
    """Exception raised for token logger specific errors"""
    pass


class ConfigurationError(TokenTrackerError):
    """Exception raised for configuration-related errors"""
    pass


class PricingError(TokenTrackerError):
    """Exception raised for pricing calculation errors"""
    pass


class TokenCountingError(TokenTrackerError):
    """Exception raised for token counting errors"""
    pass


class TelemetryError(TokenTrackerError):
    """Exception raised for telemetry export errors"""
    pass