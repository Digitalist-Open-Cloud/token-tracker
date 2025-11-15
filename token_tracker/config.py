import os
from typing import Optional
from dataclasses import dataclass

@dataclass
class TokenTrackerConfig:
    """Configuration for token usage tracking"""

    # Logging configuration
    enabled: bool = True
    log_file_path: str = "token_usage.log"
    log_rotation_size: str = "100MB"
    max_body_size: int = 10 * 1024 * 1024  # 10MB

    # Database configuration
    db_enabled: bool = False
    db_url: Optional[str] = None

    # OpenTelemetry configuration
    otel_enabled: bool = False
    otel_service_name: str = "token-tracker"
    otel_endpoint: Optional[str] = None

    # Tracking configuration
    estimate_tokens: bool = True
    track_costs: bool = True

    # Endpoints to monitor
    monitored_endpoints: list = None

    def __post_init__(self):
        if self.monitored_endpoints is None:
            self.monitored_endpoints = [
                "/api/chat/completions",
                "/api/v1/chat/completions",
                "/chat/completions",
                "/v1/chat/completions",
            ]

    @classmethod
    def from_env(cls) -> "TokenTrackerConfig":
        """Create config from environment variables"""
        return cls(
            enabled=os.getenv("TOKEN_TRACKER_ENABLED", "true").lower() == "true",
            log_file_path=os.getenv("TOKEN_TRACKER_LOG_FILE", "token_usage.log"),
            log_rotation_size=os.getenv("TOKEN_TRACKER_LOG_ROTATION", "100MB"),
            db_enabled=os.getenv("TOKEN_TRACKER_DB_ENABLED", "false").lower() == "true",
            db_url=os.getenv("TOKEN_TRACKER_DB_URL"),
            otel_enabled=os.getenv("TOKEN_TRACKER_OTEL_ENABLED", "false").lower() == "true",
            otel_endpoint=os.getenv("TOKEN_TRACKER_OTEL_ENDPOINT"),
        )