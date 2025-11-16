"""
Configuration management for token tracker
"""

import os
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TokenTrackerConfig:
    """Configuration for token usage tracking"""

    # Core settings
    enabled: bool = True

    # OpenTelemetry configuration (always on)
    otel_service_name: str = "token-tracker"
    otel_endpoint: Optional[str] = None
    otel_headers: Dict[str, str] = field(default_factory=dict)
    otel_insecure: bool = True
    otel_export_interval: int = 30  # seconds

    # Optional file logging
    file_logging_enabled: bool = False
    log_file_path: str = "token_usage.log"
    log_rotation_size: str = "100MB"
    log_level: str = "INFO"

    # Storage settings
    max_body_size: int = 10 * 1024 * 1024  # 10MB
    store_samples: bool = False
    sample_length: int = 500  # Characters to store as sample

    # Tracking configuration
    estimate_tokens: bool = True
    track_costs: bool = True
    track_performance: bool = True

    # Pricing configuration
    pricing_config: Dict[str, Any] = field(default_factory=dict)
    pricing_file: Optional[str] = None
    pricing_update_interval: int = 3600  # seconds

    # Endpoints to monitor
    monitored_endpoints: List[str] = field(default_factory=list)
    excluded_endpoints: List[str] = field(default_factory=list)

    # Token counting settings
    tiktoken_enabled: bool = True
    default_tokens_per_char: float = 0.25  # For estimation

    # Performance settings
    async_logging: bool = True
    queue_size: int = 10000
    flush_interval: int = 5  # seconds

    # Privacy settings
    redact_pii: bool = False
    hash_user_ids: bool = False

    def __post_init__(self):
        """Initialize default values after creation"""
        if not self.monitored_endpoints:
            self.monitored_endpoints = [
                "/api/chat/completions",
                "/api/chat/completed",
                "/api/v1/chat/completions",
                "/chat/completions",
                "/v1/chat/completions",
                "/api/completions",
                "/v1/completions",
                "/ollama/api/chat",
                "/ollama/api/generate",
            ]

    @classmethod
    def from_env(cls) -> "TokenTrackerConfig":
        """Create configuration from environment variables"""

        # Helper function to parse boolean
        def parse_bool(value: str) -> bool:
            return value.lower() in ("true", "1", "yes", "on")

        # Helper function to parse JSON
        def parse_json(value: str) -> Any:
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return {}

        # Load configuration from environment
        config = cls(
            enabled=parse_bool(os.getenv("TOKEN_TRACKER_ENABLED", "true")),

            # OpenTelemetry (always enabled when tracker is enabled)
            otel_service_name=os.getenv("TOKEN_TRACKER_OTEL_SERVICE_NAME", "token-tracker"),
            otel_endpoint=os.getenv("TOKEN_TRACKER_OTEL_ENDPOINT", os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")),
            otel_headers=parse_json(os.getenv("TOKEN_TRACKER_OTEL_HEADERS", os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "{}"))),
            otel_insecure=parse_bool(os.getenv("TOKEN_TRACKER_OTEL_INSECURE", "true")),
            otel_export_interval=int(os.getenv("TOKEN_TRACKER_OTEL_EXPORT_INTERVAL", "30")),

            # Optional file logging
            file_logging_enabled=parse_bool(os.getenv("TOKEN_TRACKER_FILE_LOGGING", "false")),
            log_file_path=os.getenv("TOKEN_TRACKER_LOG_FILE", "token_usage.log"),
            log_rotation_size=os.getenv("TOKEN_TRACKER_LOG_ROTATION", "100MB"),
            log_level=os.getenv("TOKEN_TRACKER_LOG_LEVEL", "INFO"),

            # Storage
            max_body_size=int(os.getenv("TOKEN_TRACKER_MAX_BODY_SIZE", str(10 * 1024 * 1024))),
            store_samples=parse_bool(os.getenv("TOKEN_TRACKER_STORE_SAMPLES", "false")),
            sample_length=int(os.getenv("TOKEN_TRACKER_SAMPLE_LENGTH", "500")),

            # Tracking
            estimate_tokens=parse_bool(os.getenv("TOKEN_TRACKER_ESTIMATE_TOKENS", "true")),
            track_costs=parse_bool(os.getenv("TOKEN_TRACKER_TRACK_COSTS", "true")),
            track_performance=parse_bool(os.getenv("TOKEN_TRACKER_TRACK_PERFORMANCE", "true")),

            # Pricing
            pricing_config=parse_json(os.getenv("TOKEN_TRACKER_PRICING_JSON", "{}")),
            pricing_file=os.getenv("TOKEN_TRACKER_PRICING_FILE"),

            # Endpoints
            monitored_endpoints=parse_json(os.getenv("TOKEN_TRACKER_MONITORED_ENDPOINTS", "[]")),
            excluded_endpoints=parse_json(os.getenv("TOKEN_TRACKER_EXCLUDED_ENDPOINTS", "[]")),

            # Performance
            async_logging=parse_bool(os.getenv("TOKEN_TRACKER_ASYNC_LOGGING", "true")),
            queue_size=int(os.getenv("TOKEN_TRACKER_QUEUE_SIZE", "10000")),
            flush_interval=int(os.getenv("TOKEN_TRACKER_FLUSH_INTERVAL", "5")),

            # Privacy
            redact_pii=parse_bool(os.getenv("TOKEN_TRACKER_REDACT_PII", "false")),
            hash_user_ids=parse_bool(os.getenv("TOKEN_TRACKER_HASH_USER_IDS", "false")),
        )

        # Load pricing from file if specified
        if config.pricing_file and os.path.exists(config.pricing_file):
            try:
                with open(config.pricing_file, 'r') as f:
                    file_pricing = json.load(f)
                    config.pricing_config.update(file_pricing)
            except Exception as e:
                print(f"Warning: Failed to load pricing file: {e}")

        return config

    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings"""
        warnings = []

        # Check file path is writable if file logging is enabled
        if self.file_logging_enabled and self.log_file_path:
            log_dir = os.path.dirname(self.log_file_path) or "."
            if not os.access(log_dir, os.W_OK):
                warnings.append(f"Log directory not writable: {log_dir}")

        # Check OTEL endpoint
        if not self.otel_endpoint:
            warnings.append("OpenTelemetry endpoint not configured - metrics will not be exported")

        # Check pricing configuration
        if self.track_costs and not self.pricing_config and not self.pricing_file:
            warnings.append("Cost tracking enabled but no pricing configured")

        return warnings
