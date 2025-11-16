"""
Logger for token usage

"""

import json
import time
import uuid
import threading
import queue
import atexit
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import re
import os
import tiktoken

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .config import TokenTrackerConfig
from .exceptions import TokenLoggerError, ConfigurationError


class TokenSource(Enum):
    """Source of token counting"""
    API = "api"
    ESTIMATED = "estimated"
    TIKTOKEN = "tiktoken"
    CUSTOM = "custom"

class LogLevel(Enum):
    """Log levels for token usage events"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class TokenUsageEntry:
    """Complete token usage log entry"""

    # Required fields
    id: str
    timestamp: int
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # Optional fields
    user_id: Optional[str] = None
    client_id: Optional[str] = None
    session_id: Optional[str] = None
    provider: Optional[str] = None
    model_version: Optional[str] = None
    deployment_id: Optional[str] = None
    token_source: str = TokenSource.API.value
    prompt_cost: Optional[float] = None
    completion_cost: Optional[float] = None
    total_cost: Optional[float] = None
    currency: str = "USD"
    endpoint: Optional[str] = None
    request_id: Optional[str] = None
    parent_request_id: Optional[str] = None
    conversation_id: Optional[str] = None
    duration_ms: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    prompt_sample: Optional[str] = None
    completion_sample: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    streaming: bool = False
    status: str = "success"
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO format
        if isinstance(data.get('created_at'), datetime):
            data['created_at'] = data['created_at'].isoformat()
        if isinstance(data.get('updated_at'), datetime):
            data['updated_at'] = data['updated_at'].isoformat()
        return data

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)


class PricingManager:
    """Manages dynamic pricing for different models and providers"""

    def __init__(self, config: TokenTrackerConfig):
        self.config = config
        self.pricing_cache = {}
        self.last_update = None
        self.update_interval = timedelta(hours=1)
        self._lock = threading.RLock()

        # Load initial pricing
        self.reload_pricing()

    def reload_pricing(self) -> None:
        """Reload pricing from configuration"""
        with self._lock:
            self.pricing_cache = self._load_pricing()
            self.last_update = datetime.utcnow()
            logger.info("Loaded pricing for %s providers", {len(self.pricing_cache)})

    def _load_pricing(self) -> Dict[str, Any]:
        """Load pricing from various sources"""
        pricing = {}

        # 1. Loading from file
        if self.config.pricing_file and os.path.exists(self.config.pricing_file):
            try:
                with open(self.config.pricing_file, 'r') as f:
                    pricing = json.load(f)
                logger.debug("Loaded pricing from file: %s",  {self.config.pricing_file})
            except Exception as e:
                logger.error("Failed to load pricing file: %s", {e})

        # 2. Override with config pricing
        if self.config.pricing_config:
            pricing.update(self.config.pricing_config)

        # 3. Apply environment variable overrides
        pricing = self._apply_env_overrides(pricing)

        # 4. Apply defaults for missing entries
        if not pricing:
            pricing = self._get_default_pricing()

        return pricing

    def _apply_env_overrides(self, pricing: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to pricing"""
        # Pattern: TOKEN_TRACKER_PRICE_{PROVIDER}_{MODEL}_{TYPE}
        pattern = re.compile(r'TOKEN_TRACKER_PRICE_([^_]+)_([^_]+)_(PROMPT|COMPLETION)')

        for key, value in os.environ.items():
            match = pattern.match(key)
            if match:
                provider = match.group(1).lower()
                model = match.group(2).lower().replace('_', '-')
                price_type = match.group(3).lower()

                if provider not in pricing:
                    pricing[provider] = {}
                if model not in pricing[provider]:
                    pricing[provider][model] = {}

                try:
                    pricing[provider][model][price_type] = float(value)
                except ValueError:
                    logging.warning("Invalid price value for %s: %s", {key}, {value})

        return pricing

    def _get_default_pricing(self) -> Dict[str, Any]:
        """Get default pricing as fallback"""
        return {
            "openai": {
                "gpt-4": {"prompt": 0.03, "completion": 0.06},
                "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
                "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
                "gpt-4o": {"prompt": 0.005, "completion": 0.015},
                "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
                "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
                "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
                "text-embedding-3-small": {"prompt": 0.00002, "completion": 0},
                "text-embedding-3-large": {"prompt": 0.00013, "completion": 0},
                "text-embedding-ada-002": {"prompt": 0.0001, "completion": 0},
            },
            "anthropic": {
                "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
                "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
                "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
                "claude-3.5-haiku": {"prompt": 0.001, "completion": 0.005},
                "claude-2.1": {"prompt": 0.008, "completion": 0.024},
                "claude-2": {"prompt": 0.008, "completion": 0.024},
                "claude-instant": {"prompt": 0.0008, "completion": 0.0024},
            },
            "google": {
                "gemini-pro": {"prompt": 0.000125, "completion": 0.000375},
                "gemini-pro-vision": {"prompt": 0.000125, "completion": 0.000375},
                "gemini-1.5-pro": {"prompt": 0.00125, "completion": 0.00375},
                "gemini-1.5-flash": {"prompt": 0.000125, "completion": 0.000375},
            },
            "cohere": {
                "command": {"prompt": 0.0015, "completion": 0.0015},
                "command-light": {"prompt": 0.00015, "completion": 0.00015},
                "command-r": {"prompt": 0.0005, "completion": 0.0015},
                "command-r-plus": {"prompt": 0.003, "completion": 0.015},
            },
            "mistral": {
                "mistral-tiny": {"prompt": 0.00025, "completion": 0.00025},
                "mistral-small": {"prompt": 0.001, "completion": 0.003},
                "mistral-medium": {"prompt": 0.0027, "completion": 0.0081},
                "mistral-large": {"prompt": 0.008, "completion": 0.024},
            }
        }

    def calculate_cost(
        self,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        currency: str = "USD"
    ) -> Optional[Dict[str, float]]:
        """Calculate cost for token usage"""
        with self._lock:
            # Check if pricing needs refresh
            if self.last_update and datetime.utcnow() - self.last_update > self.update_interval:
                self.reload_pricing()

            # Get provider pricing
            provider_pricing = self.pricing_cache.get(provider.lower(), {})
            if not provider_pricing:
                logger.debug(f"No pricing found for provider: {provider}")
                return None

            # Find model pricing
            model_pricing = self._find_model_pricing(model, provider_pricing)
            if not model_pricing:
                logger.debug(f"No pricing found for model: {provider}/{model}")
                return None

            # Calculate costs
            prompt_cost = (prompt_tokens / 1000) * model_pricing.get("prompt", 0)
            completion_cost = (completion_tokens / 1000) * model_pricing.get("completion", 0)
            total_cost = prompt_cost + completion_cost

            # Apply currency conversion if needed
            if currency != "USD" and "exchange_rates" in self.config.__dict__:
                rate = self.config.exchange_rates.get(f"USD_to_{currency}", 1.0)
                prompt_cost *= rate
                completion_cost *= rate
                total_cost *= rate

            return {
                "prompt_cost": round(prompt_cost, 8),
                "completion_cost": round(completion_cost, 8),
                "total_cost": round(total_cost, 8),
                "currency": currency
            }

    def _find_model_pricing(self, model: str, provider_pricing: Dict) -> Optional[Dict]:
        """Find pricing for a specific model"""
        model_lower = model.lower()

        # 1. Exact match
        if model_lower in provider_pricing:
            return provider_pricing[model_lower]

        # 2. Prefix match
        for model_key, prices in provider_pricing.items():
            if not model_key.startswith("regex:") and model_lower.startswith(model_key):
                return prices

        # 3. Regex match
        for model_key, prices in provider_pricing.items():
            if model_key.startswith("regex:"):
                pattern = model_key[6:]  # Remove "regex:" prefix
                if re.match(pattern, model_lower):
                    return prices

        # 4. Wildcard/default
        if "*" in provider_pricing:
            return provider_pricing["*"]

        return None


class TokenCounter:
    """Handles token counting with multiple strategies"""

    def __init__(self, config: TokenTrackerConfig):
        self.config = config
        self.tiktoken_available = False
        self.custom_counters = {}

        # Try to import tiktoken for accurate counting
        try:
            import tiktoken
            self.tiktoken_available = True
            self.encodings = {}
            logger.debug("Tiktoken available for accurate token counting")
        except ImportError:
            logger.debug("Tiktoken not available, using estimation")

    def count_tokens(
        self,
        text: str,
        model: str = None,
        method: str = "auto"
    ) -> Tuple[int, str]:
        """
        Count tokens in text

        Returns:
            Tuple of (token_count, method_used)
        """
        if not text:
            return 0, TokenSource.ESTIMATED.value

        # Try methods in order of preference
        if method == "auto":
            # Try tiktoken first if available
            if self.tiktoken_available and model:
                try:
                    count = self._count_with_tiktoken(text, model)
                    return count, TokenSource.TIKTOKEN.value
                except Exception as e:
                    logger.debug("Tiktoken counting failed: %s", {e})

            # Try custom counter if registered
            if model in self.custom_counters:
                try:
                    count = self.custom_counters[model](text)
                    return count, TokenSource.CUSTOM.value
                except Exception as e:
                    logger.debug("Custom counter failed: %s", {e})

        # Fallback to estimation
        count = self._estimate_tokens(text)
        return count, TokenSource.ESTIMATED.value

    def _count_with_tiktoken(self, text: str, model: str) -> int:
        """Count tokens using tiktoken library"""

        # Map model to encoding
        encoding_map = {
            "gpt-4": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "text-davinci": "p50k_base",
            "text-embedding": "cl100k_base",
        }

        # Find encoding
        encoding_name = None
        for prefix, enc in encoding_map.items():
            if model.startswith(prefix):
                encoding_name = enc
                break

        if not encoding_name:
            encoding_name = "cl100k_base"

        # Cache encodings
        if encoding_name not in self.encodings:
            self.encodings[encoding_name] = tiktoken.get_encoding(encoding_name)

        encoding = self.encodings[encoding_name]
        return len(encoding.encode(text))

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count based on text length"""

        # Basic character count
        char_count = len(text)

        # Count words for better estimation
        word_count = len(text.split())

        # Adjust for different languages and patterns
        # English: ~4 chars per token, ~0.75 tokens per word
        # Code: ~3 chars per token

        # Detect if text contains significant non-ASCII
        non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / max(char_count, 1)

        if non_ascii_ratio > 0.3:
            # Likely not English or latin language
            estimated = char_count / 2
        elif "```" in text or "def " in text or "function " in text:
            # Likely contains code
            estimated = char_count / 3
        else:
            # Likely English or latin language
            estimated = max(char_count / 4, word_count * 0.75)

        return max(1, int(estimated))

    def register_custom_counter(self, model: str, counter: Callable[[str], int]):
        """Register a custom token counter for a specific model"""
        self.custom_counters[model] = counter


class TokenUsageLogger:
    """Main logger class for token usage tracking"""

    def __init__(self, config: Optional[TokenTrackerConfig] = None):
        self.config = config or TokenTrackerConfig.from_env()
        self.enabled = self.config.enabled

        if not self.enabled:
            logger.info("Token usage logger disabled")
            return

        # Initialize components
        self.pricing_manager = PricingManager(self.config)
        self.token_counter = TokenCounter(self.config)

        # Threading components for async logging
        self.write_queue = queue.Queue(maxsize=10000)
        self.write_lock = threading.RLock()
        self.shutdown_event = threading.Event()

        # Statistics
        self.stats = {
            "total_logged": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "errors": 0,
            "start_time": time.time()
        }

        # Initialize storage backends
        self._init_telemetry_backend()
        self._init_file_backend()

        # Start background writer thread
        self.writer_thread = threading.Thread(target=self._background_writer, daemon=True)
        self.writer_thread.start()

        # Register shutdown handler
        atexit.register(self.shutdown)

        logger.info("Token usage logger initialized with %s backends", len(self._backends))

    def _init_telemetry_backend(self):
        """Initialize OpenTelemetry backend (always enabled)"""
        try:
            from .telemetry import TelemetryBackend
            self._telemetry_backend = TelemetryBackend(self.config)
            logger.info("OpenTelemetry backend initialized")
        except Exception as e:
            logger.error("Failed to initialize OpenTelemetry backend: %s", e)
            self._telemetry_backend = None

    def _init_file_backend(self):
        """Initialize optional file logging backend"""
        self._file_backend = None

        if self.config.file_logging_enabled and self.config.log_file_path:
            try:
                log_path = Path(self.config.log_file_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self._file_backend = log_path
                logger.info("File backend initialized: %s", log_path)
            except Exception as e:
                logger.error("Failed to initialize file backend: %s", e)

    @property
    def _backends(self) -> List[str]:
        """Get list of active backends"""
        backends = []
        if self._telemetry_backend:
            backends.append("opentelemetry")
        if self._file_backend:
            backends.append("file")
        return backends

    def log_token_usage(
        self,
        model: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        prompt_text: Optional[str] = None,
        completion_text: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        request_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        streaming: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        status: str = "success",
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        async_write: bool = True
    ) -> Optional[str]:
        """
        Log token usage with automatic token counting if needed

        Returns:
            Entry ID if successful, None otherwise
        """
        if not self.enabled:
            return None

        try:
            # Generate entry ID
            entry_id = str(uuid.uuid4())

            # Identify provider
            provider = self._identify_provider(model, endpoint)

            # Count tokens if not provided
            token_source = TokenSource.API.value

            if prompt_tokens is None and prompt_text:
                prompt_tokens, token_source = self.token_counter.count_tokens(prompt_text, model)

            if completion_tokens is None and completion_text:
                completion_tokens, source = self.token_counter.count_tokens(completion_text, model)
                if source != TokenSource.API.value:
                    token_source = source

            # Default to 0 if still None
            prompt_tokens = prompt_tokens or 0
            completion_tokens = completion_tokens or 0
            total_tokens = prompt_tokens + completion_tokens

            # Calculate cost
            cost_info = None
            if self.config.track_costs:
                cost_info = self.pricing_manager.calculate_cost(
                    model, provider, prompt_tokens, completion_tokens
                )

            # Create samples if configured
            prompt_sample = None
            completion_sample = None
            if self.config.store_samples:
                sample_length = self.config.sample_length
                if prompt_text:
                    prompt_sample = prompt_text[:sample_length]
                if completion_text:
                    completion_sample = completion_text[:sample_length]

            # Calculate performance metrics
            tokens_per_second = None
            if duration_ms and completion_tokens > 0:
                tokens_per_second = (completion_tokens / duration_ms) * 1000

            # Set Client id if it exists
            client_id = None
            if metadata and "client_id" in metadata:
                client_id = metadata["client_id"]
            elif self.config.client_id:
                client_id = self.config.client_id

            # Create entry
            entry = TokenUsageEntry(
                id=entry_id,
                timestamp=int(time.time()),
                user_id=user_id,
                client_id=client_id,
                session_id=session_id,
                model=model,
                provider=provider,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                token_source=token_source,
                prompt_cost=cost_info.get("prompt_cost") if cost_info else None,
                completion_cost=cost_info.get("completion_cost") if cost_info else None,
                total_cost=cost_info.get("total_cost") if cost_info else None,
                currency=cost_info.get("currency", "USD") if cost_info else "USD",
                endpoint=endpoint,
                request_id=request_id or entry_id,
                conversation_id=conversation_id,
                duration_ms=duration_ms,
                tokens_per_second=tokens_per_second,
                prompt_sample=prompt_sample,
                completion_sample=completion_sample,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming,
                metadata=metadata,
                tags=tags or [],
                status=status,
                error_code=error_code,
                error_message=error_message
            )

            # Update statistics
            self._update_stats(entry)

            # Write entry
            if async_write and self.write_queue:
                self.write_queue.put(entry)
            else:
                self._write_entry(entry)

            logger.debug(f"Token usage logged: {entry_id} - {model} - {total_tokens} tokens")
            return entry_id

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to log token usage: {e}", exc_info=True)
            return None

    def _identify_provider(self, model: str, endpoint: Optional[str] = None) -> str:
        """Identify provider from model name and endpoint"""
        model_lower = model.lower()
        endpoint_lower = (endpoint or "").lower()
        if "gpt-" in model_lower or "text-davinci" in model_lower:
            return "openai"
        elif "claude" in model_lower or "anthropic" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower or "palm" in model_lower:
            return "google"
        elif "command" in model_lower and "cohere" in model_lower:
            return "cohere"
        elif "mistral" in model_lower or "mixtral" in model_lower:
            return "mistral"
        elif "llama" in model_lower:
            return "meta"

        if endpoint and "ollama" in endpoint.lower():
            # Ollama is just the runtime, detect actual model provider
            if "anthropic" in model_lower or "claude" in model_lower:
                return "anthropic"
            elif "openai" in model_lower or "gpt" in model_lower:
                return "openai"
            # If can't determine, it's a local model via ollama
            return "ollama/local"
        # Check endpoint patterns first
        endpoint_patterns = {
            "openai": ["openai", "/v1/chat", "/v1/completions"],
            "anthropic": ["anthropic", "claude", "/v1/messages"],
            "google": ["google", "gemini", "vertexai"],
            "azure": ["azure", "microsoft"],
            "cohere": ["cohere"],
            "mistral": ["mistral"],
            "ollama": ["ollama", "/api/generate", "/api/chat"],
            "huggingface": ["huggingface", "hf/"],
        }

        for provider, patterns in endpoint_patterns.items():
            if any(pattern in endpoint_lower for pattern in patterns):
                return provider

        # Check model name patterns
        model_patterns = {
            "openai": ["gpt-", "text-davinci", "text-embedding", "whisper", "dall-e"],
            "anthropic": ["claude"],
            "google": ["gemini", "palm", "bard"],
            "cohere": ["command", "embed"],
            "mistral": ["mistral", "mixtral"],
            "meta": ["llama", "codellama"],
            "stability": ["stable-diffusion"],
        }

        for provider, patterns in model_patterns.items():
            if any(pattern in model_lower for pattern in patterns):
                return provider

        # Check for local/open models
        open_models = ["llama", "mistral", "mixtral", "phi", "qwen", "yi", "deepseek"]
        if any(m in model_lower for m in open_models):
            return "local"

        return "unknown"

    def _update_stats(self, entry: TokenUsageEntry):
        """Update internal statistics"""
        with self.write_lock:
            self.stats["total_logged"] += 1
            self.stats["total_tokens"] += entry.total_tokens
            if entry.total_cost:
                self.stats["total_cost"] += entry.total_cost

    def _write_entry(self, entry: TokenUsageEntry):
        """Write entry to all configured backends"""

        # Primary: Send to OpenTelemetry
        if self._telemetry_backend:
            try:
                self._telemetry_backend.record_metrics(entry)
                self._telemetry_backend.create_span(entry)
            except Exception as e:
                logger.error(f"Failed to send telemetry: {e}")

        # Optional: Write to file
        if self._file_backend:
            try:
                with self.write_lock:
                    with open(self._file_backend, 'a', encoding='utf-8') as f:
                        f.write(entry.to_json() + '\n')
                        f.flush()
            except Exception as e:
                logger.error(f"Failed to write to file: {e}")

    def _background_writer(self):
        """Background thread for async writing"""
        logger.debug("Background writer thread started")

        while not self.shutdown_event.is_set():
            try:
                # Wait for entry with timeout
                entry = self.write_queue.get(timeout=1)

                if entry is None:  # Shutdown signal
                    break

                self._write_entry(entry)
                self.write_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Background writer error: {e}")

        # Flush remaining entries
        while not self.write_queue.empty():
            try:
                entry = self.write_queue.get_nowait()
                if entry:
                    self._write_entry(entry)
            except queue.Empty:
                break

        logger.debug("Background writer thread stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self.write_lock:
            uptime = time.time() - self.stats["start_time"]
            return {
                **self.stats,
                "uptime_seconds": uptime,
                "backends": self._backends,
                "queue_size": self.write_queue.qsize() if self.write_queue else 0
            }

    def flush(self):
        """Flush all pending writes"""
        if self.write_queue:
            self.write_queue.join()

    def shutdown(self):
        """Graceful shutdown"""
        if not self.enabled:
            return

        logger.info("Shutting down token usage logger")

        # Signal shutdown
        self.shutdown_event.set()

        # Flush queue
        if self.write_queue:
            self.write_queue.put(None)  # Shutdown signal

        # Wait for writer thread
        if hasattr(self, 'writer_thread') and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=5)

        if self._telemetry_backend:
            self._telemetry_backend.close()

        # Log final stats
        stats = self.get_stats()
        logger.info(f"Token logger shutdown - Total logged: {stats['total_logged']}, "
                   f"Total tokens: {stats['total_tokens']}, "
                   f"Total cost: ${stats['total_cost']:.4f}")


# Convenience function for global logger instance
_global_logger = None

def get_token_logger(config: Optional[TokenTrackerConfig] = None) -> TokenUsageLogger:
    """Get or create global token usage logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = TokenUsageLogger(config)
    return _global_logger
