# Token Tracker for Open Web UI

Token Tracker is a Python module for tracking token usage in Open Web UI. It provides comprehensive monitoring of token consumption, costs, and performance metrics across different LLM providers. The data can be exported to OpenTelemetry-compatible systems or logged to files for analysis.

## Environment variables

Token Tracker is highly configurable with environment variables.

## WIP

This is still alpha. Some referenced settings doesn't work yet, specially redacting PII  (`TOKEN_TRACKER_REDACT_PII`) and hash user id's (`TOKEN_TRACKER_HASH_USER_IDS`)

## Installation

```bash
pip install token-tracker
```

## Configuration

Token Tracker is highly configurable through environment variables. Here's a complete guide to setting up and using the module.

### Core Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `TOKEN_TRACKER_ENABLED` | Enable/disable token tracking | `true` |
| `TOKEN_TRACKER_CLIENT_ID` | Client identifier for tracking | `None` |

### OpenTelemetry Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `TOKEN_TRACKER_OTEL_SERVICE_NAME` | Service name for telemetry | `token-tracker` |
| `TOKEN_TRACKER_OTEL_ENDPOINT` | OTLP endpoint URL | `OTEL_EXPORTER_OTLP_ENDPOINT` |
| `TOKEN_TRACKER_OTEL_HEADERS` | Headers for OTLP in JSON format | `OTEL_EXPORTER_OTLP_HEADERS` |
| `TOKEN_TRACKER_OTEL_INSECURE` | Allow insecure connections | `true` |
| `TOKEN_TRACKER_OTEL_EXPORT_INTERVAL` | Export interval in seconds | `30` |

### File Logging Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `TOKEN_TRACKER_FILE_LOGGING` | Enable file logging | `false` |
| `TOKEN_TRACKER_LOG_FILE` | Path to log file | `token_usage.log` |
| `TOKEN_TRACKER_LOG_ROTATION` | Log rotation size | `100MB` |
| `TOKEN_TRACKER_LOG_LEVEL` | Logging level | `INFO` |

### Storage Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `TOKEN_TRACKER_MAX_BODY_SIZE` | Maximum request body size to process | `10485760` (10MB) |
| `TOKEN_TRACKER_STORE_SAMPLES` | Store prompt/completion samples | `false` |
| `TOKEN_TRACKER_SAMPLE_LENGTH` | Maximum length of stored samples | `500` |

### Tracking Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `TOKEN_TRACKER_ESTIMATE_TOKENS` | Estimate tokens when not provided by API | `true` |
| `TOKEN_TRACKER_TRACK_COSTS` | Track token costs | `true` |
| `TOKEN_TRACKER_TRACK_PERFORMANCE` | Track performance metrics | `true` |

### Pricing Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `TOKEN_TRACKER_PRICING_JSON` | Pricing configuration in JSON format | `{}` |
| `TOKEN_TRACKER_PRICING_FILE` | Path to JSON file with pricing data | `None` |

### Endpoint Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `TOKEN_TRACKER_MONITORED_ENDPOINTS` | List of endpoints to monitor (JSON array) | See default list below |
| `TOKEN_TRACKER_EXCLUDED_ENDPOINTS` | List of endpoints to exclude (JSON array) | `[]` |

Default monitored endpoints:

```json
[
  "/api/chat/completions",
  "/api/chat/completed",
  "/api/v1/chat/completions",
  "/chat/completions",
  "/v1/chat/completions",
  "/api/completions",
  "/v1/completions",
  "/ollama/api/chat",
  "/ollama/api/generate"
]
```

### Performance Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `TOKEN_TRACKER_ASYNC_LOGGING` | Use async logging | `true` |
| `TOKEN_TRACKER_QUEUE_SIZE` | Maximum queue size for async logging | `10000` |
| `TOKEN_TRACKER_FLUSH_INTERVAL` | Flush interval in seconds | `5` |

### Privacy Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `TOKEN_TRACKER_REDACT_PII` | Redact personally identifiable information | `false` |
| `TOKEN_TRACKER_HASH_USER_IDS` | Hash user IDs for privacy | `false` |

## Setting Up Cost Profiles

Token Tracker supports detailed cost tracking for different models and providers. There are three ways to configure pricing:

### 1. Using a JSON Configuration File

Create a JSON file with your pricing structure:

```json
{
  "openai": {
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015}
  },
  "anthropic": {
    "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
    "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015}
  }
}
```

Then set the environment variable:

```bash
TOKEN_TRACKER_PRICING_FILE=/path/to/your/pricing.json
```

### 2. Using Environment Variables

You can set pricing directly through environment variables:

```bash
TOKEN_TRACKER_PRICE_OPENAI_GPT4_PROMPT=0.03
TOKEN_TRACKER_PRICE_OPENAI_GPT4_COMPLETION=0.06
TOKEN_TRACKER_PRICE_ANTHROPIC_CLAUDE3OPUS_PROMPT=0.015
TOKEN_TRACKER_PRICE_ANTHROPIC_CLAUDE3OPUS_COMPLETION=0.075
```

### 3. Using JSON in Environment Variable

For more complex configurations, you can provide JSON directly:

```bash
TOKEN_TRACKER_PRICING_JSON='{"openai":{"gpt-4":{"prompt":0.03,"completion":0.06}}}'
```

## Usage Examples

### Basic Integration with FastAPI

```python
from fastapi import FastAPI
from token_tracker.middleware import TokenUsageMiddleware

app = FastAPI()
app.add_middleware(TokenUsageMiddleware)
```

### Manual Token Logging

```python
from token_tracker.logger import get_token_logger

# Get the global logger instance
logger = get_token_logger()

# Log token usage
logger.log_token_usage(
    model="gpt-4",
    prompt_tokens=100,
    completion_tokens=50,
    user_id="user123",
    session_id="session456",
    endpoint="/api/chat/completions",
    duration_ms=1500
)
```

### Custom Token Counting

```python
from token_tracker.logger import get_token_logger, TokenCounter

# Get components
logger = get_token_logger()
token_counter = TokenCounter(logger.config)

# Register a custom counter for a specific model
def count_my_model_tokens(text):
    # Your custom logic here
    return len(text.split())

token_counter.register_custom_counter("my-custom-model", count_my_model_tokens)
```

## OpenTelemetry Integration

Token Tracker automatically exports metrics and traces to your configured OpenTelemetry endpoint. The following metrics are available:

- `token_usage_total`: Counter for token usage (prompt and completion)
- `token_usage_cost`: Counter for token costs
- `token_request_duration`: Histogram for request durations

Traces include detailed information about each request including token counts, costs, and samples.

## Docker Environment Example

```yaml
version: '3'
services:
  open-webui:
    image: openwebui/open-webui:latest
    environment:
      - TOKEN_TRACKER_ENABLED=true
      - TOKEN_TRACKER_FILE_LOGGING=true
      - TOKEN_TRACKER_LOG_FILE=/data/logs/token_usage.log
      - TOKEN_TRACKER_STORE_SAMPLES=true
      - TOKEN_TRACKER_OTEL_ENDPOINT=http://otel-collector:4317
      - TOKEN_TRACKER_PRICING_FILE=/data/config/pricing.json
```

## Troubleshooting

If you encounter issues with token tracking:

1. Check that `TOKEN_TRACKER_ENABLED` is set to `true`
2. Verify that your endpoints match the monitored endpoints list
3. For OpenTelemetry issues, check connectivity to your OTLP endpoint
4. Enable debug logging with `TOKEN_TRACKER_LOG_LEVEL=DEBUG`

## Advanced Usage

### Custom Pricing Patterns

You can use regex patterns in your pricing configuration to match model families:

```json
{
  "openai": {
    "regex:gpt-4.*": {"prompt": 0.03, "completion": 0.06},
    "regex:gpt-3.*": {"prompt": 0.0005, "completion": 0.0015}
  }
}
```

### Provider-wide Defaults

Set default pricing for all models from a provider:

```json
{
  "ollama": {
    "*": {"prompt": 0.0001, "completion": 0.0001}
  }
}
```
