"""
OpenTelemetry backend for token usage metrics (always enabled)
"""

from typing import Optional, Dict, Any
import time

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider, PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

from .exceptions import TelemetryError


class TelemetryBackend:
    """OpenTelemetry backend for metrics export (always enabled)"""

    def __init__(self, config: Any):
        self.config = config
        self._setup_telemetry()

    def _setup_telemetry(self):
        """Setup OpenTelemetry metrics and tracing"""

        # Create resource
        resource = Resource.create({
            "service.name": self.config.otel_service_name,
            "service.version": "1.0.0",
        })

        # Create metric readers list
        metric_readers = []

        # Setup OTLP exporter if endpoint is configured
        if self.config.otel_endpoint:
            try:
                metric_exporter = OTLPMetricExporter(
                    endpoint=self.config.otel_endpoint,
                    headers=self.config.otel_headers,
                    insecure=self.config.otel_insecure,
                )

                metric_reader = PeriodicExportingMetricReader(
                    exporter=metric_exporter,
                    export_interval_millis=self.config.otel_export_interval * 1000,
                )
                metric_readers.append(metric_reader)
            except Exception as e:
                print(f"Warning: Failed to setup OTLP exporter: {e}")

        # If no OTLP endpoint, use console exporter for debugging
        if not metric_readers:
            console_exporter = ConsoleMetricExporter()
            console_reader = PeriodicExportingMetricReader(
                exporter=console_exporter,
                export_interval_millis=60000,  # Export every minute to console
            )
            metric_readers.append(console_reader)

        # Create meter provider
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=metric_readers,
        )

        # Set global meter provider
        metrics.set_meter_provider(meter_provider)

        # Store for cleanup
        self.meter_provider = meter_provider

        # Get meter
        self.meter = metrics.get_meter(
            "token_tracker",
            version="1.0.0",
        )

        # Create metrics instruments
        self._create_instruments()

        # Setup tracer
        self.tracer = trace.get_tracer(
            "token_tracker",
            version="1.0.0",
        )

    def _create_instruments(self):
        """Create metric instruments"""

        # Counter for total tokens
        self.token_counter = self.meter.create_counter(
            name="token_usage_total",
            description="Total tokens used",
            unit="tokens",
        )

        # Counter for total cost
        self.cost_counter = self.meter.create_counter(
            name="token_usage_cost",
            description="Total cost of token usage",
            unit="USD",
        )

        # Histogram for request duration
        self.duration_histogram = self.meter.create_histogram(
            name="token_usage_duration",
            description="Duration of AI model requests",
            unit="ms",
        )

        # Histogram for tokens per request
        self.tokens_per_request_histogram = self.meter.create_histogram(
            name="tokens_per_request",
            description="Number of tokens per request",
            unit="tokens",
        )

        # Counter for requests
        self.request_counter = self.meter.create_counter(
            name="token_requests_total",
            description="Total number of token requests",
            unit="requests",
        )

        # Counter for errors
        self.error_counter = self.meter.create_counter(
            name="token_requests_errors",
            description="Total number of failed token requests",
            unit="errors",
        )

        # Gauge for active requests (optional)
        self.active_requests = self.meter.create_up_down_counter(
            name="token_requests_active",
            description="Number of active token requests",
            unit="requests",
        )

    def record_metrics(self, entry: Any) -> None:
        """Record token usage metrics to OpenTelemetry"""
        try:
            # Common attributes
            attributes = {
                "model": entry.model,
                "provider": entry.provider or "unknown",
                "endpoint": entry.endpoint or "unknown",
                "token_source": entry.token_source,
            }

            # Add user info if available and not redacted
            if entry.user_id and not self.config.redact_pii:
                attributes["user_id"] = entry.user_id

            # Record token counts
            if entry.prompt_tokens > 0:
                self.token_counter.add(
                    entry.prompt_tokens,
                    attributes={**attributes, "token_type": "prompt"}
                )

            if entry.completion_tokens > 0:
                self.token_counter.add(
                    entry.completion_tokens,
                    attributes={**attributes, "token_type": "completion"}
                )

            # Record total tokens
            self.tokens_per_request_histogram.record(
                entry.total_tokens,
                attributes=attributes
            )

            # Record cost if available
            if entry.total_cost and entry.total_cost > 0:
                self.cost_counter.add(
                    entry.total_cost,
                    attributes=attributes
                )

            # Record duration if available
            if entry.duration_ms:
                self.duration_histogram.record(
                    entry.duration_ms,
                    attributes=attributes
                )

            # Record request count
            if entry.status == "success":
                self.request_counter.add(
                    1,
                    attributes={**attributes, "status": "success"}
                )
            else:
                self.error_counter.add(
                    1,
                    attributes={
                        **attributes,
                        "status": "error",
                        "error_code": entry.error_code or "unknown"
                    }
                )

        except Exception as e:
            # Log error but don't fail the main operation
            print(f"Warning: Failed to record metrics: {e}")

    def create_span(self, entry: Any) -> None:
        """Create a trace span for the token usage"""
        try:
            # Create span with context
            with self.tracer.start_as_current_span(
                "token_usage",
                attributes={
                    "model": entry.model,
                    "provider": entry.provider or "unknown",
                    "endpoint": entry.endpoint or "unknown",
                },
            ) as span:
                # Add detailed attributes
                span.set_attribute("token.prompt_tokens", entry.prompt_tokens)
                span.set_attribute("token.completion_tokens", entry.completion_tokens)
                span.set_attribute("token.total_tokens", entry.total_tokens)
                span.set_attribute("token.source", entry.token_source)

                if entry.total_cost:
                    span.set_attribute("token.cost", entry.total_cost)
                    span.set_attribute("token.currency", entry.currency)

                if entry.duration_ms:
                    span.set_attribute("token.duration_ms", entry.duration_ms)

                if entry.temperature is not None:
                    span.set_attribute("token.temperature", entry.temperature)

                if entry.max_tokens is not None:
                    span.set_attribute("token.max_tokens", entry.max_tokens)

                # Add event
                span.add_event(
                    name="token_usage_recorded",
                    attributes={
                        "model": entry.model,
                        "tokens": entry.total_tokens,
                        "cost": entry.total_cost or 0,
                    },
                )

                # Set status
                if entry.status == "success":
                    span.set_status(trace.Status(trace.StatusCode.OK))
                else:
                    span.set_status(
                        trace.Status(
                            trace.StatusCode.ERROR,
                            entry.error_message or "Unknown error"
                        )
                    )

        except Exception as e:
            print(f"Warning: Failed to create span: {e}")

    def close(self):
        """Close telemetry connections"""
        try:
            if hasattr(self, 'meter_provider'):
                self.meter_provider.force_flush()
                self.meter_provider.shutdown()
        except Exception:
            pass