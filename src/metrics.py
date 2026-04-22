"""
Per-Model Metrics Tracking for AI Inference Gateway

Tracks detailed metrics for each model:
- Request count
- Token usage (input/output)
- Latency (with percentiles)
- Errors
- Throughput (tokens/sec)
- Active connections
- Routing decisions

Compatible with existing dashboard: ai-inference-dashboard.json
"""

from prometheus_client import Counter, Histogram, Gauge
from typing import Optional
import time

# ============================================================================
# PER-MODEL REQUEST METRICS
# ============================================================================

# Request count per model
model_requests_total = Counter(
    "gateway_model_requests_total",
    "Total requests per model",
    ["model", "backend", "status"],  # status: success, error, timeout
)

# Active gauge (increment on request start, decrement on end)
model_active_requests = Gauge(
    "gateway_model_active_requests", "Active requests per model", ["model", "backend"]
)

# Token usage per model
model_tokens_input_total = Counter(
    "gateway_model_tokens_input_total", "Input tokens per model", ["model", "backend"]
)

model_tokens_output_total = Counter(
    "gateway_model_tokens_output_total", "Output tokens per model", ["model", "backend"]
)

model_tokens_total = Counter(
    "gateway_model_tokens_total",
    "Total tokens (input + output) per model",
    ["model", "backend", "token_type"],  # token_type: input, output
)

# Request duration per model (with percentiles)
model_request_duration_seconds = Histogram(
    "gateway_model_request_duration_seconds",
    "Request duration per model",
    ["model", "backend"],
    buckets=[0.1, 0.25, 0.5, 1, 2.5, 5, 10, 15, 30, 60, 120, 300, 600],
)

# Time to first token per model
model_time_to_first_token_seconds = Histogram(
    "gateway_model_time_to_first_token_seconds",
    "Time to first token per model",
    ["model", "backend"],
    buckets=[0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 15, 30],
)

# Throughput (tokens/sec) per model
model_tokens_per_second = Gauge(
    "gateway_model_tokens_per_second",
    "Tokens per second (rolling average)",
    ["model", "backend"],
)

# Error count per model
model_errors_total = Counter(
    "gateway_model_errors_total",
    "Errors per model",
    [
        "model",
        "backend",
        "error_type",
    ],  # error_type: timeout, rate_limit, invalid_response, etc
)

# Error rate (calculated from errors_total / requests_total)
# We'll track this as a ratio gauge
model_error_rate = Gauge(
    "gateway_model_error_rate",
    "Error rate per model (rolling 5m average)",
    ["model", "backend"],
)

# ============================================================================
# ROUTING METRICS
# ============================================================================

# Routing decisions
routing_requests_total = Counter(
    "gateway_routing_requests_total",
    "Total routing decisions",
    ["requested_model", "selected_model", "backend"],
)

# Routing confidence
routing_confidence = Histogram(
    "gateway_routing_confidence",
    "Routing confidence score",
    ["reason", "specialization"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Routing overrides (when user explicitly requests a model)
routing_overrides_total = Counter(
    "gateway_routing_overrides_total",
    "Routing overrides (user requested specific model)",
    ["requested_model", "selected_model"],
)

# Specialization usage
routing_specialization_usage = Counter(
    "gateway_routing_specialization_usage",
    "Specialization usage",
    ["specialization", "model"],
)

# ============================================================================
# MODEL AVAILABILITY METRICS
# ============================================================================

# Model loaded status
model_loaded = Gauge(
    "gateway_model_loaded",
    "Model loaded status (1 = loaded, 0 = not loaded)",
    ["model"],
)

# Model availability score (from health evaluation)
model_health_score = Gauge(
    "gateway_model_health_score", "Model health score (0-100)", ["model"]
)

model_performance_score = Gauge(
    "gateway_model_performance_score", "Model performance score (0-100)", ["model"]
)

model_quality_score = Gauge(
    "gateway_model_quality_score", "Model quality score (0-100)", ["model"]
)

# ============================================================================
# BACKEND METRICS
# ============================================================================

backend_requests_total = Counter(
    "gateway_backend_requests_total",
    "Backend requests",
    ["backend", "status"],  # backend: llama-cpp, zai, vllm
)

backend_errors_total = Counter(
    "gateway_backend_errors_total", "Backend errors", ["backend", "error_type"]
)

backend_latency_seconds = Histogram(
    "gateway_backend_latency_seconds",
    "Backend latency",
    ["backend"],
    buckets=[0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60],
)

backend_healthy = Gauge(
    "gateway_backend_healthy",
    "Backend health status (1 = healthy, 0 = unhealthy)",
    ["backend"],
)

# ============================================================================
# CONTEXT WINDOW UTILIZATION
# ============================================================================

# Track context window usage (tokens / max_context)
context_utilization_percent = Gauge(
    "gateway_context_utilization_percent",
    "Context window utilization percentage",
    ["model"],
)

# Context window per request
context_window_used = Histogram(
    "gateway_context_window_used_tokens",
    "Context window used per request",
    ["model"],
    buckets=[100, 500, 1000, 2000, 4000, 8000, 16384, 32768, 65536, 131072, 262144],
)

# ============================================================================
# CONCURRENCY METRICS
# ============================================================================

# Rate limiting (requests blocked)
rate_limited_requests_total = Counter(
    "gateway_rate_limited_requests_total",
    "Rate limited requests",
    ["model", "limit_type"],  # limit_type: rpm, tpm, concurrent
)

# Queue wait time
queue_wait_time_seconds = Histogram(
    "gateway_queue_wait_time_seconds",
    "Time spent waiting in queue",
    ["model"],
    buckets=[0.01, 0.1, 0.5, 1, 2, 5, 10, 30, 60],
)

# Circuit breaker state
circuit_breaker_state = Gauge(
    "gateway_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["model", "backend"],
)

# ============================================================================
# HELPER CLASS FOR METRICS TRACKING
# ============================================================================


class ModelMetricsTracker:
    """
    Helper class to track metrics for a single request.

    Usage:
        tracker = ModelMetricsTracker(model="qwen/qwen3.5-9b", backend="llama-cpp")

        try:
            # ... process request ...
            tracker.record_success(
                input_tokens=100,
                output_tokens=200,
                latency_ms=1500
            )
        except Exception as e:
            tracker.record_error("timeout")
    """

    def __init__(self, model: str, backend: str, requested_model: Optional[str] = None):
        self.model = model
        self.backend = backend
        self.requested_model = requested_model or model
        self.start_time = time.time()
        self.first_token_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # Increment active requests
        model_active_requests.labels(model=model, backend=backend).inc()

        # Increment request counter (will adjust on success/error)
        self.request_counted = False

    def record_first_token(self):
        """Record time to first token."""
        if self.first_token_time is None:
            self.first_token_time = time.time()

            if self.start_time:
                ttft = (self.first_token_time - self.start_time) * 1000  # ms
                model_time_to_first_token_seconds.labels(
                    model=self.model, backend=self.backend
                ).observe(ttft / 1000.0)

    def record_success(
        self,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        latency_ms: float,
        model: Optional[str] = None,
    ):
        """Record successful request."""
        model = model or self.model
        self.end_time = time.time()

        # Decrement active requests
        model_active_requests.labels(model=model, backend=self.backend).dec()

        # Update token counters
        model_tokens_input_total.labels(model=model, backend=self.backend).inc(
            input_tokens
        )

        model_tokens_output_total.labels(model=model, backend=self.backend).inc(
            output_tokens
        )

        model_tokens_total.labels(
            model=model, backend=self.backend, token_type="input"
        ).inc(input_tokens)

        model_tokens_total.labels(
            model=model, backend=self.backend, token_type="output"
        ).inc(output_tokens)

        # Record request duration
        latency_seconds = latency_ms / 1000.0
        model_request_duration_seconds.labels(
            model=model, backend=self.backend
        ).observe(latency_seconds)

        # Record time to first token if available
        if self.first_token_time:
            ttft = (self.first_token_time - self.start_time) * 1000
            model_time_to_first_token_seconds.labels(
                model=model, backend=self.backend
            ).observe(ttft / 1000.0)

        # Calculate and record throughput
        if latency_ms > 0:
            tokens_per_sec = (total_tokens / latency_ms) * 1000
            model_tokens_per_second.labels(model=model, backend=self.backend).set(
                tokens_per_sec
            )

        # Mark as successful
        model_requests_total.labels(
            model=model, backend=self.backend, status="success"
        ).inc()

        self.request_counted = True

        # Update context utilization if we know max context
        from .contexts import get_context_length
        max_context = get_context_length(model)
        utilization = (total_tokens / max_context) * 100
        context_utilization_percent.labels(model=model).set(utilization)

        # Record context window used
        context_window_used.labels(model=model).observe(total_tokens)

        # Record routing decision
        if self.requested_model != model:
            routing_overrides_total.labels(
                requested_model=self.requested_model, selected_model=model
            ).inc()

        routing_requests_total.labels(
            requested_model=self.requested_model or "default",
            selected_model=model,
            backend=self.backend,
        ).inc()

    def record_error(self, error_type: str, model: Optional[str] = None):
        """Record failed request."""
        model = model or self.model
        self.end_time = time.time()

        # Decrement active requests
        model_active_requests.labels(model=model, backend=self.backend).dec()

        # Increment error counter
        model_errors_total.labels(
            model=model, backend=self.backend, error_type=error_type
        ).inc()

        # Mark as error
        model_requests_total.labels(
            model=model, backend=self.backend, status=error_type
        ).inc()

        self.request_counted = True

        # Record routing decision even on error
        routing_requests_total.labels(
            requested_model=self.requested_model or "default",
            selected_model=model,
            backend=self.backend,
        ).inc()

    def record_routing_decision(
        self, confidence: float, reason: str, specialization: Optional[str] = None
    ):
        """Record routing decision metadata."""
        routing_confidence.labels(
            reason=reason, specialization=specialization or "none"
        ).observe(confidence)

        if specialization:
            routing_specialization_usage.labels(
                specialization=specialization, model=self.model
            ).inc()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - record error if exception occurred."""
        if exc_type is not None and not self.request_counted:
            self.record_error("exception")
        return False


# ============================================================================
# ROUTING METRICS TRACKER
# ============================================================================


class RoutingMetricsTracker:
    """Track routing decisions and confidence."""

    @staticmethod
    def track_routing_decision(
        requested_model: str,
        selected_model: str,
        backend: str,
        confidence: float,
        reason: str,
        specialization: Optional[str] = None,
    ):
        """Track a routing decision."""
        # Track the decision
        routing_requests_total.labels(
            requested_model=requested_model or "default",
            selected_model=selected_model,
            backend=backend,
        ).inc()

        # Track confidence
        routing_confidence.labels(
            reason=reason, specialization=specialization or "none"
        ).observe(confidence)

        # Track override if user requested specific model
        if requested_model and requested_model != selected_model:
            routing_overrides_total.labels(
                requested_model=requested_model, selected_model=selected_model
            ).inc()

        # Track specialization usage
        if specialization:
            routing_specialization_usage.labels(
                specialization=specialization, model=selected_model
            ).inc()


# ============================================================================
# HEALTH SCORE UPDATE
# ============================================================================


def update_model_health_scores(scores: dict):
    """
    Update model health scores from evaluation results.

    Args:
        scores: Dict of {model_id: {health_score, performance_score, quality_score}}
    """
    for model_id, data in scores.items():
        if "health_score" in data:
            model_health_score.labels(model=model_id).set(data["health_score"])
        if "performance_score" in data:
            model_performance_score.labels(model=model_id).set(
                data["performance_score"]
            )
        if "quality_score" in data:
            model_quality_score.labels(model=model_id).set(data["quality_score"])


# ============================================================================
# MODEL AVAILABILITY TRACKING
# ============================================================================


def update_model_availability(models: list):
    """
    Update model availability metrics.

    Args:
        models: List of model IDs that are currently loaded
    """
    # Reset all provided models to 0 first, then set loaded ones to 1
    # This avoids issues with iterating over prometheus metric samples
    for model in models:
        try:
            model_loaded.labels(model=str(model)).set(0)
        except Exception:
            # Model might not exist yet, that's okay
            pass

    # Set currently loaded models to 1
    for model in models:
        model_loaded.labels(model=str(model)).set(1)


# ============================================================================
# ERROR RATE CALCULATION
# ============================================================================


def update_error_rates():
    """
    Calculate and update error rates for all models.
    Should be called periodically (e.g., every minute).
    """
    # Get total requests and errors per model
    # This is expensive, so don't call it too often

    for model_metric in model_requests_total._metrics:
        labels = model_metric.labels
        if not labels:
            continue

        model = labels.get("model")
        backend = labels.get("backend")

        if not model:
            continue

        # Calculate total requests
        total_requests = 0
        for metric in model_requests_total.collect():
            for sample in metric.samples:
                if (
                    sample.labels.get("model") == model
                    and sample.labels.get("backend") == backend
                ):
                    total_requests += sample.value

        # Calculate total errors
        total_errors = 0
        for metric in model_errors_total._metrics:
            for sample in metric.samples:
                if (
                    sample.labels.get("model") == model
                    and sample.labels.get("backend") == backend
                ):
                    total_errors += sample.value

        # Calculate error rate
        if total_requests > 0:
            error_rate = (total_errors / total_requests) * 100
            model_error_rate.labels(model=model, backend=backend).set(error_rate)


# Export metrics for use in other modules
__all__ = [
    "ModelMetricsTracker",
    "RoutingMetricsTracker",
    "update_model_health_scores",
    "update_model_availability",
    "update_error_rates",
    # Metrics
    "model_requests_total",
    "model_active_requests",
    "model_tokens_input_total",
    "model_tokens_output_total",
    "model_tokens_total",
    "model_request_duration_seconds",
    "model_time_to_first_token_seconds",
    "model_tokens_per_second",
    "model_errors_total",
    "model_error_rate",
    "model_loaded",
    "model_health_score",
    "model_performance_score",
    "model_quality_score",
    # Routing metrics
    "routing_requests_total",
    "routing_confidence",
    "routing_overrides_total",
    "routing_specialization_usage",
    # Backend metrics
    "backend_requests_total",
    "backend_errors_total",
    "backend_latency_seconds",
    "backend_healthy",
    # Context metrics
    "context_utilization_percent",
    "context_window_used",
    # Concurrency metrics
    "rate_limited_requests_total",
    "queue_wait_time_seconds",
    "circuit_breaker_state",
]
