from .base import Middleware

__all__ = [
    "Middleware",
]

# Optional imports
try:
    from .observability import ObservabilityMiddleware  # noqa: F401

    __all__.append("ObservabilityMiddleware")
except ImportError:
    pass

try:
    from .security_filter import SecurityFilterMiddleware  # noqa: F401

    __all__.append("SecurityFilterMiddleware")
except ImportError:
    pass

try:
    from .rate_limiter import RateLimiterMiddleware  # noqa: F401

    __all__.append("RateLimiterMiddleware")
except ImportError:
    pass

try:
    from .circuit_breaker import (  # noqa: F401
        CircuitBreaker,
        CircuitBreakerState,
        CircuitBreakerOpenError,
    )

    __all__.extend(["CircuitBreaker", "CircuitBreakerState", "CircuitBreakerOpenError"])
except ImportError:
    pass

try:
    from .load_balancer import LoadBalancerMiddleware, BackendInstance, BackendState  # noqa: F401

    __all__.extend(["LoadBalancerMiddleware", "BackendInstance", "BackendState"])
except ImportError:
    pass

try:
    from .concurrency_limiter import ConcurrencyLimiter  # noqa: F401

    __all__.append("ConcurrencyLimiter")
except ImportError:
    pass
