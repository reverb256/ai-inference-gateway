#!/usr/bin/env python3
"""
SearXNG Monitoring and Observability Module

Provides:
- Prometheus metrics collection
- Health check endpoints
- Performance tracking
- Alert conditions
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not available. Metrics disabled.")

logger = logging.getLogger(__name__)


class SearXNGMetrics:
    """
    Prometheus metrics for SearXNG search operations.

    Metrics tracked:
    - Total search requests (by category, engine)
    - Search duration histograms
    - Cache hits/misses
    - Active engines
    - Result quality scores
    """

    def __init__(self, enable_prometheus: bool = True, metrics_port: int = 9090):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.metrics_port = metrics_port

        if self.enable_prometheus:
            logger.info("Prometheus metrics enabled")
            self._init_metrics()
        else:
            logger.warning("Prometheus metrics disabled")

    def _init_metrics(self):
        """Initialize Prometheus metrics."""

        # Request counters
        self.search_requests_total = Counter(
            'searxng_search_requests_total',
            'Total search requests',
            ['category', 'domain', 'engine']
        )

        self.search_errors_total = Counter(
            'searxng_search_errors_total',
            'Total search errors',
            ['error_type']
        )

        # Duration histograms
        self.search_duration_seconds = Histogram(
            'searxng_search_duration_seconds',
            'Search request duration',
            ['category', 'domain'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )

        self.engine_duration_seconds = Histogram(
            'searxng_engine_duration_seconds',
            'Search engine response time',
            ['engine'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )

        # Cache metrics
        self.cache_hits_total = Counter(
            'searxng_cache_hits_total',
            'Total cache hits'
        )

        self.cache_misses_total = Counter(
            'searxng_cache_misses_total',
            'Total cache misses'
        )

        self.cache_size = Gauge(
            'searxng_cache_size',
            'Current cache size'
        )

        # Result metrics
        self.results_returned = Histogram(
            'searxng_results_returned',
            'Number of results returned',
            buckets=[0, 5, 10, 20, 50, 100]
        )

        self.result_quality_scores = Histogram(
            'searxng_result_quality_score',
            'Result quality scores',
            ['domain'],
            buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )

        # Engine metrics
        self.active_engines = Gauge(
            'searxng_active_engines',
            'Number of active search engines'
        )

        self.engine_success_rate = Gauge(
            'searxng_engine_success_rate',
            'Engine success rate',
            ['engine']
        )

        # Learning metrics
        self.learning_patterns_total = Gauge(
            'searxng_learning_patterns_total',
            'Total learned query patterns'
        )

    async def start_metrics_server(self):
        """Start Prometheus metrics HTTP server."""
        if not self.enable_prometheus:
            return

        try:
            start_http_server(self.metrics_port)
            logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
            logger.info(f"Metrics available at http://localhost:{self.metrics_port}/metrics")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")

    def record_search_request(
        self,
        category: str,
        domain: str,
        engine: str,
        duration: float,
        num_results: int,
        cached: bool = False,
    ):
        """Record a search request."""
        if not self.enable_prometheus:
            return

        try:
            self.search_requests_total.labels(
                category=category,
                domain=domain,
                engine=engine
            ).inc()

            self.search_duration_seconds.labels(
                category=category,
                domain=domain
            ).observe(duration)

            self.results_returned.observe(num_results)

            if cached:
                self.cache_hits_total.inc()
            else:
                self.cache_misses_total.inc()

        except Exception as e:
            logger.warning(f"Failed to record metrics: {e}")

    def record_search_error(self, error_type: str):
        """Record a search error."""
        if not self.enable_prometheus:
            return

        try:
            self.search_errors_total.labels(error_type=error_type).inc()
        except Exception as e:
            logger.warning(f"Failed to record error metric: {e}")

    def update_cache_size(self, size: int):
        """Update current cache size."""
        if not self.enable_prometheus:
            return

        try:
            self.cache_size.set(size)
        except Exception as e:
            logger.warning(f"Failed to update cache size: {e}")

    def update_active_engines(self, count: int):
        """Update active engine count."""
        if not self.enable_prometheus:
            return

        try:
            self.active_engines.set(count)
        except Exception as e:
            logger.warning(f"Failed to update active engines: {e}")

    def update_engine_performance(self, engine: str, success_rate: float):
        """Update engine performance metrics."""
        if not self.enable_prometheus:
            return

        try:
            self.engine_success_rate.labels(engine=engine).set(success_rate)
        except Exception as e:
            logger.warning(f"Failed to update engine performance: {e}")

    def record_quality_scores(self, domain: str, scores: list):
        """Record result quality scores."""
        if not self.enable_prometheus or not scores:
            return

        try:
            for score in scores:
                self.result_quality_scores.labels(domain=domain).observe(score)
        except Exception as e:
            logger.warning(f"Failed to record quality scores: {e}")


class SearXNGHealthChecker:
    """
    Health check system for SearXNG cluster.

    Monitors:
    - SearXNG service availability
    - Cache health
    - Engine status
    - Performance degradation
    """

    def __init__(self, searxng_url: str = "http://10.4.98.141:7777"):
        self.searxng_url = searxng_url
        self.health_status = {
            "searxng": "unknown",
            "cache": "unknown",
            "engines": "unknown",
            "performance": "unknown",
        }
        self.last_check = None

    async def check_searxng_health(self) -> Dict[str, Any]:
        """Check SearXNG service health."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.searxng_url}/")

                if response.status_code == 200:
                    self.health_status["searxng"] = "healthy"
                    return {
                        "status": "healthy",
                        "response_time": response.elapsed.total_seconds(),
                        "status_code": response.status_code,
                    }
                else:
                    self.health_status["searxng"] = "unhealthy"
                    return {
                        "status": "unhealthy",
                        "reason": f"HTTP {response.status_code}",
                    }
        except Exception as e:
            self.health_status["searxng"] = "unreachable"
            return {
                "status": "unreachable",
                "error": str(e),
            }

    async def check_cache_health(self, cache_size: int, max_size: int = 10000) -> Dict[str, Any]:
        """Check cache health."""
        try:
            usage_percent = (cache_size / max_size) * 100

            if usage_percent > 90:
                health = "warning"
                message = f"Cache nearly full ({usage_percent:.1f}%)"
            elif usage_percent > 95:
                health = "critical"
                message = f"Cache critically full ({usage_percent:.1f}%)"
            else:
                health = "healthy"
                message = f"Cache usage normal ({usage_percent:.1f}%)"

            self.health_status["cache"] = health

            return {
                "status": health,
                "cache_size": cache_size,
                "max_size": max_size,
                "usage_percent": usage_percent,
                "message": message,
            }
        except Exception as e:
            self.health_status["cache"] = "error"
            return {
                "status": "error",
                "error": str(e),
            }

    async def check_engine_health(self, engine_performance: Dict[str, Dict]) -> Dict[str, Any]:
        """Check engine health based on performance data."""
        try:
            unhealthy_engines = []
            total_engines = len(engine_performance)
            healthy_engines = 0

            for engine, perf in engine_performance.items():
                attempts = perf.get("attempts", 0)
                successes = perf.get("successes", 0)

                if attempts > 0:
                    success_rate = successes / attempts
                    if success_rate < 0.5:  # Less than 50% success rate
                        unhealthy_engines.append(engine)
                    else:
                        healthy_engines += 1

            if total_engines == 0:
                health = "unknown"
                message = "No engine performance data"
            elif len(unhealthy_engines) == 0:
                health = "healthy"
                message = f"All {total_engines} engines performing well"
            elif len(unhealthy_engines) < total_engines / 2:
                health = "degraded"
                message = f"{len(unhealthy_engines)} engines underperforming"
            else:
                health = "critical"
                message = f"Majority of engines ({len(unhealthy_engines)}/{total_engines}) failing"

            self.health_status["engines"] = health

            return {
                "status": health,
                "total_engines": total_engines,
                "healthy_engines": healthy_engines,
                "unhealthy_engines": unhealthy_engines,
                "message": message,
            }
        except Exception as e:
            self.health_status["engines"] = "error"
            return {
                "status": "error",
                "error": str(e),
            }

    async def comprehensive_health_check(
        self,
        cache_size: int = 0,
        engine_performance: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive health check.

        Returns overall health status with details for each component.
        """
        self.last_check = datetime.now().isoformat()

        # Run all health checks in parallel
        results = await asyncio.gather(
            self.check_searxng_health(),
            self.check_cache_health(cache_size),
            self.check_engine_health(engine_performance or {}),
            return_exceptions=True,
        )

        searxng_health, cache_health, engine_health = results

        # Determine overall health
        component_statuses = [
            searxng_health.get("status", "unknown"),
            cache_health.get("status", "unknown"),
            engine_health.get("status", "unknown"),
        ]

        if "unreachable" in component_statuses or "critical" in component_statuses:
            overall_status = "critical"
        elif "unhealthy" in component_statuses or "warning" in component_statuses:
            overall_status = "degraded"
        elif "error" in component_statuses:
            overall_status = "error"
        else:
            overall_status = "healthy"

        return {
            "overall_status": overall_status,
            "timestamp": self.last_check,
            "components": {
                "searxng": searxng_health,
                "cache": cache_health,
                "engines": engine_health,
            },
            "health_status": self.health_status,
        }

    def get_health_summary(self) -> Dict[str, str]:
        """Get quick health summary."""
        return {
            "last_check": self.last_check or "never",
            "overall": self.health_status,
        }


# Global instances
_metrics_instance: Optional[SearXNGMetrics] = None
_health_checker: Optional[SearXNGHealthChecker] = None


def get_metrics(enable_prometheus: bool = True, metrics_port: int = 9090) -> SearXNGMetrics:
    """Get or create global metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = SearXNGMetrics(enable_prometheus=enable_prometheus, metrics_port=metrics_port)
    return _metrics_instance


def get_health_checker(searxng_url: str = "http://10.4.98.141:7777") -> SearXNGHealthChecker:
    """Get or create global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = SearXNGHealthChecker(searxng_url=searxng_url)
    return _health_checker
