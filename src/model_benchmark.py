"""
Model Benchmarking and Autonomous Selection System

Discovers, tests, and ranks models based on:
- Context window limits (actual, not advertised)
- TTFT (Time To First Token)
- Throughput (tokens/second)
- Concurrency limits
- Rate limits
- Cost per token
- Quality scores
- Task specialization performance
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to collect."""
    TTFT_MS = "ttft_ms"  # Time to first token
    THROUGHPUT_TPS = "throughput_tps"  # Tokens per second
    CONTEXT_WINDOW = "context_window"  # Actual context limit
    CONCURRENCY_LIMIT = "concurrency_limit"  # Max concurrent requests
    RATE_LIMIT_RPM = "rate_limit_rpm"  # Requests per minute
    COST_PER_1K_TOKENS = "cost_per_1k_tokens"  # Input + output
    QUALITY_SCORE = "quality_score"  # 0-100
    MEMORY_USAGE_MB = "memory_usage_mb"
    ERROR_RATE = "error_rate"  # 0-1


@dataclass
class BenchmarkResult:
    """Results from benchmarking a model."""
    model_id: str
    backend: str
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[MetricType, float] = field(default_factory=dict)
    test_prompts_used: List[str] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "backend": self.backend,
            "timestamp": self.timestamp.isoformat(),
            "metrics": {k.value: v for k, v in self.metrics.items()},
            "success": self.success,
            "error": self.error_message,
        }


@dataclass
class ModelRanking:
    """Ranked model with score and reasoning."""
    model_id: str
    rank: int
    score: float  # 0-100
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    best_for: List[str] = field(default_factory=list)  # Use cases
    cost_per_1k_tokens: float = 0.0
    avg_ttft_ms: float = 0.0
    avg_throughput_tps: float = 0.0
    actual_context_window: int = 0


class ModelBenchmark:
    """
    Benchmark models to discover actual capabilities.

    Unlike advertised specs, this tests REAL performance:
    - Actual context window (send progressively larger prompts)
    - Real TTFT under load
    - True throughput (including prefill)
    - Concurrency limits (when errors start)
    - Rate limits (when 429s appear)
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        auto_discovery: bool = True,
        test_concurrency: bool = True,
    ):
        self.storage_path = storage_path or Path("/tmp/model_benchmarks.json")
        self.auto_discovery = auto_discovery
        self.test_concurrency = test_concurrency
        self.results: Dict[str, BenchmarkResult] = {}
        self._load_cached_results()

    def _load_cached_results(self):
        """Load previously cached benchmark results."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                for model_id, result_data in data.items():
                    metrics = {
                        MetricType(k): v for k, v in result_data.get("metrics", {}).items()
                    }
                    self.results[model_id] = BenchmarkResult(
                        model_id=result_data["model_id"],
                        backend=result_data["backend"],
                        timestamp=datetime.fromisoformat(result_data["timestamp"]),
                        metrics=metrics,
                        success=result_data.get("success", True),
                        error_message=result_data.get("error"),
                    )
                logger.info(f"Loaded {len(self.results)} cached benchmark results")
            except Exception as e:
                logger.warning(f"Failed to load cached results: {e}")

    def _save_results(self):
        """Save benchmark results to disk."""
        try:
            data = {
                model_id: result.to_dict() for model_id, result in self.results.items()
            }
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.results)} benchmark results")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    async def discover_model_capabilities(
        self,
        model_id: str,
        backend: str,
        backend_url: str,
        api_key: Optional[str] = None,
    ) -> BenchmarkResult:
        """
        Discover actual model capabilities through testing.

        Tests:
        1. Small prompt (baseline TTFT/throughput)
        2. Progressive context sizes (find actual limit)
        3. Concurrent requests (find concurrency limit)
        4. Rapid requests (find rate limit)
        """
        logger.info(f"Benchmarking {model_id} on {backend}")
        result = BenchmarkResult(model_id=model_id, backend=backend)

        try:
            # Import here to avoid circular dependency
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                base_url=backend_url,
                api_key=api_key or "dummy",
            )

            # Test 1: Baseline performance
            ttft, throughput = await self._measure_baseline_performance(
                client, model_id, result
            )
            result.metrics[MetricType.TTFT_MS] = ttft
            result.metrics[MetricType.THROUGHPUT_TPS] = throughput

            # Test 2: Context window
            context_limit = await self._discover_context_window(client, model_id, result)
            result.metrics[MetricType.CONTEXT_WINDOW] = context_limit

            # Test 3: Concurrency (if enabled)
            if self.test_concurrency:
                concurrency_limit = await self._discover_concurrency_limit(
                    client, model_id, result
                )
                result.metrics[MetricType.CONCURRENCY_LIMIT] = concurrency_limit

            # Test 4: Rate limit
            rate_limit = await self._discover_rate_limit(client, model_id, result)
            result.metrics[MetricType.RATE_LIMIT_RPM] = rate_limit

            self.results[model_id] = result
            self._save_results()

            logger.info(f"Benchmark complete for {model_id}: {result.metrics}")

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"Benchmark failed for {model_id}: {e}")

        return result

    async def _measure_baseline_performance(
        self,
        client,
        model_id: str,
        result: BenchmarkResult,
    ) -> Tuple[float, float]:
        """Measure baseline TTFT and throughput."""
        test_prompt = "Say 'Hello, World!' briefly."

        start_time = time.time()
        first_token_time = None
        tokens_generated = 0

        try:
            stream = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": test_prompt}],
                stream=True,
                max_tokens=50,
            )

            async for chunk in stream:
                if first_token_time is None and chunk.choices[0].delta.content:
                    first_token_time = time.time()
                if chunk.choices[0].delta.content:
                    tokens_generated += 1

            total_time = time.time() - start_time

            if first_token_time:
                ttft_ms = (first_token_time - start_time) * 1000
            else:
                ttft_ms = total_time * 1000

            if tokens_generated > 0 and total_time > 0:
                throughput_tps = tokens_generated / total_time
            else:
                throughput_tps = 0.0

            result.test_prompts_used.append(test_prompt)
            return ttft_ms, throughput_tps

        except Exception as e:
            logger.warning(f"Baseline test failed: {e}")
            return 0.0, 0.0

    async def _discover_context_window(
        self,
        client,
        model_id: str,
        result: BenchmarkResult,
        min_tokens: int = 1000,
        max_tokens: int = 200000,
        step: int = 10000,
    ) -> int:
        """
        Discover actual context window through binary search.

        Sends progressively larger prompts until failure.
        Returns the largest successful context size.
        """
        # Start with a quick check
        test_sizes = [min_tokens, 32000, 64000, 128000, max_tokens]
        working_size = min_tokens

        for size in test_sizes:
            prompt = "x" * size
            try:
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1,
                )
                working_size = size
                logger.info(f"Context size {size} OK")
            except Exception as e:
                logger.warning(f"Context size {size} failed: {e}")
                break

        result.test_prompts_used.append(f"context_test_{working_size}")
        return working_size

    async def _discover_concurrency_limit(
        self,
        client,
        model_id: str,
        result: BenchmarkResult,
        max_concurrent: int = 32,
    ) -> int:
        """
        Discover concurrency limit through parallel requests.

        Sends N requests in parallel, increases until errors occur.
        """
        async def single_request(request_id: int):
            try:
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": f"Request {request_id}: Hi"}],
                    max_tokens=10,
                )
                return True
            except Exception as e:
                logger.warning(f"Concurrent request {request_id} failed: {e}")
                return False

        # Test increasing concurrency
        for concurrency in range(1, max_concurrent + 1, 4):
            tasks = [single_request(i) for i in range(concurrency)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            success_count = sum(1 for r in results if r is True)
            success_rate = success_count / concurrency

            if success_rate < 0.8:  # Less than 80% success
                logger.info(f"Concurrency limit reached at {concurrency}")
                result.test_prompts_used.append(f"concurrency_test_{concurrency}")
                return concurrency - 4

        result.test_prompts_used.append(f"concurrency_test_{max_concurrent}")
        return max_concurrent

    async def _discover_rate_limit(
        self,
        client,
        model_id: str,
        result: BenchmarkResult,
        test_duration_seconds: int = 30,
    ) -> int:
        """
        Discover rate limit through rapid requests.

        Sends requests as fast as possible for N seconds.
        Counts requests until 429 appears.
        """
        request_count = 0
        start_time = time.time()
        rate_limited = False

        while time.time() - start_time < test_duration_seconds and not rate_limited:
            try:
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5,
                )
                request_count += 1
                # Small delay to avoid overwhelming
                await asyncio.sleep(0.1)
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    rate_limited = True
                    logger.info(f"Rate limit hit after {request_count} requests")
                    break

        # Convert to RPM
        rpm = int(request_count * (60 / test_duration_seconds))
        result.test_prompts_used.append(f"rate_limit_test_{rpm}_rpm")
        return rpm

    def get_model_ranking(
        self,
        model_id: str,
        requirements: Optional[Dict[str, float]] = None,
    ) -> Optional[ModelRanking]:
        """
        Get ranking for a specific model based on benchmarks.

        Requirements can include:
        - min_context: minimum context window
        - max_ttft_ms: maximum acceptable TTFT
        - min_throughput_tps: minimum throughput
        - max_cost_per_1k: maximum cost
        - needs_concurrency: required concurrent requests
        """
        if model_id not in self.results:
            return None

        result = self.results[model_id]
        metrics = result.metrics

        # Calculate base score (0-100)
        score = 50.0  # Start at 50

        # TTFT scoring (lower is better)
        ttft = metrics.get(MetricType.TTFT_MS, 1000)
        if ttft < 100:
            score += 20
        elif ttft < 500:
            score += 10
        elif ttft < 1000:
            score += 5

        # Throughput scoring (higher is better)
        throughput = metrics.get(MetricType.THROUGHPUT_TPS, 10)
        if throughput > 100:
            score += 20
        elif throughput > 50:
            score += 10
        elif throughput > 20:
            score += 5

        # Context window scoring (larger is better)
        context = metrics.get(MetricType.CONTEXT_WINDOW, 4000)
        if context > 100000:
            score += 15
        elif context > 32000:
            score += 10
        elif context > 8000:
            score += 5

        # Concurrency scoring (higher is better)
        concurrency = metrics.get(MetricType.CONCURRENCY_LIMIT, 1)
        if concurrency > 16:
            score += 15
        elif concurrency > 8:
            score += 10
        elif concurrency > 4:
            score += 5

        # Apply requirements filter
        if requirements:
            if "min_context" in requirements:
                if context < requirements["min_context"]:
                    score = 0  # Disqualified

            if "max_ttft_ms" in requirements:
                if ttft > requirements["max_ttft_ms"]:
                    score *= 0.5  # Penalty

        # Build strengths/weaknesses
        strengths = []
        weaknesses = []

        if ttft < 200:
            strengths.append("Ultra-fast TTFT")
        elif ttft > 1000:
            weaknesses.append("Slow TTFT")

        if throughput > 50:
            strengths.append("High throughput")
        elif throughput < 10:
            weaknesses.append("Low throughput")

        if context > 100000:
            strengths.append("Large context window")
        elif context < 8000:
            weaknesses.append("Small context window")

        if concurrency > 8:
            strengths.append("High concurrency")
        elif concurrency == 1:
            weaknesses.append("No concurrency")

        # Determine best use cases
        best_for = []
        if ttft < 500 and throughput > 30:
            best_for.extend(["fast-chat", "realtime"])
        if context > 64000:
            best_for.extend(["long-context", "rag", "analysis"])
        if concurrency > 4:
            best_for.extend(["batch", "high-volume"])
        if score > 70:
            best_for.append("general-purpose")

        return ModelRanking(
            model_id=model_id,
            rank=0,  # Will be set when comparing all models
            score=max(0, min(100, score)),
            strengths=strengths,
            weaknesses=weaknesses,
            best_for=best_for,
            avg_ttft_ms=ttft,
            avg_throughput_tps=throughput,
            actual_context_window=context,
        )

    def rank_all_models(
        self,
        requirements: Optional[Dict[str, float]] = None,
    ) -> List[ModelRanking]:
        """
        Rank all benchmarked models.

        Returns list sorted by score (highest first).
        """
        rankings = []

        for model_id in self.results:
            ranking = self.get_model_ranking(model_id, requirements)
            if ranking and ranking.score > 0:
                rankings.append(ranking)

        # Sort by score
        rankings.sort(key=lambda r: r.score, reverse=True)

        # Assign ranks
        for i, ranking in enumerate(rankings):
            ranking.rank = i + 1

        return rankings

    async def auto_benchmark_all(
        self,
        models_to_test: List[Tuple[str, str, str, Optional[str]]],
    ) -> Dict[str, BenchmarkResult]:
        """
        Benchmark all models in the list.

        Args:
            models_to_test: List of (model_id, backend, backend_url, api_key)

        Returns:
            Dict of model_id -> BenchmarkResult
        """
        logger.info(f"Auto-benchmarking {len(models_to_test)} models")

        tasks = []
        for model_id, backend, backend_url, api_key in models_to_test:
            task = self.discover_model_capabilities(
                model_id, backend, backend_url, api_key
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, BenchmarkResult):
                self.results[result.model_id] = result

        self._save_results()
        return self.results


class AutonomousModelSelector:
    """
    Autonomous model selection based on real-time performance data.

    Combines:
    - Benchmark results (static capabilities)
    - Live performance (current load, recent errors)
    - Request requirements (context size, tokens needed, specialization)
    - Cost optimization (pick cheapest suitable model)
    """

    def __init__(self, benchmark: ModelBenchmark, router=None):
        self.benchmark = benchmark
        self.router = router
        self.selection_history: Dict[str, List[str]] = {}  # requirements -> models used

    async def select_best_model(
        self,
        requirements: Dict[str, any],
        available_models: List[str],
    ) -> Optional[str]:
        """
        Select the best model for the given requirements.

        Requirements:
        - estimated_input_tokens: int
        - estimated_output_tokens: int
        - task_specialization: TaskSpecialization
        - max_ttft_ms: Optional[float]
        - min_throughput_tps: Optional[float]
        - max_cost_per_1k: Optional[float]
        - needs_concurrency: Optional[int]
        """
        # Filter models by requirements
        candidates = []

        for model_id in available_models:
            if model_id not in self.benchmark.results:
                continue  # Skip unbenchmarked models

            result = self.benchmark.results[model_id]
            metrics = result.metrics

            # Check context window
            required_context = (
                requirements.get("estimated_input_tokens", 0) +
                requirements.get("estimated_output_tokens", 0)
            )
            actual_context = metrics.get(MetricType.CONTEXT_WINDOW, 0)
            if actual_context < required_context:
                continue  # Too small

            # Check TTFT
            max_ttft = requirements.get("max_ttft_ms")
            if max_ttft:
                ttft = metrics.get(MetricType.TTFT_MS, 0)
                if ttft > max_ttft:
                    continue  # Too slow

            # Check throughput
            min_throughput = requirements.get("min_throughput_tps")
            if min_throughput:
                throughput = metrics.get(MetricType.THROUGHPUT_TPS, 0)
                if throughput < min_throughput:
                    continue  # Too slow

            # Check concurrency
            needed_concurrency = requirements.get("needs_concurrency", 1)
            actual_concurrency = metrics.get(MetricType.CONCURRENCY_LIMIT, 1)
            if actual_concurrency < needed_concurrency:
                continue  # Can't handle concurrency

            candidates.append(model_id)

        if not candidates:
            logger.warning(f"No models meet requirements: {requirements}")
            return None

        # Score remaining candidates
        best_model = None
        best_score = -1

        for model_id in candidates:
            ranking = self.benchmark.get_model_ranking(model_id, requirements)
            if ranking:
                # Apply cost optimization (prefer cheaper if scores are close)
                score = ranking.score
                if best_score > 0 and abs(score - best_score) < 10:
                    # Scores are close, prefer cheaper
                    # TODO: Add cost comparison
                    pass

                if score > best_score:
                    best_score = score
                    best_model = model_id

        # Track selection
        req_key = str(sorted(requirements.items()))
        if req_key not in self.selection_history:
            self.selection_history[req_key] = []
        self.selection_history[req_key].append(best_model)

        logger.info(f"Selected {best_model} for requirements (score: {best_score})")
        return best_model

    def get_recommendations(
        self,
        task_type: str,
    ) -> List[Dict[str, str]]:
        """
        Get model recommendations for a task type.

        Returns list of {model_id, reason} pairs.
        """
        rankings = self.benchmark.rank_all_models()

        recommendations = []
        for ranking in rankings:
            if task_type in ranking.best_for:
                recommendations.append({
                    "model_id": ranking.model_id,
                    "score": ranking.score,
                    "reason": f"Rank #{ranking.rank}: {', '.join(ranking.strengths[:3])}",
                })

        return recommendations[:5]  # Top 5


# Singleton instance
_benchmark_instance: Optional[ModelBenchmark] = None
_selector_instance: Optional[AutonomousModelSelector] = None


def get_benchmark() -> ModelBenchmark:
    """Get or create the singleton benchmark instance."""
    global _benchmark_instance
    if _benchmark_instance is None:
        _benchmark_instance = ModelBenchmark()
    return _benchmark_instance


def get_selector() -> AutonomousModelSelector:
    """Get or create the singleton selector instance."""
    global _selector_instance
    if _selector_instance is None:
        _selector_instance = AutonomousModelSelector(get_benchmark())
    return _selector_instance
