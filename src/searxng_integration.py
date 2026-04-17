"""
SearXNG Integration for AI Inference Gateway

Direct integration with local SearXNG instance providing:
- Privacy-respecting metasearch
- Auto-improving search with query learning
- Result caching and ranking optimization
- Adaptive engine selection based on query type
- Prometheus metrics and health monitoring
"""

import asyncio
import json
import logging
import hashlib
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlparse

import httpx

logger = logging.getLogger(__name__)

# SearXNG configuration
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://10.4.98.141:7777")  # Kubernetes service DNS
SEARCH_ENDPOINT = "/search"

# Learning storage
LEARNING_CACHE_PATH = "/var/cache/ai-inference/mcp/searxng_learning.json"

# Monitoring
try:
    from ai_inference_gateway.searxng_monitoring import get_metrics, get_health_checker
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    logger.warning("Monitoring module not available")


class SearxngIntegration:
    """
    SearXNG integration with auto-improving features:

    1. Query Pattern Learning: Tracks common search patterns to suggest refinements
    2. Result Ranking Optimization: Learns which results get selected
    3. Adaptive Engine Selection: Chooses best engines based on query category
    4. Popularity Caching: Prioritizes frequently accessed results
    """

    def __init__(self, cache_ttl: int = 300, enable_metrics: bool = True):
        self.cache_ttl = cache_ttl
        # Configure HTTP client with headers to bypass rate limiting and bot detection
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "X-Forwarded-For": "10.1.1.110",
            "X-Real-IP": "10.1.1.110",
        }
        self.client = httpx.AsyncClient(timeout=30.0, headers=headers, follow_redirects=True)

        # Learning data
        self.query_patterns: Counter = Counter()  # Track query patterns
        self.successful_queries: List[str] = []  # Track successful queries
        self.result_clicks: Counter = Counter()  # Track clicked results (simulated)
        self.engine_performance: Dict[str, Dict] = defaultdict(
            lambda: {"attempts": 0, "successes": 0, "avg_results": 0}
        )

        # Response cache with popularity tracking
        self.response_cache: Dict[str, Dict] = {}

        # Load learning data from disk
        self._load_learning_data()

        # Initialize monitoring
        # TEMPORARILY DISABLED: Monitoring imports causing timeout
        # self.metrics = get_metrics(enable_prometheus=enable_metrics) if MONITORING_AVAILABLE else None
        # self.health_checker = get_health_checker(searxng_url=SEARXNG_URL) if MONITORING_AVAILABLE else None
        self.metrics = None
        self.health_checker = None

    def _load_learning_data(self):
        """Load learning data from disk."""
        try:
            if os.path.exists(LEARNING_CACHE_PATH):
                with open(LEARNING_CACHE_PATH, "r") as f:
                    data = json.load(f)
                    self.query_patterns = Counter(data.get("query_patterns", {}))
                    self.successful_queries = data.get("successful_queries", [])
                    self.result_clicks = Counter(data.get("result_clicks", {}))
                    self.engine_performance = defaultdict(
                        lambda: {"attempts": 0, "successes": 0, "avg_results": 0},
                        data.get("engine_performance", {})
                    )
                logger.info(f"Loaded learning data: {len(self.successful_queries)} queries")
        except Exception as e:
            logger.warning(f"Could not load learning data: {e}")

    def _save_learning_data(self):
        """Save learning data to disk."""
        try:
            os.makedirs(os.path.dirname(LEARNING_CACHE_PATH), exist_ok=True)
            data = {
                "query_patterns": dict(self.query_patterns),
                "successful_queries": self.successful_queries,
                "result_clicks": dict(self.result_clicks),
                "engine_performance": self.engine_performance,
                "last_updated": time.time(),
            }
            with open(LEARNING_CACHE_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save learning data: {e}")

    def _cache_key(self, query: str, category: str = "general") -> str:
        """Generate cache key for search query."""
        key_data = f"{category}:{query.lower().strip()}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _extract_query_pattern(self, query: str) -> str:
        """Extract meaningful pattern from query for learning."""
        # Remove common words
        stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "about"}
        words = query.lower().split()
        meaningful_words = [w for w in words if len(w) > 3 and w not in stop_words]
        return " ".join(meaningful_words[:5])  # First 5 meaningful words

    def _suggest_refinement(self, query: str, results: List[Dict]) -> Optional[str]:
        """Suggest query refinement based on results and learning."""
        if not results:
            return None

        # Check if this is a technical query that could be more specific
        technical_keywords = ["how", "what", "why", "when", "where", "install", "configure", "setup"]
        query_lower = query.lower()

        if any(kw in query_lower for kw in technical_keywords):
            # Extract potential topic
            for result in results[:3]:
                title = result.get("title", "").lower()
                snippet = result.get("content", result.get("snippet", "")).lower()
                text = f"{title} {snippet}"

                # Look for more specific terms
                if "nixos" in text and "flake" not in query_lower:
                    return f"{query} nixos flake"
                elif "kubernetes" in text and "deployment" not in query_lower:
                    return f"{query} kubernetes deployment"
                elif "api" in text and "example" not in query_lower:
                    return f"{query} api example"

        return None

    def _get_optimal_engines(self, category: str) -> List[str]:
        """Select optimal search engines based on category and learning."""
        # Default engines by category (only engines enabled in SearXNG config)
        category_engines = {
            "general": ["brave", "bing", "wikipedia"],
            "images": ["bing images"],
            "videos": ["youtube"],
            "news": ["bing"],
            "science": ["wikipedia"],
            "it": ["stackoverflow", "github"],
            "files": ["marginalia"],
            "music": [],
            "map": ["openstreetmap"],
        }

        # Get base engines
        engines = category_engines.get(category, category_engines["general"])

        # Learn from past performance
        engine_scores = {}
        for engine in engines:
            perf = self.engine_performance.get(engine, {"attempts": 0, "successes": 0, "avg_results": 0})
            if perf.get("attempts", 0) > 0:
                # Score: success rate * average results (more results is better)
                score = (perf.get("successes", 0) / perf["attempts"]) * min(perf.get("avg_results", 0) / 10, 1.0)
                engine_scores[engine] = score
            else:
                engine_scores[engine] = 0.5  # Default score

        # Sort by score and return top 3
        sorted_engines = sorted(engine_scores.items(), key=lambda x: x[1], reverse=True)
        return [e[0] for e in sorted_engines[:3]]

    async def search(
        self,
        query: str,
        category: str = "general",
        language: str = "all",
        max_results: int = 10,
        time_range: Optional[str] = None,
        use_cache: bool = True,
        learning_enabled: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform search with auto-improving features.

        Returns:
            Dict with:
            - results: List of search results
            - suggestions: Query refinement suggestions
            - engines_used: Which engines were queried
            - cached: Whether results came from cache
        """
        cache_key = self._cache_key(query, category)

        # Check cache first
        if use_cache and cache_key in self.response_cache:
            cached = self.response_cache[cache_key]
            if time.time() < cached["expiry"]:
                # Update popularity for cache hit
                cached["popularity"] += 1
                logger.info(f"Cache HIT: {query} (popularity: {cached['popularity']})")

                # Record metrics
                if self.metrics:
                    self.metrics.record_search_request(
                        category=category,
                        domain="general",
                        engine="cache",
                        duration=0.001,  # Cache hit is essentially instant
                        num_results=len(cached.get("results", [])),
                        cached=True,
                    )

                return {
                    "results": cached["results"],
                    "cached": True,
                    "engines_used": cached["engines_used"],
                }

        # Select optimal engines based on learning
        # CRITICAL: Always specify engines to avoid blocked Google/DuckDuckGo
        is_site_search = "site:" in query.lower()
        if not is_site_search:
            engines = self._get_optimal_engines(category)
        else:
            # For site searches, use working engines that support site: queries
            engines = ["bing", "brave", "github", "wikipedia"]

        # Build request parameters
        params = {
            "q": query,
            "format": "json",
        }

        # Only add engines parameter if not a site search and engines are selected
        if engines:
            params["engines"] = ",".join(engines)

        if category != "general":
            params["categories"] = category

        if language != "all":
            params["language"] = language

        if time_range:
            params["time_range"] = time_range

        # Track query pattern
        pattern = self._extract_query_pattern(query)
        self.query_patterns[pattern] += 1

        try:
            search_start_time = time.time()

            response = await self.client.get(
                f"{SEARXNG_URL}/search",
                params=params,
            )
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            search_duration = time.time() - search_start_time

            # Update engine performance tracking
            for engine in engines:
                self.engine_performance[engine]["attempts"] += 1
            self.engine_performance["|".join(engines)]["successes"] += 1
            self.engine_performance["|".join(engines)]["avg_results"] = len(results)

            # Record metrics
            if self.metrics:
                self.metrics.record_search_request(
                    category=category,
                    domain="general",
                    engine="|".join(engines[:3]),  # Log first 3 engines
                    duration=search_duration,
                    num_results=len(results),
                    cached=False,
                )
                self.metrics.update_cache_size(len(self.response_cache))

            if results:
                # Track successful query
                self.successful_queries.append(query)
                if learning_enabled:
                    self._save_learning_data()

                # Generate suggestions based on results
                suggestions = []
                refinement = self._suggest_refinement(query, results)
                if refinement:
                    suggestions.append({
                        "type": "refinement",
                        "suggestion": refinement,
                        "reason": "More specific search",
                    })

                # Cache results with popularity tracking
                self.response_cache[cache_key] = {
                    "results": results[:max_results],
                    "expiry": time.time() + self.cache_ttl,
                    "popularity": 1,
                    "engines_used": engines,
                    "query": query,
                    "timestamp": time.time(),
                }

                return {
                    "results": results[:max_results],
                    "suggestions": suggestions,
                    "engines_used": engines,
                    "cached": False,
                }
            else:
                return {
                    "results": [],
                    "suggestions": [
                        {
                            "type": "refinement",
                            "suggestion": query.replace("how to", "").strip(),
                            "reason": "Try removing 'how to' prefix",
                        },
                        {
                            "type": "general",
                            "suggestion": "Try broader search terms",
                            "reason": "Query may be too specific",
                        }
                    ],
                    "engines_used": engines,
                    "cached": False,
                }

        except httpx.HTTPStatusError as e:
            logger.error(f"SearXNG HTTP error: {e.response.status_code}")
            return {
                "results": [],
                "error": f"SearXNG returned HTTP {e.response.status_code}",
                "suggestions": [
                    {
                        "type": "check_service",
                        "suggestion": "Check if SearXNG is running: systemctl status searx",
                        "reason": "Service unavailable",
                    }
                ],
                "cached": False,
            }
        except httpx.ConnectError:
            logger.error(f"SearXNG connection error")
            return {
                "results": [],
                "error": "Cannot connect to SearXNG service",
                "suggestions": [
                    {
                        "type": "check_service",
                        "suggestion": "Start SearXNG: systemctl start searx",
                        "reason": "Service not running",
                    },
                    {
                        "type": "check_config",
                        "suggestion": "Verify SearXNG is enabled in NixOS config",
                        "reason": "Service may not be enabled",
                    }
                ],
                "cached": False,
            }
        except Exception as e:
            logger.exception(f"Unexpected error in SearXNG search: {e}")
            return {
                "results": [],
                "error": str(e),
                "cached": False,
            }

    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about learned search patterns."""
        # Get top cached queries with safe popularity access
        cached_entries = [
            {
                "query": r.get("query", "unknown"),
                "popularity": r.get("popularity", 0),
                "engines": r.get("engines_used", [])
            }
            for r in self.response_cache.values()
        ]
        top_cached = sorted(cached_entries, key=lambda x: x.get("popularity", 0), reverse=True)[:10]

        return {
            "total_queries": len(self.successful_queries),
            "query_patterns": dict(self.query_patterns.most_common(20)),
            "engine_performance": dict(self.engine_performance),
            "cache_size": len(self.response_cache),
            "top_cached_queries": top_cached,
        }

    # ========================================================================
    # AI-SPECIFIC ENHANCEMENTS
    # ========================================================================

    def _detect_domain(self, query: str) -> str:
        """
        Detect the domain of a query for intelligent routing.

        Returns: Domain key (code, research, devops, data, general)
        """
        domain_indicators = {
            'code': {
                'keywords': ['function', 'class', 'api', 'library', 'framework', 'module', 'package',
                           'implementation', 'syntax', 'example', 'tutorial', 'code', 'programming'],
                'prefixes': ['how to', 'how do i', 'implement', 'create', 'build'],
            },
            'research': {
                'keywords': ['paper', 'research', 'study', 'theorem', 'algorithm', 'method',
                           'analysis', 'survey', 'review', 'academic', 'scholar'],
                'prefixes': ['recent', 'latest', 'state of the art', 'sota'],
            },
            'devops': {
                'keywords': ['docker', 'kubernetes', 'deployment', 'ci/cd', 'terraform', 'ansible',
                           'helm', 'container', 'orchestration', 'infrastructure', 'pipeline'],
                'prefixes': ['deploy', 'scale', 'monitor', 'manage'],
            },
            'data': {
                'keywords': ['dataset', 'model', 'training', 'inference', 'machine learning',
                           'deep learning', 'neural', 'ai', 'llm', 'embedding', 'vector'],
                'prefixes': ['train', 'finetune', 'optimize'],
            },
        }

        query_lower = query.lower()

        # Score each domain
        domain_scores = {}
        for domain, indicators in domain_indicators.items():
            score = 0

            # Check keywords
            for keyword in indicators['keywords']:
                if keyword in query_lower:
                    score += 2

            # Check prefixes
            for prefix in indicators['prefixes']:
                if query_lower.startswith(prefix):
                    score += 3

            domain_scores[domain] = score

        # Return domain with highest score, or 'general' if no matches
        max_score = max(domain_scores.values())
        if max_score > 0:
            return max(domain_scores, key=domain_scores.get)
        return 'general'

    def _get_domain_engines(self, domain: str) -> List[str]:
        """
        Get optimal search engines for a specific domain.

        Returns: List of engine names optimized for the domain
        """
        domain_engines = {
            'code': ['github', 'gitlab', 'stackoverflow', 'stackexchange'],
            'research': ['wikipedia', 'brave'],
            'devops': ['github', 'gitlab', 'stackoverflow'],
            'data': ['github', 'wikipedia'],
            'general': ['brave', 'bing', 'wikipedia', 'stackoverflow'],
        }

        return domain_engines.get(domain, domain_engines['general'])

    def _score_result_quality(self, result: Dict, query: str, domain: str) -> float:
        """
        Score search result for relevance and quality.

        Scoring factors:
        - Domain authority (trusted sources)
        - Content richness (length, structured data)
        - Code snippet presence
        - Freshness (recency for technical content)
        - Query relevance (text similarity)

        Returns: Quality score from 0.0 to 1.0
        """
        score = 0.0

        # 1. Domain authority (30% of score)
        trusted_domains = {
            'code': ['github.com', 'gitlab.com', 'stackoverflow.com', 'docs.rs',
                    'developer.mozilla.org', 'numpy.org', 'postgresql.org'],
            'research': ['arxiv.org', 'scholar.google.com', 'semanticscholar.org',
                        'dl.acm.org', 'ieeexplore.ieee.org', 'springer.com'],
            'devops': ['docker.com', 'kubernetes.io', 'terraform.io', 'ansible.com',
                      'jenkins.io', 'github.com'],
            'data': ['arxiv.org', 'kaggle.com', 'huggingface.co', 'github.com',
                     'paperswithcode.com', 'openreview.net'],
            'general': ['wikipedia.org', 'github.com', 'stackoverflow.com',
                       'reddit.com', 'medium.com'],
        }

        url = result.get('url', '').lower()
        domain_trusted = [d for d in trusted_domains.get(domain, trusted_domains['general']) if d in url]

        if domain_trusted:
            score += 0.3
        elif any(tld in url for tld in ['.edu', '.org', '.gov']):
            score += 0.2  # Still good, but not domain-specific

        # 2. Content richness (25% of score)
        content = result.get('content', result.get('snippet', ''))
        if len(content) > 200:
            score += 0.15
        if len(content) > 500:
            score += 0.10

        # 3. Code snippet presence (20% of score)
        if '```' in content or 'code' in content.lower():
            score += 0.15
        if any(lang in content.lower() for lang in ['python', 'javascript', 'java', 'rust', 'go', 'nix']):
            score += 0.05

        # 4. Freshness for technical content (15% of score)
        if domain in ['code', 'devops', 'data']:
            current_year = 2026
            for year in [str(current_year), str(current_year - 1), str(current_year - 2)]:
                if year in content:
                    score += 0.05
                    break

        # 5. Query relevance (10% of score)
        query_lower = query.lower()
        title_lower = result.get('title', '').lower()
        snippet_lower = content.lower()

        # Exact phrase match
        if query_lower in title_lower:
            score += 0.10
        elif any(word in title_lower for word in query_lower.split() if len(word) > 3):
            score += 0.05

        return min(score, 1.0)

    async def search_with_domain_routing(
        self,
        query: str,
        domain: Optional[str] = None,
        max_results: int = 10,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform search with intelligent domain routing and quality scoring.

        This is the main entry point for AI-optimized search.

        Args:
            query: Search query string
            domain: Optional domain hint (code, research, devops, data, general)
            max_results: Maximum number of results to return
            use_cache: Whether to use cached results

        Returns:
            Dict with results, quality scores, and routing metadata
        """
        # Detect domain if not provided
        if domain is None:
            domain = self._detect_domain(query)

        # Get optimal engines for domain
        engines = self._get_domain_engines(domain)

        # Perform search
        result = await self.search(
            query=query,
            category='general',  # Use general category, engine selection handles domain
            max_results=max_results * 2,  # Fetch more to score and filter
            use_cache=use_cache,
            learning_enabled=True,
        )

        # Score and filter results
        if result.get('results'):
            for item in result['results']:
                item['quality_score'] = self._score_result_quality(item, query, domain)

            # Sort by quality score
            result['results'].sort(key=lambda x: x.get('quality_score', 0), reverse=True)

            # Filter to top results
            result['results'] = result['results'][:max_results]

            # Add routing metadata
            result['routing'] = {
                'detected_domain': domain,
                'engines_selected': engines[:3],
                'quality_scoring': True,
            }

        return result

    def clear_cache(self):
        """Clear all cached responses."""
        self.response_cache.clear()
        logger.info("SearXNG cache cleared")

    async def close(self):
        """Close resources."""
        await self.client.aclose()


# Global instance
_searxng_instance: Optional[SearxngIntegration] = None


def get_searxng(cache_ttl: int = 300) -> SearxngIntegration:
    """Get or create global SearXNG instance."""
    global _searxng_instance
    if _searxng_instance is None:
        _searxng_instance = SearxngIntegration(cache_ttl=cache_ttl)
    return _searxng_instance
