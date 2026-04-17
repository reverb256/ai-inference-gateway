"""
SearXNG Result Clustering

Groups search results into semantic topics using clustering algorithms.
Features automatic topic label generation and hierarchical clustering.

Features:
- DBSCAN clustering for automatic topic discovery
- Keyword-based topic label generation
- Hierarchical clustering for subtopics
- Dendrogram visualization data
"""

import asyncio
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import scikit-learn
try:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available - clustering features disabled")

# Try to import numpy for similarity computation
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


@dataclass
class Cluster:
    """A cluster of similar search results."""

    cluster_id: int
    label: str
    keywords: List[str]
    results: List[Dict[str, Any]]
    score: float
    parent_id: Optional[int] = None
    children_ids: List[int] = None

    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []

    @property
    def size(self) -> int:
        """Number of results in cluster."""
        return len(self.results)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "keywords": self.keywords,
            "size": self.size,
            "score": self.score,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "sample_results": [
                {"title": r.get("title", ""), "url": r.get("url", "")}
                for r in self.results[:3]
            ],
        }


class ResultClusterer:
    """
    Clusters search results into semantic topics.

    Uses DBSCAN for automatic cluster discovery and generates
    human-readable labels from frequent keywords.
    """

    def __init__(
        self,
        min_cluster_size: int = 2,
        eps: float = 0.3,
        max_keywords: int = 5,
        language: str = "english",
    ):
        """
        Initialize result clusterer.

        Args:
            min_cluster_size: Minimum results per cluster
            eps: DBSCAN epsilon parameter (similarity threshold)
            max_keywords: Maximum keywords per topic label
            language: Language for stopword removal
        """
        self.min_cluster_size = min_cluster_size
        self.eps = eps
        self.max_keywords = max_keywords
        self.language = language

        # TF-IDF vectorizer for text similarity
        self.vectorizer = None

        if SKLEARN_AVAILABLE:
            # Common stopwords for multiple languages
            stopwords = self._get_stopwords()
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=stopwords,
                ngram_range=(1, 2),
                min_df=1,
            )

    def _get_stopwords(self) -> List[str]:
        """Get stopwords for the configured language."""
        # Basic English stopwords
        english_stops = [
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
            "the",
            "this",
            "but",
            "they",
            "you",
            "your",
            "how",
            "what",
            "when",
            "where",
            "why",
            "which",
            "who",
            "can",
            "or",
            "if",
            "we",
            "do",
            "not",
            "more",
            "also",
            "into",
            "over",
            "then",
        ]

        if self.language == "english":
            return english_stops
        return english_stops  # Default to English

    def _extract_text(self, result: Dict[str, Any]) -> str:
        """Extract searchable text from a result."""
        title = result.get("title", "")
        content = result.get("content", result.get("snippet", ""))

        # Combine title and content
        text = f"{title} {content}".strip()

        # Clean up text
        text = text.lower()

        # Remove common URL patterns
        import re

        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)

        # Remove extra whitespace
        text = " ".join(text.split())

        return text

    def _generate_cluster_label(
        self, cluster_results: List[Dict[str, Any]], tfidf_matrix: np.ndarray, indices: List[int]
    ) -> Tuple[str, List[str]]:
        """
        Generate a human-readable label for a cluster.

        Args:
            cluster_results: Results in the cluster
            tfidf_matrix: TF-IDF matrix for all results
            indices: Indices of results in this cluster

        Returns:
            Tuple of (label, keywords)
        """
        if not SKLEARN_AVAILABLE or not NUMPY_AVAILABLE:
            # Fallback: use word frequency
            return self._generate_label_from_frequency(cluster_results)

        # Get TF-IDF scores for this cluster
        if len(indices) == 0:
            return "Uncategorized", []

        cluster_tfidf = tfidf_matrix[indices]

        # Sum TF-IDF scores across all documents in cluster
        mean_scores = np.mean(cluster_tfidf, axis=0).flatten()

        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()

        # Get top keywords by TF-IDF score
        top_indices = mean_scores.argsort()[-self.max_keywords :][::-1]
        keywords = [feature_names[i] for i in top_indices if mean_scores[i] > 0]

        if not keywords:
            return "Uncategorized", []

        # Generate label from top keywords
        label = " | ".join(keywords[:3])

        return label, keywords

    def _generate_label_from_frequency(
        self, cluster_results: List[Dict[str, Any]]
    ) -> Tuple[str, List[str]]:
        """Generate label from word frequency (fallback method)."""
        all_words = []

        for result in cluster_results:
            text = self._extract_text(result)
            words = text.split()
            # Filter short words and stopwords
            stops = set(self._get_stopwords())
            words = [w for w in words if len(w) > 3 and w not in stops]
            all_words.extend(words)

        if not all_words:
            return "Uncategorized", []

        # Count word frequency
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(self.max_keywords)

        keywords = [w for w, c in top_words]
        label = " | ".join(keywords[:3])

        return label, keywords

    def _calculate_cluster_score(
        self, cluster_results: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate a quality score for a cluster.

        Factors:
        - Size (more results = higher score)
        - Average quality score of results
        - Keyword coherence

        Returns:
            Score from 0.0 to 1.0
        """
        if not cluster_results:
            return 0.0

        # Size score (logarithmic to avoid bias toward large clusters)
        size_score = min(len(cluster_results) / 10.0, 1.0)

        # Quality score from results
        quality_scores = [
            r.get("quality_score", r.get("score", 0.5)) for r in cluster_results
        ]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5

        # Combine scores
        return (size_score * 0.4) + (avg_quality * 0.6)

    async def cluster_results(
        self,
        results: List[Dict[str, Any]],
        max_clusters: Optional[int] = None,
    ) -> List[Cluster]:
        """
        Cluster search results into semantic topics.

        Args:
            results: Search results to cluster
            max_clusters: Maximum number of clusters (None = auto)

        Returns:
            List of clusters
        """
        if not results:
            return []

        if not SKLEARN_AVAILABLE:
            # Fallback: simple grouping by domain
            return await self._simple_cluster_by_domain(results)

        # Extract text from results
        texts = [self._extract_text(r) for r in results]

        # Filter empty texts
        valid_indices = [i for i, t in enumerate(texts) if t]
        valid_texts = [texts[i] for i in valid_indices]

        if not valid_texts:
            return await self._simple_cluster_by_domain(results)

        try:
            # Create TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform(valid_texts)

            # Compute pairwise similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Convert to distance matrix
            distance_matrix = 1 - similarity_matrix

            # Run DBSCAN
            clustering = DBSCAN(
                eps=self.eps, min_samples=self.min_cluster_size, metric="precomputed"
            )
            labels = clustering.fit_predict(distance_matrix)

            # Group results by cluster
            clusters_dict: Dict[int, List[Dict[str, Any]]] = {}

            for idx, label in enumerate(labels):
                if label == -1:
                    # Noise point - add to "uncategorized" cluster
                    label = -2
                result_idx = valid_indices[idx]

                if label not in clusters_dict:
                    clusters_dict[label] = []

                clusters_dict[label].append(results[result_idx])

            # Create Cluster objects
            clusters = []

            for cluster_id, cluster_results in clusters_dict.items():
                if len(cluster_results) < self.min_cluster_size and cluster_id != -2:
                    continue

                # Get indices in original valid_texts
                cluster_indices = [
                    valid_indices.index(i) for i in range(len(results)) if results[i] in cluster_results
                ]

                # Generate label and keywords
                label_text, keywords = self._generate_cluster_label(
                    cluster_results, tfidf_matrix, cluster_indices
                )

                # Calculate score
                score = self._calculate_cluster_score(cluster_results)

                clusters.append(
                    Cluster(
                        cluster_id=cluster_id,
                        label=label_text,
                        keywords=keywords,
                        results=cluster_results,
                        score=score,
                    )
                )

            # Sort by score and size
            clusters.sort(key=lambda c: (c.score, c.size), reverse=True)

            # Apply max_clusters limit
            if max_clusters and len(clusters) > max_clusters:
                clusters = clusters[:max_clusters]

            # Re-index clusters
            for i, cluster in enumerate(clusters):
                cluster.cluster_id = i

            return clusters

        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            # Fallback to simple clustering
            return await self._simple_cluster_by_domain(results)

    async def _simple_cluster_by_domain(
        self, results: List[Dict[str, Any]]
    ) -> List[Cluster]:
        """
        Simple fallback clustering by domain.

        Args:
            results: Search results

        Returns:
            List of clusters
        """
        from urllib.parse import urlparse

        domain_groups: Dict[str, List[Dict[str, Any]]] = {}

        for result in results:
            url = result.get("url", "")
            try:
                domain = urlparse(url).netloc or "unknown"
            except Exception:
                domain = "unknown"

            if domain not in domain_groups:
                domain_groups[domain] = []

            domain_groups[domain].append(result)

        clusters = []

        for cluster_id, (domain, cluster_results) in enumerate(domain_groups.items()):
            if len(cluster_results) < self.min_cluster_size:
                continue

            # Generate keywords from domain
            keywords = [domain.replace("www.", "").split(".")[0]]

            # Calculate score
            score = self._calculate_cluster_score(cluster_results)

            clusters.append(
                Cluster(
                    cluster_id=cluster_id,
                    label=f"Results from {domain}",
                    keywords=keywords,
                    results=cluster_results,
                    score=score,
                )
            )

        return clusters

    async def hierarchical_cluster(
        self,
        results: List[Dict[str, Any]],
        max_depth: int = 2,
    ) -> List[Cluster]:
        """
        Perform hierarchical clustering for subtopic discovery.

        Args:
            results: Search results
            max_depth: Maximum hierarchy depth

        Returns:
            List of root clusters with children
        """
        if not SKLEARN_AVAILABLE:
            # Fallback to flat clustering
            return await self.cluster_results(results)

        # First, do flat clustering
        root_clusters = await self.cluster_results(results)

        if max_depth <= 1:
            return root_clusters

        # For each large cluster, do sub-clustering
        for cluster in root_clusters:
            if cluster.size > self.min_cluster_size * 3:
                try:
                    # Sub-cluster this cluster's results
                    sub_clusters = await self.cluster_results(
                        cluster.results, max_clusters=3
                    )

                    # Link sub-clusters
                    for sub_cluster in sub_clusters:
                        sub_cluster.parent_id = cluster.cluster_id
                        cluster.children_ids.append(sub_cluster.cluster_id)

                except Exception as e:
                    logger.error(f"Sub-clustering failed: {e}")

        return root_clusters

    def get_dendrogram_data(self, clusters: List[Cluster]) -> Dict[str, Any]:
        """
        Generate dendrogram data for visualization.

        Args:
            clusters: Cluster hierarchy

        Returns:
            Dendrogram data structure
        """
        nodes = []
        edges = []

        def add_cluster(cluster: Cluster, depth: int = 0):
            nodes.append(
                {
                    "id": cluster.cluster_id,
                    "label": cluster.label,
                    "size": cluster.size,
                    "score": cluster.score,
                    "depth": depth,
                    "keywords": cluster.keywords,
                }
            )

            for child_id in cluster.children_ids:
                edges.append({"source": cluster.cluster_id, "target": child_id})

        for cluster in clusters:
            add_cluster(cluster)

        return {
            "nodes": nodes,
            "edges": edges,
            "generated_at": datetime.now().isoformat(),
        }

    async def cluster_and_summarize(
        self,
        results: List[Dict[str, Any]],
        max_clusters: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Cluster results and generate summary.

        Args:
            results: Search results
            max_clusters: Maximum number of clusters

        Returns:
            Summary with clusters and statistics
        """
        clusters = await self.cluster_results(results, max_clusters)

        # Calculate statistics
        total_results = len(results)
        clustered_results = sum(c.size for c in clusters)
        unclustered = total_results - clustered_results

        # Get top keywords across all clusters
        all_keywords = []
        for cluster in clusters:
            all_keywords.extend(cluster.keywords)

        keyword_freq = Counter(all_keywords)

        return {
            "total_results": total_results,
            "num_clusters": len(clusters),
            "clustered_results": clustered_results,
            "unclustered_results": unclustered,
            "top_keywords": keyword_freq.most_common(10),
            "clusters": [c.to_dict() for c in clusters],
        }


def create_clusterer(
    min_cluster_size: int = 2,
    eps: float = 0.3,
    max_keywords: int = 5,
    language: str = "english",
) -> ResultClusterer:
    """
    Create result clusterer.

    Args:
        min_cluster_size: Minimum results per cluster
        eps: DBSCAN epsilon parameter
        max_keywords: Maximum keywords per topic label
        language: Language for stopword removal

    Returns:
        Configured ResultClusterer
    """
    return ResultClusterer(
        min_cluster_size=min_cluster_size,
        eps=eps,
        max_keywords=max_keywords,
        language=language,
    )
