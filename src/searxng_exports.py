"""
SearXNG Search Result Export Formats

Supports exporting search results to various formats:
- JSON: Structured data export
- CSV: Spreadsheet-compatible export
- RSS/ATOM: Feed syndication formats
"""

import csv
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Dict, List, Optional
from xml.dom import minidom

logger = logging.getLogger(__name__)


class SearchResultsExporter:
    """
    Export search results to various formats.

    Supports JSON, CSV, RSS, and ATOM formats.
    """

    def __init__(self, base_url: str = "http://10.0.0.102:8080"):
        """
        Initialize exporter.

        Args:
            base_url: Base URL for generating links
        """
        self.base_url = base_url

    def to_json(
        self,
        results: List[Dict[str, Any]],
        query: str,
        pretty: bool = True,
        include_metadata: bool = True,
    ) -> str:
        """
        Export results to JSON format.

        Args:
            results: Search results
            query: Original search query
            pretty: Pretty-print JSON
            include_metadata: Include routing metadata

        Returns:
            JSON string
        """
        export_data = {
            "query": query,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_results": len(results),
            "results": [],
        }

        for result in results:
            result_data = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", result.get("snippet", "")),
                "score": result.get("score", result.get("quality_score", 0.0)),
                "engine": result.get("engine", ""),
            }

            if include_metadata:
                result_data["metadata"] = {
                    k: v
                    for k, v in result.items()
                    if k not in ["title", "url", "content", "snippet", "score", "quality_score", "engine"]
                }

            export_data["results"].append(result_data)

        if pretty:
            return json.dumps(export_data, indent=2, ensure_ascii=False)
        return json.dumps(export_data, ensure_ascii=False)

    def to_csv(
        self,
        results: List[Dict[str, Any]],
        query: str,
        include_metadata: bool = False,
    ) -> str:
        """
        Export results to CSV format.

        Args:
            results: Search results
            query: Original search query
            include_metadata: Include additional columns

        Returns:
            CSV string
        """
        output = StringIO()

        # Define base columns
        fieldnames = ["title", "url", "content", "score", "engine"]

        if include_metadata:
            # Add common metadata fields
            metadata_fields = set()
            for result in results:
                for key in result.keys():
                    if key not in fieldnames and key not in ["snippet", "quality_score"]:
                        metadata_fields.add(key)
            fieldnames.extend(sorted(metadata_fields))

        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")

        writer.writeheader()

        for result in results:
            row = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", result.get("snippet", "")),
                "score": result.get("score", result.get("quality_score", 0.0)),
                "engine": result.get("engine", ""),
            }

            # Add metadata fields
            if include_metadata:
                for key in fieldnames[5:]:  # Skip base columns
                    row[key] = result.get(key, "")

            writer.writerow(row)

        return output.getvalue()

    def to_rss(
        self,
        results: List[Dict[str, Any]],
        query: str,
        channel_title: Optional[str] = None,
        channel_description: Optional[str] = None,
    ) -> str:
        """
        Export results to RSS 2.0 format.

        Args:
            results: Search results
            query: Original search query
            channel_title: Override channel title
            channel_description: Override channel description

        Returns:
            RSS XML string
        """
        # Create RSS element
        rss = ET.Element("rss", version="2.0")
        rss.set("xmlns:atom", "http://www.w3.org/2005/Atom")

        # Create channel
        channel = ET.SubElement(rss, "channel")

        # Channel metadata
        ET.SubElement(channel, "title").text = channel_title or f"Search: {query}"
        ET.SubElement(channel, "description").text = (
            channel_description or f"Results for '{query}' from SearXNG"
        )
        ET.SubElement(channel, "link").text = f"{self.base_url}/search?q={query}"
        ET.SubElement(channel, "language").text = "en-us"
        ET.SubElement(channel, "lastBuildDate").text = datetime.now(timezone.utc).strftime(
            "%a, %d %b %Y %H:%M:%S %z"
        )

        # Atom self link
        atom_link = ET.SubElement(channel, "{http://www.w3.org/2005/Atom}link")
        atom_link.set("rel", "self")
        atom_link.set("href", f"{self.base_url}/export/rss?q={query}")
        atom_link.set("type", "application/rss+xml")

        # Add items
        for result in results:
            item = ET.SubElement(channel, "item")

            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", result.get("snippet", ""))

            ET.SubElement(item, "title").text = title
            ET.SubElement(item, "link").text = url
            ET.SubElement(item, "description").text = content[:500]

            # GUID
            guid = ET.SubElement(item, "guid")
            guid.text = url
            guid.set("isPermaLink", "true")

            # Pub date (use current time for results)
            ET.SubElement(item, "pubDate").text = datetime.now(timezone.utc).strftime(
                "%a, %d %b %Y %H:%M:%S %z"
            )

            # Author (from domain if available)
            if url:
                from urllib.parse import urlparse

                parsed = urlparse(url)
                if parsed.netloc:
                    ET.SubElement(item, "author").text = f"nobody@{parsed.netloc}"

        # Pretty print XML
        return self._prettify_xml(rss)

    def to_atom(
        self,
        results: List[Dict[str, Any]],
        query: str,
        feed_title: Optional[str] = None,
        feed_id: Optional[str] = None,
    ) -> str:
        """
        Export results to Atom 1.0 format.

        Args:
            results: Search results
            query: Original search query
            feed_title: Override feed title
            feed_id: Override feed ID

        Returns:
            Atom XML string
        """
        # Create Atom feed
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        feed = ET.Element("{http://www.w3.org/2005/Atom}feed")

        # Feed metadata
        ET.SubElement(feed, "{http://www.w3.org/2005/Atom}title").text = (
            feed_title or f"Search: {query}"
        )
        ET.SubElement(feed, "{http://www.w3.org/2005/Atom}id").text = feed_id or f"{self.base_url}/search?q={query}"
        ET.SubElement(feed, "{http://www.w3.org/2005/Atom}updated").text = datetime.now(
            timezone.utc
        ).isoformat()

        # Self link
        link = ET.SubElement(feed, "{http://www.w3.org/2005/Atom}link")
        link.set("rel", "self")
        link.set("href", f"{self.base_url}/export/atom?q={query}")
        link.set("type", "application/atom+xml")

        # Alternate link
        alt_link = ET.SubElement(feed, "{http://www.w3.org/2005/Atom}link")
        alt_link.set("rel", "alternate")
        alt_link.set("href", f"{self.base_url}/search?q={query}")

        # Add entries
        for result in results:
            entry = ET.SubElement(feed, "{http://www.w3.org/2005/Atom}entry")

            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", result.get("snippet", ""))

            ET.SubElement(entry, "{http://www.w3.org/2005/Atom}title").text = title
            ET.SubElement(entry, "{http://www.w3.org/2005/Atom}id").text = url
            ET.SubElement(entry, "{http://www.w3.org/2005/Atom}updated").text = datetime.now(
                timezone.utc
            ).isoformat()
            ET.SubElement(entry, "{http://www.w3.org/2005/Atom}published").text = datetime.now(
                timezone.utc
            ).isoformat()

            # Link
            link = ET.SubElement(entry, "{http://www.w3.org/2005/Atom}link")
            link.set("href", url)
            link.set("rel", "alternate")

            # Summary
            summary = ET.SubElement(entry, "{http://www.w3.org/2005/Atom}summary")
            summary.set("type", "text")
            summary.text = content[:500]

            # Content
            content_elem = ET.SubElement(entry, "{http://www.w3.org/2005/Atom}content")
            content_elem.set("type", "text")
            content_elem.text = content

            # Author
            if url:
                from urllib.parse import urlparse

                parsed = urlparse(url)
                if parsed.netloc:
                    author = ET.SubElement(entry, "{http://www.w3.org/2005/Atom}author")
                    ET.SubElement(author, "{http://www.w3.org/2005/Atom}name").text = parsed.netloc

        # Pretty print XML
        return self._prettify_xml(feed)

    def _prettify_xml(self, elem: ET.Element) -> str:
        """Pretty print XML element."""
        rough_string = ET.tostring(elem, encoding="unicode")
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ", encoding="unicode")

    def export(
        self,
        results: List[Dict[str, Any]],
        query: str,
        format: str = "json",
        **kwargs,
    ) -> str:
        """
        Export results to specified format.

        Args:
            results: Search results
            query: Original search query
            format: Export format (json, csv, rss, atom)
            **kwargs: Additional format-specific options

        Returns:
            Formatted string
        """
        format = format.lower()

        if format == "json":
            return self.to_json(results, query, **kwargs)
        elif format == "csv":
            return self.to_csv(results, query, **kwargs)
        elif format == "rss":
            return self.to_rss(results, query, **kwargs)
        elif format in ("atom", "xml"):
            return self.to_atom(results, query, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")


def create_exporter(base_url: str = "http://10.0.0.102:8080") -> SearchResultsExporter:
    """
    Create search results exporter.

    Args:
        base_url: Base URL for generating links

    Returns:
        Configured SearchResultsExporter
    """
    return SearchResultsExporter(base_url=base_url)
