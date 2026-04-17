"""
Files API for AI Inference Gateway.

Provides file upload/download functionality using Garage S3 storage.
Compatible with Anthropic/OpenAI Files API format.

File operations:
- Upload files to Garage S3 with metadata
- Download files by ID
- List files
- Delete files
"""

import base64
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, BinaryIO
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Garage S3 configuration (matches services.garage-cluster)
GARAGE_S3_HOST = "127.0.0.1"
GARAGE_S3_PORT = 3900
GARAGE_S3_REGION = "garage"
GARAGE_S3_ENDPOINT = f"http://{GARAGE_S3_HOST}:{GARAGE_S3_PORT}"

# Default bucket for AI gateway files
DEFAULT_BUCKET = "ai-gateway-files"

# MIME type mapping for common extensions
MIME_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".json": "application/json",
    ".jsonl": "application/jsonl",
    ".csv": "text/csv",
    ".html": "text/html",
    ".xml": "application/xml",
    ".yaml": "application/x-yaml",
    ".yml": "application/x-yaml",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/typescript",
    ".txt": "text/plain",
    ".log": "text/plain",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".zip": "application/zip",
    ".tar": "application/x-tar",
    ".gz": "application/gzip",
    ".bz2": "application/x-bzip2",
}


def get_mime_type(filename: str) -> str:
    """Get MIME type based on file extension."""
    ext = Path(filename).suffix.lower()
    return MIME_TYPES.get(ext, "application/octet-stream")


def generate_file_id() -> str:
    """Generate a unique file ID in Anthropic format (file_...)."""
    # Use timestamp + random bytes for uniqueness
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    random_bytes = os.urandom(8)
    random_str = base64.urlsafe_b64encode(random_bytes).decode("ascii")[:8]
    return f"file_{timestamp}_{random_str}"


@dataclass
class FileMetadata:
    """Metadata for uploaded files."""

    id: str
    filename: str
    bytes: int
    created_at: str  # ISO 8601 timestamp
    purpose: Optional[str] = None  # "assistant", "user", etc.
    mime_type: str = "application/octet-stream"


class FileStorageError(Exception):
    """Base exception for file storage operations."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class FileNotFoundError(FileStorageError):
    """Raised when a requested file_id doesn't exist."""

    pass


class FileUploadError(FileStorageError):
    """Raised when file upload fails."""

    pass


class GarageS3Client:
    """
    Client for interacting with Garage S3 API.

    Uses HTTP requests to Garage's S3-compatible API.
    """

    def __init__(
        self,
        endpoint: str = GARAGE_S3_ENDPOINT,
        region: str = GARAGE_S3_REGION,
        access_key: str = "garage",
        secret_key: str = "garage",
    ):
        """
        Initialize Garage S3 client.

        Args:
            endpoint: S3 API endpoint URL
            region: S3 region ( Garage uses "garage")
            access_key: Access key (Garage doesn't enforce this)
            secret_key: Secret key (Garage doesn't enforce this)
        """
        self.endpoint = endpoint
        self.region = region
        self.access_key = access_key
        self.secret_key = secret_key
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=300.0)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def ensure_bucket_exists(self, bucket: str) -> bool:
        """
        Check if bucket exists, create if not.

        Args:
            bucket: Bucket name

        Returns:
            True if bucket exists (or was created successfully)
        """
        # Garage doesn't have HeadBucket, try to list objects instead
        # If bucket doesn't exist, we'll create it on first PutObject
        return True

    async def put_file(
        self,
        bucket: str,
        key: str,
        content: bytes,
        content_type: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Upload a file to Garage S3.

        Args:
            bucket: Bucket name
            key: Object key (file_id)
            content: File content as bytes
            content_type: MIME type
            metadata: Optional metadata to attach

        Returns:
            Dictionary with file_id and other info
        """
        await self.ensure_bucket_exists(bucket)

        client = await self._get_client()

        # Build S3 PUT request
        url = f"{self.endpoint}/{bucket}/{key}"

        headers = {
            "Content-Type": content_type,
        }

        # Add custom metadata as x-amz-meta-* headers
        if metadata:
            for k, v in metadata.items():
                headers[f"x-amz-meta-{k}"] = str(v)

        logger.debug(f"Uploading file to Garage: {url}")

        try:
            response = await client.put(url, content=content, headers=headers)
            response.raise_for_status()

            return {
                "id": key,
                "filename": metadata.get("filename", key) if metadata else key,
                "bytes": len(content),
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
        except httpx.HTTPStatusError as e:
            logger.error(f"Garage S3 upload failed: {e.response.status_code} - {e.response.text}")
            raise FileUploadError(f"Failed to upload file to S3: {e.response.text}", status_code=e.response.status_code)
        except Exception as e:
            logger.error(f"Garage S3 upload error: {e}")
            raise FileUploadError(f"Failed to upload file: {str(e)}")

    async def get_file(self, bucket: str, key: str) -> tuple[bytes, Dict[str, str]]:
        """
        Download a file from Garage S3.

        Args:
            bucket: Bucket name
            key: Object key (file_id)

        Returns:
            Tuple of (content_bytes, metadata_dict)

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        client = await self._get_client()

        url = f"{self.endpoint}/{bucket}/{key}"

        try:
            response = await client.get(url)
            response.raise_for_status()

            content = response.content

            # Extract metadata from headers
            metadata = {}
            for k, v in response.headers.items():
                if k.startswith("x-amz-meta-"):
                    meta_key = k[12:]  # Remove "x-amz-meta-" prefix
                    metadata[meta_key] = v

            return content, metadata

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise FileNotFoundError(f"File not found: {key}", status_code=404)
            logger.error(f"Garage S3 download failed: {e.response.status_code} - {e.response.text}")
            raise FileStorageError(f"Failed to download file: {e.response.text}", status_code=e.response.status_code)
        except Exception as e:
            logger.error(f"Garage S3 download error: {e}")
            raise FileStorageError(f"Failed to download file: {str(e)}")

    async def delete_file(self, bucket: str, key: str) -> bool:
        """
        Delete a file from Garage S3.

        Args:
            bucket: Bucket name
            key: Object key (file_id)

        Returns:
            True if deleted successfully
        """
        client = await self._get_client()

        url = f"{self.endpoint}/{bucket}/{key}"

        try:
            response = await client.delete(url)
            response.raise_for_status()
            logger.debug(f"Deleted file from Garage: {key}")
            return True
        except httpx.HTTPStatusError as e:
            logger.error(f"Garage S3 delete failed: {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 404:
                # File doesn't exist, consider it "deleted"
                return True
            raise FileStorageError(f"Failed to delete file: {e.response.text}", status_code=e.response.status_code)
        except Exception as e:
            logger.error(f"Garage S3 delete error: {e}")
            raise FileStorageError(f"Failed to delete file: {str(e)}")

    async def list_files(self, bucket: str) -> List[Dict[str, str]]:
        """
        List all files in the bucket.

        Args:
            bucket: Bucket name

        Returns:
            List of file metadata dicts
        """
        client = await self._get_client()

        # S3 ListObjectsV2 request
        url = f"{self.endpoint}/{bucket}/?list-type=2"

        try:
            response = await client.get(url)
            response.raise_for_status()

            # Parse XML response
            from xml.etree import ElementTree

            root = ElementTree.fromstring(response.content)

            files = []
            for contents in root.findall(".//{http://s3.amazonaws.com/doc/2006-03-01/}Contents"):
                key = contents.find(".//{http://s3.amazonaws.com/doc/2006-03-01/}Key")
                if key is not None:
                    file_id = key.text

                    # Extract metadata from user metadata
                    metadata = {}
                    for meta in contents.findall(".//{http://s3.amazonaws.com/doc/2006-03-01/}UserMetadata"):
                        for item in meta:
                            metadata[item.tag] = item.text

                    size_elem = contents.find(".//{http://s3.amazonaws.com/doc/2006-03-01/}Size")
                    size = int(size_elem.text) if size_elem is not None else 0

                    files.append({
                        "id": file_id,
                        "filename": metadata.get("filename", file_id),
                        "bytes": size,
                        "created_at": metadata.get("created_at", ""),
                    })

            return files

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Bucket doesn't exist yet, no files
                return []
            logger.error(f"Garage S3 list failed: {e.response.status_code} - {e.response.text}")
            raise FileStorageError(f"Failed to list files: {e.response.text}", status_code=e.response.status_code)
        except Exception as e:
            logger.error(f"Garage S3 list error: {e}")
            raise FileStorageError(f"Failed to list files: {str(e)}")

    async def get_file_metadata(self, bucket: str, key: str) -> Optional[FileMetadata]:
        """
        Get file metadata without downloading content.

        Args:
            bucket: Bucket name
            key: Object key (file_id)

        Returns:
            FileMetadata or None if not found
        """
        client = await self._get_client()

        # S3 HeadObject request
        url = f"{self.endpoint}/{bucket}/{key}"

        try:
            response = await client.head(url)
            response.raise_for_status()

            # Extract headers
            bytes_size = int(response.headers.get("Content-Length", 0))
            content_type = response.headers.get("Content-Type", "application/octet-stream")

            # Extract custom metadata
            metadata = {}
            for k, v in response.headers.items():
                if k.startswith("x-amz-meta-"):
                    metadata[k[12:]] = v

            return FileMetadata(
                id=key,
                filename=metadata.get("filename", key),
                bytes=bytes_size,
                created_at=metadata.get("created_at", ""),
                mime_type=content_type,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error(f"Garage S3 head failed: {e.response.status_code}")
            return None


# Global client instance (lazy initialization)
_garage_client: Optional[GarageS3Client] = None


def get_garage_client() -> GarageS3Client:
    """Get or create the global Garage S3 client."""
    global _garage_client
    if _garage_client is None:
        _garage_client = GarageS3Client()
    return _garage_client
