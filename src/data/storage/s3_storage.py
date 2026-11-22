"""S3-compatible Object Storage Client for MinIO Medallion Architecture.

Implements an Adapter pattern around boto3 S3 client to handle bucket management,
object upload/download with metadata provenance, and existence checks.
"""

import hashlib
import logging
import os
from typing import Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3Storage:
    """Gateway adapter for local MinIO S3-compatible storage."""

    DEFAULT_BUCKETS: tuple[str, ...] = ("sec-bronze", "sec-silver", "sec-gold")

    def __init__(
        self,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region_name: str = "us-east-1",
    ) -> None:
        """Initialize S3 client with environment fallback."""
        app_env = os.getenv("APP_ENV", "").lower()
        if app_env in ("docker", "prod", "production"):
            default_endpoint = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
        else:
            default_endpoint = (
                os.getenv("S3_ENDPOINT_URL_LOCAL") or os.getenv("S3_ENDPOINT_URL") or "http://localhost:9000"
            )

        self.endpoint_url = endpoint_url or default_endpoint
        self.access_key = access_key or os.getenv("S3_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER")
        self.secret_key = secret_key or os.getenv("S3_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD")
        self.region_name = region_name

        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region_name,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
        logger.info("Initialized S3Storage connected to %s", self.endpoint_url)

    def ensure_buckets_exist(self, buckets: list[str] | None = None) -> None:
        """Ensure all required medallion buckets exist, creating them if necessary."""
        target_buckets = buckets or self.DEFAULT_BUCKETS
        for bucket in target_buckets:
            try:
                self.client.head_bucket(Bucket=bucket)
                logger.debug("Bucket '%s' already exists.", bucket)
            except ClientError as exc:
                error_code = exc.response.get("Error", {}).get("Code")
                if error_code in ("404", "NoSuchBucket"):
                    logger.info("Creating bucket: %s", bucket)
                    self.client.create_bucket(Bucket=bucket)
                else:
                    logger.error("Failed to inspect bucket '%s': %s", bucket, exc)
                    raise

    def upload_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> bool:
        """Upload raw bytes to a target bucket and key with optional custom metadata."""
        try:
            extra_args: dict[str, Any] = {"ContentType": content_type}
            if metadata:
                # S3 metadata values must be strings
                extra_args["Metadata"] = {k: str(v) for k, v in metadata.items()}

            self.client.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                **extra_args,
            )
            logger.info("Uploaded %d bytes to s3://%s/%s", len(data), bucket, key)
            return True
        except ClientError as exc:
            logger.error("Failed to upload object to s3://%s/%s: %s", bucket, key, exc)
            raise

    def download_bytes(self, bucket: str, key: str) -> bytes:
        """Download raw bytes from a specified bucket and key."""
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
            data = response["Body"].read()
            logger.debug("Downloaded %d bytes from s3://%s/%s", len(data), bucket, key)
            return data
        except ClientError as exc:
            logger.error("Failed to download object from s3://%s/%s: %s", bucket, key, exc)
            raise

    def object_exists(self, bucket: str, key: str) -> bool:
        """Check whether an object exists in the given bucket."""
        try:
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code in ("404", "NoSuchKey"):
                return False
            logger.warning("Error checking existence for s3://%s/%s: %s", bucket, key, exc)
            raise

    def delete_object(self, bucket: str, key: str) -> bool:
        """Delete an object from a specified bucket and key.

        Args:
            bucket: Target bucket name.
            key: Target object key.

        Returns:
            True if deletion was issued without error.
        """
        try:
            self.client.delete_object(Bucket=bucket, Key=key)
            logger.info("Deleted object s3://%s/%s", bucket, key)
            return True
        except ClientError as exc:
            logger.error("Failed to delete object from s3://%s/%s: %s", bucket, key, exc)
            raise

    def get_metadata(self, bucket: str, key: str) -> dict[str, Any]:
        """Retrieve system and user metadata for an object."""
        try:
            response = self.client.head_object(Bucket=bucket, Key=key)
            return {
                "content_length": response.get("ContentLength"),
                "content_type": response.get("ContentType"),
                "etag": response.get("ETag", "").strip('"'),
                "last_modified": response.get("LastModified"),
                "metadata": response.get("Metadata", {}),
            }
        except ClientError as exc:
            logger.error("Failed to fetch metadata for s3://%s/%s: %s", bucket, key, exc)
            raise

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        """List all object keys in a bucket matching an optional prefix."""
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            keys: list[str] = []
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for item in page.get("Contents", []):
                    keys.append(item["Key"])
            return keys
        except ClientError as exc:
            logger.error("Failed to list objects in bucket '%s': %s", bucket, exc)
            raise

    def verify_integrity(self, bucket: str, key: str) -> bool:
        """Verify whether a stored object's content matches its recorded SHA-256 metadata.

        Args:
            bucket: The S3 bucket name.
            key: The S3 object key.

        Returns:
            True if the object exists and its computed SHA-256 matches the metadata hash.
            False if metadata is missing, hash mismatches, or the object cannot be read.
        """
        try:
            metadata_info = self.get_metadata(bucket, key)
            user_metadata = metadata_info.get("metadata", {})
            expected_hash = user_metadata.get("sha256")

            if not expected_hash:
                logger.warning("No sha256 metadata recorded for s3://%s/%s", bucket, key)
                return False

            data = self.download_bytes(bucket, key)
            actual_hash = hashlib.sha256(data).hexdigest()

            if actual_hash.lower() != expected_hash.lower():
                logger.error(
                    "Integrity check failed for s3://%s/%s: expected %s, got %s",
                    bucket,
                    key,
                    expected_hash,
                    actual_hash,
                )
                return False

            logger.debug("Integrity verified for s3://%s/%s (SHA256: %s)", bucket, key, actual_hash[:8])
            return True
        except ClientError as exc:
            logger.warning("Failed to verify integrity for s3://%s/%s: %s", bucket, key, exc)
            return False
