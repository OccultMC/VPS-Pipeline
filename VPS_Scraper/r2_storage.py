"""
Cloudflare R2 Storage Client

S3-compatible upload/download with retry logic and multipart support.
"""
import os
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict
import collections
try:
    if not hasattr(collections, 'Callable'):
        collections.Callable = collections.abc.Callable
except Exception:
    pass

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class R2Client:
    """Cloudflare R2 storage client using boto3 S3-compatible API."""

    def __init__(
        self,
        account_id: str = None,
        access_key_id: str = None,
        secret_access_key: str = None,
        bucket_name: str = None,
    ):
        self.account_id = account_id or os.environ.get("R2_ACCOUNT_ID", "")
        self.access_key_id = access_key_id or os.environ.get("R2_ACCESS_KEY_ID", "")
        self.secret_access_key = secret_access_key or os.environ.get("R2_SECRET_ACCESS_KEY", "")
        self.bucket_name = bucket_name or os.environ.get("R2_BUCKET_NAME", "")

        if not all([self.account_id, self.access_key_id, self.secret_access_key, self.bucket_name]):
            raise ValueError(
                "Missing R2 credentials. Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, "
                "R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME as env vars or pass them directly."
            )

        endpoint_url = f"https://{self.account_id}.r2.cloudflarestorage.com"

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=Config(
                retries={"max_attempts": 3, "mode": "adaptive"},
                s3={"addressing_style": "path"},
            ),
            region_name="auto",
        )

    # ── Upload ────────────────────────────────────────────────────────────

    def upload_file(
        self,
        local_path: str,
        bucket_key: str,
        max_retries: int = 3,
        multipart_threshold_mb: int = 100,
        progress_callback=None,
    ) -> bool:
        """
        Upload a local file to R2.

        Args:
            local_path: Path to the local file.
            bucket_key: Destination key in the bucket.
            max_retries: Number of retry attempts.
            multipart_threshold_mb: Files larger than this use multipart upload.
            progress_callback: Optional callable(bytes_transferred, total_bytes).

        Returns:
            True on success, False on failure.
        """
        local_path = str(local_path)
        file_size = os.path.getsize(local_path)

        for attempt in range(1, max_retries + 1):
            try:
                extra_args = {}
                callback = None

                if progress_callback:
                    transferred = [0]

                    def _cb(bytes_amount):
                        transferred[0] += bytes_amount
                        progress_callback(transferred[0], file_size)

                    callback = _cb

                config = boto3.s3.transfer.TransferConfig(
                    multipart_threshold=multipart_threshold_mb * 1024 * 1024,
                    multipart_chunksize=100 * 1024 * 1024,  # 100MB chunks
                    max_concurrency=4,
                )

                self.s3.upload_file(
                    local_path,
                    self.bucket_name,
                    bucket_key,
                    ExtraArgs=extra_args,
                    Callback=callback,
                    Config=config,
                )

                # Verify upload
                if self._verify_upload(bucket_key, file_size):
                    logger.info(f"Uploaded {bucket_key} ({file_size:,} bytes)")
                    return True
                else:
                    logger.warning(f"Upload verification failed for {bucket_key}, attempt {attempt}")

            except Exception as e:
                logger.error(f"Upload attempt {attempt}/{max_retries} failed for {bucket_key}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff

        logger.error(f"Upload FAILED after {max_retries} attempts: {bucket_key}")
        return False

    def _verify_upload(self, bucket_key: str, expected_size: int) -> bool:
        """Verify uploaded file exists and has correct size."""
        try:
            response = self.s3.head_object(Bucket=self.bucket_name, Key=bucket_key)
            actual_size = response["ContentLength"]
            if actual_size != expected_size:
                logger.warning(
                    f"Size mismatch for {bucket_key}: expected {expected_size}, got {actual_size}"
                )
                return False
            return True
        except ClientError:
            return False

    def upload_json(self, bucket_key: str, data: dict, max_retries: int = 3) -> bool:
        """Upload a dict as JSON to R2."""
        import json as _json
        body = _json.dumps(data).encode('utf-8')
        for attempt in range(1, max_retries + 1):
            try:
                self.s3.put_object(
                    Bucket=self.bucket_name,
                    Key=bucket_key,
                    Body=body,
                    ContentType='application/json',
                )
                return True
            except Exception as e:
                logger.error(f"JSON upload attempt {attempt}/{max_retries} failed for {bucket_key}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
        return False

    # ── Download ──────────────────────────────────────────────────────────

    def download_file(
        self,
        bucket_key: str,
        local_path: str,
        max_retries: int = 3,
        progress_callback=None,
    ) -> bool:
        """
        Download a file from R2 to local path.

        Args:
            bucket_key: Source key in the bucket.
            local_path: Destination local path.
            max_retries: Number of retry attempts.
            progress_callback: Optional callable(bytes_transferred, total_bytes).

        Returns:
            True on success, False on failure.
        """
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

        for attempt in range(1, max_retries + 1):
            try:
                # Get file size first
                head = self.s3.head_object(Bucket=self.bucket_name, Key=bucket_key)
                total_size = head["ContentLength"]

                callback = None
                if progress_callback:
                    transferred = [0]

                    def _cb(bytes_amount):
                        transferred[0] += bytes_amount
                        progress_callback(transferred[0], total_size)

                    callback = _cb

                self.s3.download_file(
                    self.bucket_name, bucket_key, local_path, Callback=callback
                )

                # Verify download
                actual_size = os.path.getsize(local_path)
                if actual_size == total_size:
                    logger.info(f"Downloaded {bucket_key} ({total_size:,} bytes)")
                    return True
                else:
                    logger.warning(
                        f"Download size mismatch: expected {total_size}, got {actual_size}"
                    )

            except Exception as e:
                logger.error(f"Download attempt {attempt}/{max_retries} failed for {bucket_key}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        logger.error(f"Download FAILED after {max_retries} attempts: {bucket_key}")
        return False

    # ── Utilities ─────────────────────────────────────────────────────────

    def file_exists(self, bucket_key: str) -> bool:
        """Check if an object exists in the bucket."""
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=bucket_key)
            return True
        except ClientError:
            return False

    def list_files(self, prefix: str = "") -> List[Dict]:
        """List objects under a prefix."""
        results = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                results.append({
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"],
                })
        return results

    def download_json(self, bucket_key: str) -> dict:
        """Download and parse a JSON file from R2. Returns empty dict on failure."""
        import json as _json
        try:
            resp = self.s3.get_object(Bucket=self.bucket_name, Key=bucket_key)
            body = resp['Body'].read()
            return _json.loads(body)
        except Exception:
            return {}

    def delete_file(self, bucket_key: str) -> bool:
        """Delete a file from the bucket."""
        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=bucket_key)
            return True
        except ClientError as e:
            logger.error(f"Delete failed for {bucket_key}: {e}")
            return False
