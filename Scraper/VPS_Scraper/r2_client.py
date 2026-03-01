"""Cloudflare R2 Storage Client (S3-compatible via boto3)."""

import io
import json
import os
import time
import logging
from typing import Optional, List, Dict, Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class R2Client:
    """Cloudflare R2 storage client using boto3 S3-compatible API."""

    def __init__(
        self,
        endpoint_url: str = None,
        access_key_id: str = None,
        secret_access_key: str = None,
        bucket_name: str = None,
    ):
        self.endpoint_url = endpoint_url or os.environ.get("R2_ENDPOINT_URL", "")
        self.access_key_id = access_key_id or os.environ.get("R2_ACCESS_KEY_ID", "")
        self.secret_access_key = secret_access_key or os.environ.get("R2_SECRET_ACCESS_KEY", "")
        self.bucket_name = bucket_name or os.environ.get("R2_BUCKET_NAME", "streetview-data")

        if not all([self.endpoint_url, self.access_key_id, self.secret_access_key, self.bucket_name]):
            raise ValueError("Missing R2 credentials. Check .env file.")

        self.s3 = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=Config(
                retries={"max_attempts": 3, "mode": "adaptive"},
                s3={"addressing_style": "path"},
            ),
            region_name="auto",
        )

    def upload_file(self, local_path: str, bucket_key: str, max_retries: int = 3) -> bool:
        """Upload a local file to R2."""
        local_path = str(local_path)
        file_size = os.path.getsize(local_path)

        for attempt in range(1, max_retries + 1):
            try:
                config = boto3.s3.transfer.TransferConfig(
                    multipart_threshold=100 * 1024 * 1024,
                    multipart_chunksize=100 * 1024 * 1024,
                    max_concurrency=4,
                )
                self.s3.upload_file(local_path, self.bucket_name, bucket_key, Config=config)

                resp = self.s3.head_object(Bucket=self.bucket_name, Key=bucket_key)
                if resp["ContentLength"] == file_size:
                    logger.info(f"Uploaded {bucket_key} ({file_size:,} bytes)")
                    return True
                else:
                    logger.warning(f"Size mismatch for {bucket_key}")
            except Exception as e:
                logger.warning(f"Upload attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        return False

    def upload_bytes(self, bucket_key: str, data: bytes, content_type: str = "application/octet-stream") -> bool:
        """Upload raw bytes to R2."""
        try:
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=bucket_key,
                Body=data,
                ContentType=content_type,
            )
            logger.info(f"Uploaded {bucket_key} ({len(data):,} bytes)")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {bucket_key}: {e}")
            return False

    def upload_json(self, bucket_key: str, data: dict, max_retries: int = 3) -> bool:
        """Upload a dict as JSON to R2."""
        body = json.dumps(data).encode("utf-8")
        for attempt in range(1, max_retries + 1):
            try:
                self.s3.put_object(
                    Bucket=self.bucket_name,
                    Key=bucket_key,
                    Body=body,
                    ContentType="application/json",
                )
                return True
            except Exception as e:
                logger.warning(f"JSON upload attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
        return False

    def download_file(self, bucket_key: str, local_path: str, max_retries: int = 3) -> bool:
        """Download a file from R2 to local disk."""
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

        for attempt in range(1, max_retries + 1):
            try:
                head = self.s3.head_object(Bucket=self.bucket_name, Key=bucket_key)
                total_size = head["ContentLength"]
                self.s3.download_file(self.bucket_name, bucket_key, local_path)

                if os.path.getsize(local_path) == total_size:
                    logger.info(f"Downloaded {bucket_key} ({total_size:,} bytes)")
                    return True
            except Exception as e:
                logger.warning(f"Download attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        return False

    def download_bytes(self, bucket_key: str) -> Optional[bytes]:
        """Download raw bytes from R2."""
        try:
            resp = self.s3.get_object(Bucket=self.bucket_name, Key=bucket_key)
            return resp["Body"].read()
        except Exception as e:
            logger.error(f"Failed to download {bucket_key}: {e}")
            return None

    def download_json(self, bucket_key: str) -> dict:
        """Download and parse a JSON file from R2. Returns empty dict on failure."""
        try:
            resp = self.s3.get_object(Bucket=self.bucket_name, Key=bucket_key)
            body = resp["Body"].read()
            return json.loads(body)
        except Exception:
            return {}

    def file_exists(self, bucket_key: str) -> bool:
        """Check if an object exists in the bucket."""
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=bucket_key)
            return True
        except Exception:
            return False

    def list_objects(self, prefix: str) -> List[str]:
        """List all object keys under a prefix."""
        keys = []
        try:
            paginator = self.s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
        except Exception as e:
            logger.error(f"Failed to list objects under {prefix}: {e}")
        return keys

    def delete_object(self, bucket_key: str) -> bool:
        """Delete an object from R2."""
        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=bucket_key)
            logger.info(f"Deleted {bucket_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {bucket_key}: {e}")
            return False
