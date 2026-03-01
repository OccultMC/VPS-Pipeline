"""
Cloudflare R2 Storage Client (Pipeline Worker)

Optimized for large feature file uploads with multipart and verification.
"""
import os
import time
import logging
from datetime import datetime, timezone, timedelta
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

_BRISBANE_TZ = timezone(timedelta(hours=10))


def _ts() -> str:
    return datetime.now(_BRISBANE_TZ).strftime("%Y-%m-%d %H:%M:%S AEST")


def _log(tag: str, msg: str):
    print(f"[{_ts()}] [R2:{tag}] {msg}", flush=True)


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
            raise ValueError("Missing R2 credentials")

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

    def upload_file(self, local_path: str, bucket_key: str, max_retries: int = 3,
                    progress_callback=None) -> bool:
        local_path = str(local_path)
        file_size = os.path.getsize(local_path)

        for attempt in range(1, max_retries + 1):
            try:
                callback = None
                if progress_callback:
                    transferred = [0]
                    def _cb(n):
                        transferred[0] += n
                        progress_callback(transferred[0], file_size)
                    callback = _cb

                config = boto3.s3.transfer.TransferConfig(
                    multipart_threshold=100 * 1024 * 1024,
                    multipart_chunksize=100 * 1024 * 1024,
                    max_concurrency=4,
                )
                self.s3.upload_file(local_path, self.bucket_name, bucket_key,
                                    Callback=callback, Config=config)

                # Verify
                resp = self.s3.head_object(Bucket=self.bucket_name, Key=bucket_key)
                if resp["ContentLength"] == file_size:
                    _log("OK", f"Uploaded {bucket_key} ({file_size:,} bytes)")
                    return True
                else:
                    _log("WARN", f"Size mismatch for {bucket_key}")
            except Exception as e:
                err_str = str(e)
                # Handle SSL validation failures — often transient on long uploads
                if "SSL validation failed" in err_str:
                    _log("WARN", f"SSL error on attempt {attempt}/{max_retries} for {bucket_key} — will retry")
                # Handle rate limiting (ServiceUnavailable / concurrent request rate)
                elif "ServiceUnavailable" in err_str or "Reduce your concurrent request rate" in err_str:
                    wait = min(30 * attempt, 120)
                    _log("WARN", f"Rate limited on attempt {attempt}/{max_retries} for {bucket_key} — waiting {wait}s")
                    time.sleep(wait)
                    continue
                else:
                    _log("WARN", f"Upload attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        return False

    def upload_json(self, bucket_key: str, data: dict, max_retries: int = 3) -> bool:
        """Upload a dict as JSON to R2 without needing a temp file."""
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
                err_str = str(e)
                if "ServiceUnavailable" in err_str or "Reduce your concurrent request rate" in err_str:
                    wait = min(10 * attempt, 60)
                    _log("WARN", f"JSON rate limited for {bucket_key} — waiting {wait}s (attempt {attempt}/{max_retries})")
                    time.sleep(wait)
                else:
                    _log("WARN", f"JSON upload attempt {attempt}/{max_retries} for {bucket_key}: {e}")
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
        return False

    def file_exists(self, bucket_key: str) -> bool:
        """Check if an object exists in the bucket."""
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=bucket_key)
            return True
        except Exception:
            return False

    def delete_object(self, bucket_key: str) -> bool:
        """Delete an object from R2."""
        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=bucket_key)
            _log("OK", f"Deleted {bucket_key}")
            return True
        except Exception as e:
            _log("WARN", f"Failed to delete {bucket_key}: {e}")
            return False

    def download_json(self, bucket_key: str) -> dict:
        """Download and parse a JSON file from R2. Returns empty dict on failure."""
        import json as _json
        try:
            resp = self.s3.get_object(Bucket=self.bucket_name, Key=bucket_key)
            body = resp['Body'].read()
            return _json.loads(body)
        except Exception:
            return {}

    def download_file(self, bucket_key: str, local_path: str, max_retries: int = 3,
                      progress_callback=None) -> bool:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

        for attempt in range(1, max_retries + 1):
            try:
                head = self.s3.head_object(Bucket=self.bucket_name, Key=bucket_key)
                total_size = head["ContentLength"]

                callback = None
                if progress_callback:
                    transferred = [0]
                    def _cb(n):
                        transferred[0] += n
                        progress_callback(transferred[0], total_size)
                    callback = _cb

                self.s3.download_file(self.bucket_name, bucket_key, local_path, Callback=callback)

                if os.path.getsize(local_path) == total_size:
                    _log("OK", f"Downloaded {bucket_key} ({total_size:,} bytes)")
                    return True
            except Exception as e:
                _log("WARN", f"Download attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        return False
