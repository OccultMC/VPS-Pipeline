"""
Cloudflare R2 Storage Client (Pipeline Worker)

Optimized for large feature file uploads with multipart and verification.
"""
import os
import time
import logging
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
            raise ValueError("Missing R2 credentials")

        self._endpoint_url = f"https://{self.account_id}.r2.cloudflarestorage.com"
        self.s3 = self._make_client()

    def _make_client(self):
        """Create a fresh boto3 S3 client (avoids stale connection pool issues)."""
        return boto3.client(
            "s3",
            endpoint_url=self._endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=Config(
                retries={"max_attempts": 8, "mode": "adaptive"},
                s3={"addressing_style": "path"},
                read_timeout=120,     # 2 min per request
                connect_timeout=30,
            ),
            region_name="auto",
        )

    def reset_client(self):
        """Reset the S3 client to clear stale connections."""
        self.s3 = self._make_client()

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

                # Single PUT for anything under 512MB: R2's single-PUT limit
                # is ~5GB and our largest NPY is ~258MB, so the whole file
                # goes in ONE request that either lands or fails atomically.
                # Multipart to R2 from rented machines was failing every
                # attempt ("connection closed before valid response") and
                # each retry created yet another orphaned multipart upload.
                # Files above the threshold (none today) use 64MB parts.
                config = boto3.s3.transfer.TransferConfig(
                    multipart_threshold=512 * 1024 * 1024,
                    multipart_chunksize=64 * 1024 * 1024,
                    max_concurrency=4,
                )
                self.s3.upload_file(local_path, self.bucket_name, bucket_key,
                                    Callback=callback, Config=config)

                # Verify
                resp = self.s3.head_object(Bucket=self.bucket_name, Key=bucket_key)
                if resp["ContentLength"] == file_size:
                    print(f"[R2] Uploaded {bucket_key} ({file_size:,} bytes)")
                    return True
                else:
                    print(f"[R2] Size mismatch for {bucket_key}")
            except Exception as e:
                print(f"[R2] Upload attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    # Reset client to clear stale connections and abort any
                    # orphaned multipart upload before retry
                    self.reset_client()
                    self.abort_pending_multipart(bucket_key)
                    time.sleep(2 ** attempt)

        self.abort_pending_multipart(bucket_key)
        return False

    def abort_pending_multipart(self, bucket_key: str) -> int:
        """Abort any in-progress multipart uploads for this exact key.

        Failed multipart attempts leave orphaned uploads server-side; each
        retry that re-creates the upload from scratch adds another orphan
        whose parts count against storage forever. Call before re-attempting
        a key and on final failure. Returns the number aborted.
        """
        aborted = 0
        try:
            resp = self.s3.list_multipart_uploads(
                Bucket=self.bucket_name, Prefix=bucket_key
            )
            for up in resp.get("Uploads", []):
                if up.get("Key") != bucket_key:
                    continue
                upload_id = up.get("UploadId", "")
                try:
                    self.s3.abort_multipart_upload(
                        Bucket=self.bucket_name, Key=bucket_key,
                        UploadId=upload_id,
                    )
                    aborted += 1
                except Exception as e:
                    print(f"[R2] Abort multipart {upload_id[:12]} for "
                          f"{bucket_key} failed: {e}")
            if aborted:
                print(f"[R2] Aborted {aborted} pending multipart upload(s) "
                      f"for {bucket_key}")
        except Exception as e:
            print(f"[R2] list_multipart_uploads failed for {bucket_key}: {e}")
        return aborted

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
                print(f"[R2] JSON upload attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
        return False

    def list_objects(self, prefix: str, suffix: str = None) -> List[str]:
        """List object keys under a prefix, optionally filtered by suffix."""
        keys = []
        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if suffix is None or key.endswith(suffix):
                    keys.append(key)
        return keys

    def file_exists(self, bucket_key: str, expected_size: int = None) -> bool:
        """Check if an object exists in the bucket.

        If expected_size is given, the object's ContentLength must match —
        a half-landed or stale object must not pass as "already uploaded".
        """
        try:
            resp = self.s3.head_object(Bucket=self.bucket_name, Key=bucket_key)
            if expected_size is not None and resp.get("ContentLength") != expected_size:
                return False
            return True
        except Exception:
            return False

    def object_size(self, bucket_key: str) -> int:
        """Return the object's size in bytes, or -1 if missing/unreachable."""
        try:
            resp = self.s3.head_object(Bucket=self.bucket_name, Key=bucket_key)
            return int(resp.get("ContentLength", 0))
        except Exception:
            return -1

    def object_missing(self, bucket_key: str) -> Optional[bool]:
        """Classify a key: True = definitely missing (404/NoSuchKey),
        False = exists, None = transient error (unknown).

        Used to distinguish "city CSVs really aren't on R2" (blacklist)
        from "R2 is having a moment" (retry later, no blacklist)."""
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=bucket_key)
            return False
        except ClientError as e:
            code = str((e.response.get("Error") or {}).get("Code", ""))
            status = (e.response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
            if code in ("404", "NoSuchKey", "NotFound") or status == 404:
                return True
            return None
        except Exception:
            return None

    def delete_object(self, bucket_key: str) -> bool:
        """Delete an object from R2."""
        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=bucket_key)
            print(f"[R2] Deleted {bucket_key}")
            return True
        except Exception as e:
            print(f"[R2] Failed to delete {bucket_key}: {e}")
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
                    print(f"[R2] Downloaded {bucket_key} ({total_size:,} bytes)")
                    return True
            except Exception as e:
                print(f"[R2] Download attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        return False
