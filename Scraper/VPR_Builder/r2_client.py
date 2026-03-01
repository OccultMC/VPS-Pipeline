"""Cloudflare R2 Storage Client for Builder container."""

import json
import os
import time
import logging
from typing import List, Optional

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)


class R2Client:
    """R2 client configured from environment variables."""

    def __init__(self):
        self.endpoint_url = os.environ["R2_ENDPOINT_URL"]
        self.bucket_name = os.environ.get("R2_BUCKET_NAME", "streetview-data")

        self.s3 = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            config=Config(
                retries={"max_attempts": 3, "mode": "adaptive"},
                s3={"addressing_style": "path"},
            ),
            region_name="auto",
        )

    def download_bytes(self, key: str) -> Optional[bytes]:
        try:
            resp = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            return resp["Body"].read()
        except Exception as e:
            logger.error(f"Failed to download {key}: {e}")
            return None

    def download_file(self, key: str, local_path: str) -> bool:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        for attempt in range(1, 4):
            try:
                self.s3.download_file(self.bucket_name, key, local_path)
                return True
            except Exception as e:
                logger.warning(f"Download attempt {attempt}/3 for {key}: {e}")
                if attempt < 3:
                    time.sleep(2 ** attempt)
        return False

    def upload_file(self, local_path: str, key: str, max_retries: int = 3) -> bool:
        file_size = os.path.getsize(local_path)
        for attempt in range(1, max_retries + 1):
            try:
                config = boto3.s3.transfer.TransferConfig(
                    multipart_threshold=100 * 1024 * 1024,
                    multipart_chunksize=100 * 1024 * 1024,
                    max_concurrency=4,
                )
                self.s3.upload_file(local_path, self.bucket_name, key, Config=config)
                resp = self.s3.head_object(Bucket=self.bucket_name, Key=key)
                if resp["ContentLength"] == file_size:
                    logger.info(f"Uploaded {key} ({file_size:,} bytes)")
                    return True
            except Exception as e:
                logger.warning(f"Upload attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
        return False

    def upload_json(self, key: str, data: dict) -> bool:
        body = json.dumps(data).encode("utf-8")
        try:
            self.s3.put_object(Bucket=self.bucket_name, Key=key, Body=body, ContentType="application/json")
            return True
        except Exception as e:
            logger.error(f"Failed to upload JSON {key}: {e}")
            return False

    def list_objects(self, prefix: str) -> List[str]:
        keys = []
        try:
            paginator = self.s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
        except Exception as e:
            logger.error(f"Failed to list objects under {prefix}: {e}")
        return keys
