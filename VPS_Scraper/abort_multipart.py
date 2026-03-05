"""
Abort all incomplete multipart uploads in the R2 bucket.
Run from VPS_Scraper dir: python abort_multipart.py
Reads credentials from .env file (same dir) then falls back to environment variables.
"""
import os
import sys
import collections
import collections.abc
from pathlib import Path

# Python 3.10+ removed collections.Callable — patch before boto3 imports dateutil
if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable

import boto3
from botocore.config import Config

# Load .env from the same directory as this script
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

account_id      = os.environ.get("R2_ACCOUNT_ID", "")
access_key_id   = os.environ.get("R2_ACCESS_KEY_ID", "")
secret_key      = os.environ.get("R2_SECRET_ACCESS_KEY", "")
bucket_name     = os.environ.get("R2_BUCKET_NAME", "")

if not all([account_id, access_key_id, secret_key, bucket_name]):
    print("ERROR: Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME env vars.")
    sys.exit(1)

s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_key,
    region_name="auto",
    config=Config(signature_version="s3v4"),
)

print(f"Listing incomplete multipart uploads in bucket: {bucket_name}")

uploads = []
kwargs = {"Bucket": bucket_name}
while True:
    resp = s3.list_multipart_uploads(**kwargs)
    for u in resp.get("Uploads", []):
        uploads.append((u["Key"], u["UploadId"]))
    if resp.get("IsTruncated"):
        kwargs["KeyMarker"] = resp["NextKeyMarker"]
        kwargs["UploadIdMarker"] = resp["NextUploadIdMarker"]
    else:
        break

if not uploads:
    print("No incomplete multipart uploads found.")
    sys.exit(0)

print(f"\nFound {len(uploads)} incomplete multipart upload(s):")
for key, uid in uploads:
    print(f"  {key}  (UploadId: {uid[:16]}...)")

print()
confirm = input("Abort all of them? [y/N] ").strip().lower()
if confirm != "y":
    print("Aborted — nothing changed.")
    sys.exit(0)

ok = 0
fail = 0
for key, uid in uploads:
    try:
        s3.abort_multipart_upload(Bucket=bucket_name, Key=key, UploadId=uid)
        print(f"  ABORTED: {key}")
        ok += 1
    except Exception as e:
        print(f"  FAILED:  {key} — {e}")
        fail += 1

print(f"\nDone: {ok} aborted, {fail} failed.")
