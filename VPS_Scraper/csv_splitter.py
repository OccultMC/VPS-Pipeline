"""
CSV Splitter & R2 Uploader

Splits a CSV file into fixed-size chunks or N worker segments
and uploads them to Cloudflare R2.

Chunk mode:  {CityName}_chunk_0001.csv  (1K panos each)
Legacy mode: {CityName}_{worker_idx}.{num_workers}.csv
"""
import csv
import os
import math
import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def split_csv(
    csv_path: str,
    num_workers: int,
    city_name: str,
    output_dir: str = None,
) -> List[str]:
    """
    Split a CSV file into ``num_workers`` roughly-equal segments.

    Each segment keeps the original header and is named
    ``{city_name}_{i}.{num_workers}.csv`` (1-indexed).

    Args:
        csv_path: Path to the source CSV.
        num_workers: Number of segments to create.
        city_name: City name used in output filenames.
        output_dir: Directory for output segments (defaults to csv_path's dir).

    Returns:
        List of absolute paths to the created segment files.
    """
    csv_path = str(csv_path)
    if output_dir is None:
        output_dir = os.path.dirname(csv_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Read all rows
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    total_rows = len(rows)
    if total_rows == 0:
        logger.warning(f"CSV is empty: {csv_path}")
        return []

    # Cap workers to available rows so filenames stay consistent with NUM_WORKERS
    effective_workers = min(num_workers, total_rows)
    if effective_workers < num_workers:
        logger.warning(
            f"Requested {num_workers} workers but only {total_rows} rows available; "
            f"capping to {effective_workers} workers"
        )

    rows_per_worker = math.ceil(total_rows / effective_workers)
    segment_paths: List[str] = []

    for i in range(effective_workers):
        start = i * rows_per_worker
        end = min(start + rows_per_worker, total_rows)
        segment_rows = rows[start:end]

        filename = f"{city_name}_{i + 1}.{effective_workers}.csv"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(segment_rows)

        segment_paths.append(filepath)
        logger.info(
            f"Segment {i + 1}/{effective_workers}: {len(segment_rows)} rows -> {filename}"
        )

    logger.info(
        f"Split {total_rows} rows into {len(segment_paths)} segments "
        f"(~{rows_per_worker} rows each)"
    )
    return segment_paths


def upload_csv_segments(
    segment_paths: List[str],
    r2_client,
    country: str,
    state: str,
    city: str,
) -> List[str]:
    """
    Upload CSV segments to R2 under ``CSV/{country}/{state}/{city}/``.

    Args:
        segment_paths: List of local CSV file paths.
        r2_client: An ``R2Client`` instance.
        country: Country name/code for the bucket path.
        state: State name for the bucket path.
        city: City name for the bucket path.

    Returns:
        List of R2 bucket keys that were successfully uploaded.
    """
    prefix = f"CSV/{country}/{state}/{city}"
    uploaded_keys: List[str] = []

    for path in segment_paths:
        filename = os.path.basename(path)
        bucket_key = f"{prefix}/{filename}"

        success = r2_client.upload_file(path, bucket_key)
        if success:
            uploaded_keys.append(bucket_key)
            logger.info(f"Uploaded {filename} -> {bucket_key}")
        else:
            logger.error(f"FAILED to upload {filename}")

    logger.info(f"Uploaded {len(uploaded_keys)}/{len(segment_paths)} CSV segments to R2")
    return uploaded_keys


def split_csv_chunks(
    csv_path: str,
    chunk_size: int = 1000,
    city_name: str = "Unknown",
    output_dir: str = None,
) -> List[str]:
    """
    Split a CSV into fixed-size chunks of ``chunk_size`` panos each.

    Each chunk keeps the original header and is named
    ``{city_name}_chunk_{NNNN}.csv`` (1-indexed, zero-padded to 4 digits).
    Supports up to 9999 chunks (~10M panos).

    Args:
        csv_path: Path to the source CSV.
        chunk_size: Number of rows per chunk (default 1000).
        city_name: City name used in output filenames.
        output_dir: Directory for output chunks (defaults to csv_path's dir / chunks).

    Returns:
        List of absolute paths to the created chunk files.
    """
    csv_path = str(csv_path)
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(csv_path) or ".", "chunks")
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    total_rows = len(rows)
    if total_rows == 0:
        logger.warning(f"CSV is empty: {csv_path}")
        return []

    total_chunks = math.ceil(total_rows / chunk_size)
    chunk_paths: List[str] = []

    for i in range(total_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_rows)
        chunk_rows = rows[start:end]

        chunk_id = f"chunk_{i + 1:04d}"
        filename = f"{city_name}_{chunk_id}.csv"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(chunk_rows)

        chunk_paths.append(filepath)

    logger.info(
        f"Split {total_rows} rows into {total_chunks} chunks "
        f"of {chunk_size} rows each"
    )
    return chunk_paths
