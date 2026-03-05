"""
Build shapes_tracker.json from R2 bucket contents.

Scans CSV/, Features/, Index/ prefixes and builds a consolidated
status tracker JSON. Uploads it to status/shapes_tracker.json on R2.

Usage:
    python build_tracker.py
"""
import os
import json
import logging
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from r2_storage import R2Client

logger = logging.getLogger(__name__)

TRACKER_KEY = "status/shapes_tracker.json"


def _extract_region(key, prefix):
    """Extract country/state/city from an R2 key like 'CSV/AU/Queensland/Bundaberg/file.csv'."""
    parts = key.split('/')
    # prefix/ country / state / city / filename...
    if len(parts) >= 4:
        return f"{parts[1]}/{parts[2]}/{parts[3]}"
    return None


def _read_first_row_coords(r2, csv_key):
    """Read first data row of a CSV on R2, return (lat, lon) or (None, None)."""
    try:
        resp = r2.s3.get_object(
            Bucket=r2.bucket_name, Key=csv_key, Range='bytes=0-1024'
        )
        chunk = resp['Body'].read().decode('utf-8', errors='ignore')
        lines = chunk.splitlines()
        if len(lines) < 2:
            return None, None
        parts = lines[1].split(',')
        if len(parts) < 3:
            return None, None
        return float(parts[1]), float(parts[2])
    except Exception:
        return None, None


def build_tracker(r2):
    """
    Scan R2 for CSV/, Features/, Index/ prefixes and build tracker dict.

    Returns:
        dict mapping "country/state/city" -> {lat, lon, csv, features, index}
    """
    tracker = {}

    # --- Scan CSV/ ---
    logger.info("Scanning CSV/ ...")
    csv_files = r2.list_files(prefix="CSV/")
    csv_regions = defaultdict(list)
    for f in csv_files:
        key = f['key']
        if not key.lower().endswith('.csv'):
            continue
        region = _extract_region(key, "CSV")
        if region:
            csv_regions[region].append(key)

    logger.info(f"  Found {len(csv_regions)} CSV regions")
    for region, files in csv_regions.items():
        lat, lon = _read_first_row_coords(r2, files[0])
        tracker[region] = {
            'lat': lat or 0,
            'lon': lon or 0,
            'csv': True,
            'features': False,
            'index': False,
        }

    # --- Scan Features/ ---
    logger.info("Scanning Features/ ...")
    features_files = r2.list_files(prefix="Features/")
    features_regions = set()
    for f in features_files:
        region = _extract_region(f['key'], "Features")
        if region:
            features_regions.add(region)

    logger.info(f"  Found {len(features_regions)} Features regions")
    for region in features_regions:
        if region in tracker:
            tracker[region]['features'] = True
        else:
            tracker[region] = {
                'lat': 0, 'lon': 0,
                'csv': False, 'features': True, 'index': False,
            }

    # --- Scan Index/ ---
    logger.info("Scanning Index/ ...")
    index_files = r2.list_files(prefix="Index/")
    index_regions = set()
    for f in index_files:
        region = _extract_region(f['key'], "Index")
        if region:
            index_regions.add(region)

    logger.info(f"  Found {len(index_regions)} Index regions")
    for region in index_regions:
        if region in tracker:
            tracker[region]['index'] = True
        else:
            tracker[region] = {
                'lat': 0, 'lon': 0,
                'csv': False, 'features': False, 'index': True,
            }

    return tracker


def upload_tracker(r2, tracker):
    """Upload tracker dict to R2."""
    return r2.upload_json(TRACKER_KEY, tracker)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    r2 = R2Client()
    logger.info("Scanning R2 bucket...")
    tracker = build_tracker(r2)
    logger.info(f"Found {len(tracker)} regions total")

    for key, val in sorted(tracker.items()):
        flags = []
        if val['csv']:
            flags.append('CSV')
        if val['features']:
            flags.append('Features')
        if val['index']:
            flags.append('Index')
        status = ', '.join(flags) if flags else 'empty'
        logger.info(f"  {key}: [{status}] ({val['lat']:.4f}, {val['lon']:.4f})")

    if upload_tracker(r2, tracker):
        logger.info(f"Uploaded {TRACKER_KEY}")
    else:
        logger.error("Failed to upload tracker")
