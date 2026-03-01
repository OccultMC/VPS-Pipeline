"""Consumer: batch inference with MegaLoc, write features to memory-mapped .npy.

Reads (panoid, view_idx, tensor) items from the queue, batches them, runs FP16
inference on GPU, and writes 8448-dim L2-normalized features to a memory-mapped
numpy file. Uploads the .npy and _Metadata.json to R2 when done.

Uses the MegaLoc model from the Feature Extraction reference
(Code/Feature Extraction/megaloc_model.py).
"""

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import torch
from safetensors.torch import load_file

from megaloc_model import MegaLoc
from r2_client import R2Client

logger = logging.getLogger(__name__)

FEAT_DIM = 8448
MODEL_PATH = "/app/model.safetensors"


def load_model(device: torch.device) -> MegaLoc:
    """Load MegaLoc model with pre-baked weights.

    Uses the exact same architecture as the reference:
        model = MegaLoc()  # feat_dim=8448 by default
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)
    """
    model = MegaLoc(feat_dim=FEAT_DIM)
    state_dict = load_file(MODEL_PATH)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).half().eval()
    logger.info(f"MegaLoc loaded on {device} (FP16, feat_dim={FEAT_DIM})")
    return model


@torch.inference_mode()
def _process_batch(model, device, batch_tensors):
    """Run MegaLoc on a batch, return FP16 numpy features.

    Separated so it can be called via run_in_executor to avoid blocking
    the async event loop during GPU inference.
    """
    batch_tensor = torch.stack(batch_tensors).to(device)
    features = model(batch_tensor)  # [B, 8448] FP16, L2-normalized
    return features.cpu().numpy()


async def run_consumer(
    queue: asyncio.Queue,
    r2: R2Client,
    output_r2_path: str,
    city_name: str,
    worker_number: int,
    total_workers: int,
    expected_panos: int,
    batch_size: int = 32,
    on_progress=None,
):
    """Consume items from queue, run inference, write features.

    Args:
        queue: asyncio.Queue yielding (panoid, view_idx, tensor) or None sentinel.
        r2: R2Client instance.
        output_r2_path: R2 prefix for feature output (e.g. "Features/US/California/Sacramento/").
        city_name: For output filename.
        worker_number: This worker's number (1-based).
        total_workers: Total number of workers.
        expected_panos: Expected panorama count (for memmap sizing).
        batch_size: GPU batch size (calculated from VRAM).
        on_progress: Optional callback(features_done).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model in executor to not block event loop
    loop = asyncio.get_running_loop()
    inference_executor = ThreadPoolExecutor(max_workers=1)
    model = await loop.run_in_executor(inference_executor, load_model, device)

    # Pre-allocate memory-mapped file for features
    # Each panorama produces 8 views => 8 feature vectors
    max_features = expected_panos * 8
    local_dir = "/tmp/features"
    os.makedirs(local_dir, exist_ok=True)

    npy_filename = f"{city_name}_{worker_number}_{total_workers}.npy"
    npy_path = os.path.join(local_dir, npy_filename)

    features_mmap = np.memmap(npy_path, dtype=np.float16, mode="w+", shape=(max_features, FEAT_DIM))

    write_idx = 0
    features_done = 0
    batch_items = []

    t_start = time.time()

    while True:
        item = await queue.get()

        if item is None:
            # Sentinel — process remaining batch
            if batch_items:
                tensors = [it[2] for it in batch_items]
                features_np = await loop.run_in_executor(
                    inference_executor, _process_batch, model, device, tensors
                )
                n = features_np.shape[0]
                features_mmap[write_idx:write_idx + n] = features_np
                write_idx += n
                features_done += len(batch_items)
                if on_progress:
                    on_progress(features_done)
            break

        batch_items.append(item)

        if len(batch_items) >= batch_size:
            tensors = [it[2] for it in batch_items]
            features_np = await loop.run_in_executor(
                inference_executor, _process_batch, model, device, tensors
            )
            n = features_np.shape[0]
            features_mmap[write_idx:write_idx + n] = features_np
            write_idx += n
            features_done += len(batch_items)
            batch_items = []
            if on_progress:
                on_progress(features_done)

    # Flush memmap
    features_mmap.flush()
    inference_executor.shutdown(wait=False)

    actual_features = write_idx
    elapsed = time.time() - t_start
    logger.info(f"Inference done: {actual_features} features in {elapsed:.1f}s "
                f"({actual_features / max(elapsed, 0.01):.1f} feat/s)")

    if actual_features == 0:
        logger.warning("No features produced — nothing to upload")
        return 0

    # Save trimmed copy for upload
    trimmed = np.array(features_mmap[:actual_features])
    del features_mmap

    trimmed_path = os.path.join(local_dir, f"trimmed_{npy_filename}")
    np.save(trimmed_path, trimmed)

    # Build metadata JSON (spec Section 6.7)
    num_panos = actual_features // 8
    metadata = {
        "path": output_r2_path.replace("Features/", "").rstrip("/"),
        "worker": worker_number,
        "total_workers": total_workers,
        "num_panos": num_panos,
        "num_views_per_pano": 8,
        "total_views": actual_features,
        "feature_dim": FEAT_DIM,
        "model": "MegaLoc",
        "precision": "fp16",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    meta_filename = f"{city_name}_{worker_number}_{total_workers}_Metadata.json"
    meta_path = os.path.join(local_dir, meta_filename)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Ensure output path has trailing slash
    if output_r2_path and not output_r2_path.endswith("/"):
        output_r2_path += "/"

    # Upload to R2
    r2_feat_key = f"{output_r2_path}{npy_filename}"
    r2_meta_key = f"{output_r2_path}{meta_filename}"

    r2.upload_file(trimmed_path, r2_feat_key)
    r2.upload_file(meta_path, r2_meta_key)

    logger.info(f"Uploaded features to {r2_feat_key} ({actual_features} vectors)")
    logger.info(f"Uploaded metadata to {r2_meta_key}")

    # Cleanup
    for p in [npy_path, trimmed_path, meta_path]:
        if os.path.exists(p):
            os.remove(p)

    return actual_features
