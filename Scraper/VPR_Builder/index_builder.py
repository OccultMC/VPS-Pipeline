"""FAISS index builder — downloads .npy feature files from R2, concatenates,
trains a PQ or IVF_PQ index, and uploads the result.

Adapted from build_megaloc_index.py for the VPR Builder Docker container.
Uses METRIC_INNER_PRODUCT (cosine similarity for L2-normalized vectors).
"""

import gc
import json
import logging
import os
import time
from typing import List

import faiss
import numpy as np

from r2_client import R2Client

logger = logging.getLogger(__name__)

FEAT_DIM = 8448


def build_index(
    r2: R2Client,
    features_r2_path: str,
    output_r2_path: str,
    index_type: str = "PQ",
    m: int = 256,
    nbits: int = 8,
    training_samples: int = 1_000_000,
    nlist: int = 0,
    nprobe: int = 100,
) -> bool:
    """Download feature .npy files from R2, build FAISS index, upload result.

    Args:
        r2: R2Client instance.
        features_r2_path: R2 prefix (e.g. "Features/US/California/Sacramento/").
        output_r2_path: R2 prefix for index output (e.g. "Index/US/California/Sacramento/").
        index_type: "PQ" or "IVF_PQ".
        m: Number of sub-quantizers (must divide FEAT_DIM).
        nbits: Bits per sub-quantizer code.
        training_samples: Max vectors for training.
        nlist: Number of IVF clusters (for IVF_PQ). 0 = auto.
        nprobe: Number of probes at search time (stored in config).

    Returns:
        True on success.
    """
    local_dir = "/tmp/builder"
    os.makedirs(local_dir, exist_ok=True)

    # ── Step 1: Download all feature .npy files ──
    logger.info(f"Listing features under {features_r2_path}")
    keys = r2.list_objects(features_r2_path)
    npy_keys = [k for k in keys if k.endswith(".npy") and "_meta" not in k]

    if not npy_keys:
        logger.error("No .npy feature files found")
        return False

    logger.info(f"Found {len(npy_keys)} feature files")

    # Download and concatenate
    all_features = []
    total_vectors = 0

    for key in npy_keys:
        local_path = os.path.join(local_dir, os.path.basename(key))
        logger.info(f"Downloading {key}...")
        if not r2.download_file(key, local_path):
            logger.warning(f"Failed to download {key}, skipping")
            continue

        arr = np.load(local_path)
        # Convert FP16 to FP32 for FAISS
        if arr.dtype == np.float16:
            arr = arr.astype(np.float32)

        logger.info(f"  {arr.shape[0]} vectors, dim={arr.shape[1]}")
        all_features.append(arr)
        total_vectors += arr.shape[0]

        # Remove downloaded file to save disk
        os.remove(local_path)

    if not all_features:
        logger.error("No feature files were successfully downloaded")
        return False

    # Concatenate into single array
    logger.info(f"Concatenating {total_vectors} vectors...")
    features = np.concatenate(all_features, axis=0)
    del all_features
    gc.collect()

    n_vectors = features.shape[0]
    feature_dim = features.shape[1]
    logger.info(f"Total: {n_vectors} vectors, dim={feature_dim}")
    logger.info(f"Dataset size: {features.nbytes / (1024**3):.2f} GB")

    # Normalize (should already be L2-normed, but re-normalize for safety)
    faiss.normalize_L2(features)

    # ── Step 2: Validate parameters ──
    if feature_dim % m != 0:
        old_m = m
        m = feature_dim // (feature_dim // m)
        logger.warning(f"Adjusted m from {old_m} to {m} (must divide {feature_dim})")

    # ── Step 3: Sample training data ──
    train_n = min(n_vectors, training_samples)
    logger.info(f"Sampling {train_n} vectors for training...")
    rng = np.random.default_rng(42)
    train_indices = np.sort(rng.choice(n_vectors, size=train_n, replace=False))
    train_data = features[train_indices].copy()
    logger.info(f"Training data: {train_data.nbytes / (1024**2):.0f} MB")

    # ── Step 4: Build index ──
    d = feature_dim
    index_type_upper = index_type.upper()

    if index_type_upper == "PQ":
        logger.info(f"Building PQ index: d={d}, m={m}, nbits={nbits}")
        index = faiss.IndexPQ(d, m, nbits, faiss.METRIC_INNER_PRODUCT)

    elif index_type_upper in ("IVF_PQ", "IVFPQ"):
        # Auto nlist
        if nlist <= 0:
            nlist = max(64, 2 ** int(np.log2(n_vectors // 39)))
            logger.info(f"Auto nlist: {nlist}")

        # Cap nlist
        max_nlist = n_vectors // 39
        if nlist > max_nlist:
            old = nlist
            nlist = max(64, 2 ** int(np.log2(max_nlist)))
            logger.info(f"Adjusted nlist: {old} -> {nlist}")

        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
        logger.info(f"Building IVF_PQ index: d={d}, nlist={nlist}, m={m}, nbits={nbits}")

    else:
        logger.error(f"Unknown index_type: {index_type}")
        return False

    # Train
    logger.info("Training index...")
    t_train = time.time()
    index.train(train_data)
    logger.info(f"Training complete in {time.time() - t_train:.1f}s")
    del train_data
    gc.collect()

    # Add vectors in batches
    ADD_BATCH = 10_000
    logger.info(f"Adding {n_vectors} vectors in batches of {ADD_BATCH}...")
    t_add = time.time()
    for start in range(0, n_vectors, ADD_BATCH):
        end = min(start + ADD_BATCH, n_vectors)
        batch = features[start:end].copy()
        faiss.normalize_L2(batch)
        index.add(batch)
        del batch
        if (start // ADD_BATCH) % 100 == 0:
            logger.info(f"  Added {min(end, n_vectors)}/{n_vectors}")
    logger.info(f"Adding complete in {time.time() - t_add:.1f}s, total={index.ntotal}")

    del features
    gc.collect()

    # ── Step 5: Save index locally ──
    index_path = os.path.join(local_dir, "megaloc.index")
    faiss.write_index(index, index_path)
    index_size_mb = os.path.getsize(index_path) / (1024**2)
    logger.info(f"Index saved: {index_path} ({index_size_mb:.1f} MB)")

    # Save config
    config = {
        "n_vectors": int(index.ntotal),
        "dimension": feature_dim,
        "index_type": index_type_upper,
        "m": m,
        "nbits": nbits,
        "nlist": nlist if "IVF" in index_type_upper else 0,
        "nprobe": nprobe,
        "index_file": "megaloc.index",
    }
    config_path = os.path.join(local_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # ── Step 6: Upload to R2 ──
    logger.info(f"Uploading index to {output_r2_path}...")
    r2.upload_file(index_path, f"{output_r2_path}megaloc.index")
    r2.upload_file(config_path, f"{output_r2_path}config.json")

    # Also upload concatenated metadata if available
    meta_keys = [k for k in keys if k.endswith("_meta.npy")]
    if meta_keys:
        _upload_merged_metadata(r2, meta_keys, output_r2_path, local_dir)

    logger.info("Index build and upload complete")
    return True


def _upload_merged_metadata(r2: R2Client, meta_keys: List[str], output_r2_path: str, local_dir: str):
    """Download and merge all metadata files, upload as single file."""
    all_meta = []
    for key in meta_keys:
        local_path = os.path.join(local_dir, os.path.basename(key))
        if r2.download_file(key, local_path):
            with open(local_path) as f:
                all_meta.extend(f.readlines())
            os.remove(local_path)

    if all_meta:
        merged_path = os.path.join(local_dir, "metadata.csv")
        with open(merged_path, "w") as f:
            f.writelines(all_meta)
        r2.upload_file(merged_path, f"{output_r2_path}metadata.csv")
        logger.info(f"Uploaded merged metadata: {len(all_meta)} entries")
