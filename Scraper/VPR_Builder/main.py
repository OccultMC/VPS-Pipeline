"""VPR Builder — main entry point.

Runs inside a Vast.AI CPU Docker container. Flow:
1. Read env vars (features path, output path, R2 credentials, FAISS params)
2. Download all .npy feature files from R2
3. Build FAISS index (PQ or IVF_PQ)
4. Upload index + config to R2
5. Self-destruct
"""

import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("builder")

from r2_client import R2Client
from index_builder import build_index
from self_destruct import destroy_self


def main():
    t0 = time.time()

    # ── Read config from env ──
    features_r2_path = os.environ.get("FEATURES_R2_PATH", "")
    output_r2_path = os.environ.get("OUTPUT_R2_PATH", "")
    index_type = os.environ.get("INDEX_TYPE", "PQ")
    m = int(os.environ.get("M", "256"))
    nbits = int(os.environ.get("NBITS", "8"))
    training_samples = int(os.environ.get("TRAINING_SAMPLES", "1000000"))
    nlist = int(os.environ.get("NLIST", "0"))
    nprobe = int(os.environ.get("NPROBE", "100"))

    logger.info(f"Builder starting")
    logger.info(f"Features: {features_r2_path}")
    logger.info(f"Output: {output_r2_path}")
    logger.info(f"Index: {index_type}, m={m}, nbits={nbits}, nlist={nlist}")

    if not features_r2_path:
        logger.error("FEATURES_R2_PATH not set")
        sys.exit(1)

    # ── Init R2 ──
    r2 = R2Client()

    # ── Build index ──
    success = build_index(
        r2=r2,
        features_r2_path=features_r2_path,
        output_r2_path=output_r2_path,
        index_type=index_type,
        m=m,
        nbits=nbits,
        training_samples=training_samples,
        nlist=nlist,
        nprobe=nprobe,
    )

    elapsed = time.time() - t0
    if success:
        logger.info(f"Builder complete in {elapsed:.1f}s")
    else:
        logger.error(f"Builder failed after {elapsed:.1f}s")

    # ── Self-destruct ──
    logger.info("Initiating self-destruct...")
    destroy_self()


if __name__ == "__main__":
    main()
