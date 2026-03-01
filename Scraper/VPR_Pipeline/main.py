"""VPR Pipeline — main entry point.

Runs inside a Vast.AI GPU Docker container. Flow:
1. Print system diagnostics (GPU, VRAM, CUDA, RAM)
2. Calculate optimal batch size from available VRAM
3. Download CSV from R2 and count panoramas
4. Start producer (download + view extraction) and consumer (MegaLoc inference)
5. Upload features + metadata to R2
6. Upload final log
7. Self-destruct
"""

import asyncio
import csv
import io
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("pipeline")

from r2_client import R2Client
from producer import run_producer
from consumer import run_consumer, FEAT_DIM
from pipeline_logger import PipelineLogger
from self_destruct import self_destruct


# ── System Diagnostics ──────────────────────────────────────────────────────

def _get_gpu_info() -> dict:
    """Detect GPU name, VRAM, CUDA version, and driver version."""
    info = {
        "cuda_version": "N/A",
        "driver_version": "N/A",
        "gpu_name": "N/A",
        "vram_total_mb": 0,
        "vram_available_mb": 0,
        "ram_total_gb": 0,
    }

    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "N/A"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_total_mb"] = props.total_mem // (1024 * 1024)
            free, _ = torch.cuda.mem_get_info(0)
            info["vram_available_mb"] = free // (1024 * 1024)
    except Exception:
        pass

    # Driver version from nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        info["driver_version"] = out.split("\n")[0].strip()
    except Exception:
        pass

    # System RAM
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    info["ram_total_gb"] = kb // (1024 * 1024)
                    break
    except Exception:
        pass

    return info


def _calculate_batch_size(vram_available_mb: int) -> int:
    """Calculate optimal batch size based on available VRAM.

    MegaLoc ViT-B/14 at FP16 with 322x322 input:
    - Model weights: ~350 MB
    - Per-sample activation memory: ~45 MB at FP16
    - Reserve 512 MB for CUDA overhead

    Hard cap at 64 regardless of VRAM.
    """
    MODEL_FOOTPRINT_MB = 400
    PER_SAMPLE_MB = 45
    CUDA_OVERHEAD_MB = 512
    MAX_BATCH = 64

    usable = vram_available_mb - MODEL_FOOTPRINT_MB - CUDA_OVERHEAD_MB
    if usable < PER_SAMPLE_MB:
        return 0

    batch_size = min(usable // PER_SAMPLE_MB, MAX_BATCH)
    return max(1, batch_size)


def _print_diagnostics(gpu_info: dict, worker_number: int, total_workers: int,
                       csv_r2_path: str, batch_size: int, queue_size: int):
    """Print startup diagnostics (spec Section 6.2)."""
    print("=== VPR Pipeline Worker ===")
    print(f"Worker: {worker_number} / {total_workers}")
    print(f"CSV: {csv_r2_path}")
    print(f"CUDA Version: {gpu_info['cuda_version']}")
    print(f"Driver Version: {gpu_info['driver_version']}")
    print(f"GPU: {gpu_info['gpu_name']}")
    print(f"VRAM Total: {gpu_info['vram_total_mb']} MB")
    print(f"VRAM Available: {gpu_info['vram_available_mb']} MB")
    print(f"RAM Total: {gpu_info['ram_total_gb']} GB")
    print(f"Calculated Batch Size: {batch_size}")
    print(f"Max Queue Size: {queue_size}")
    print("============================")


# ── Main ────────────────────────────────────────────────────────────────────

async def main():
    t0 = time.time()

    # ── Read config from env ──
    csv_r2_path = os.environ.get("CSV_R2_PATH", "")
    output_r2_path = os.environ.get("OUTPUT_R2_PATH", "")
    logs_r2_path = os.environ.get("LOGS_R2_PATH", "")
    worker_number = int(os.environ.get("WORKER_NUMBER", "1"))
    total_workers = int(os.environ.get("TOTAL_WORKERS", "1"))
    city_name = os.environ.get("CITY_NAME", "Unknown")

    if not csv_r2_path:
        logger.error("CSV_R2_PATH not set")
        sys.exit(1)

    # ── GPU diagnostics & batch size ──
    gpu_info = _get_gpu_info()
    queue_size = 500
    batch_size = _calculate_batch_size(gpu_info["vram_available_mb"])

    if batch_size == 0:
        logger.error(
            f"Insufficient VRAM ({gpu_info['vram_available_mb']} MB available). "
            "Cannot fit even batch size 1. Exiting."
        )
        sys.exit(1)

    _print_diagnostics(gpu_info, worker_number, total_workers,
                       csv_r2_path, batch_size, queue_size)

    # ── Init R2 ──
    r2 = R2Client()

    # ── Download CSV once — pass bytes to producer (no second download) ──
    csv_bytes = r2.download_bytes(csv_r2_path)
    if csv_bytes is None:
        logger.error(f"Failed to download CSV: {csv_r2_path}")
        sys.exit(1)

    reader = csv.reader(io.StringIO(csv_bytes.decode("utf-8")))
    pano_count = sum(1 for row in reader if len(row) >= 3 and row[0].strip())
    logger.info(f"CSV contains {pano_count} panoramas")

    # ── Init logger ──
    pipe_logger = PipelineLogger(
        r2=r2,
        logs_r2_path=logs_r2_path,
        city_name=city_name,
        worker_number=worker_number,
        total_workers=total_workers,
        total_panos=pano_count,
    )
    pipe_logger.start()

    # ── Producer-Consumer queue ──
    queue = asyncio.Queue(maxsize=queue_size)

    try:
        def on_scrape_progress(done, total, failed):
            pipe_logger.update_scrape(done, total, failed)

        def on_feature_progress(features_done):
            pipe_logger.update_features(features_done)

        producer_task = asyncio.create_task(
            run_producer(
                queue=queue,
                csv_bytes=csv_bytes,
                max_concurrent=150,
                on_progress=on_scrape_progress,
            )
        )

        consumer_task = asyncio.create_task(
            run_consumer(
                queue=queue,
                r2=r2,
                output_r2_path=output_r2_path,
                city_name=city_name,
                worker_number=worker_number,
                total_workers=total_workers,
                expected_panos=pano_count,
                batch_size=batch_size,
                on_progress=on_feature_progress,
            )
        )

        await asyncio.gather(producer_task, consumer_task)

        elapsed = time.time() - t0
        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        pipe_logger.set_done()

    except Exception as e:
        logger.exception("Pipeline failed")
        pipe_logger.set_error(str(e))

    # ── Final log upload ──
    await pipe_logger.stop()

    # ── Self-destruct ──
    logger.info("Initiating self-destruct...")
    await self_destruct()


if __name__ == "__main__":
    asyncio.run(main())
