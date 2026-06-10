#!/usr/bin/env python3
"""
VPS Pipeline: Google Street View Downloader → MegaLoc Feature Extraction → R2 Upload

Queue-based distributed worker — pulls 1K-pano chunks from a shared Redis queue,
processes each chunk, uploads results to R2, and grabs the next chunk.

Workers self-destruct when the queue is empty and all tasks are done.

Progress is logged to stdout in structured format for vastai logs polling:
    PROGRESS|{instance_id}|{chunk_id}|{processed}|{total}|{status}
"""

import asyncio
import csv
import gc
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration (Hardcoded per spec)
# ═══════════════════════════════════════════════════════════════════════════════

HARDCODED_CONFIG = {
    # Lowered from 150 → 30 to stay under Google's per-IP 403 threshold.
    # Local scraper sustains 60 panos/sec @ max_threads=20 without ever
    # triggering the block; 30 is a modest bump with GPU pipelining.
    # Tunable via MAX_THREADS env var.
    'max_threads': int(os.environ.get('MAX_THREADS', '30')),

    # Server-rendered thumbnail mode: 8 perspective views per pano fetched
    # directly from streetviewpixels-pa.googleapis.com — no tile fetch, no
    # equirect stitch, no client-side projection.
    'view_resolution': 322,    # MegaLoc input dim (fed straight to the GPU)
    'view_fov': 70.0,          # degrees
    'num_views': 8,            # 360° / 8 = 45° spacing
    'view_offset': 0.0,        # base yaw added to heading_deg (or 0 if absent)
    'view_pitch': 5.0,         # +5° = looks down slightly (matches Google URL pitch)

    'output_dir': None,
    # batch_size is set dynamically by GpuExtractor._probe_max_batch_size()
    # based on the GPU's VRAM minus model baseline minus a safety pad. The
    # OOM auto-shrink (in _run_inference) further halves it on demand.
    'queue_size': 512,
    # Small random jitter (seconds) applied before each pano dispatch —
    # de-synchronises the 30 concurrent workers so we don't send exact
    # bursts every time the semaphore refills.
    'pano_jitter_max': float(os.environ.get('PANO_JITTER_MAX', '0.3')),
}


# Network signature is now handled entirely by curl_cffi's `impersonate`
# flag at AsyncSession-creation time below — it sets a Chrome-identical TLS
# fingerprint (JA3/JA4), HTTP/2/3 negotiation, full Sec-Ch-Ua header set, and
# Accept-Encoding incl. brotli. The old _BROWSER_HEADERS dict only spoofed
# User-Agent + a few headers and was still flagged by Google's abuse gate
# under load because the TLS handshake gave aiohttp away. No explicit headers
# block needed here anymore.

# ═══════════════════════════════════════════════════════════════════════════════
# Environment Variables
# ═══════════════════════════════════════════════════════════════════════════════

REDIS_URL = os.environ.get('REDIS_URL', '')
REDIS_TOKEN = os.environ.get('REDIS_TOKEN', '')
REGION = os.environ.get('REGION', '')
CSV_BUCKET_PREFIX = os.environ.get('CSV_BUCKET_PREFIX', 'CSV')
FEATURES_BUCKET_PREFIX = os.environ.get('FEATURES_BUCKET_PREFIX', 'Features')
CITY_NAME = os.environ.get('CITY_NAME', 'Unknown')
INSTANCE_ID = (os.environ.get('INSTANCE_ID', '')
               or os.environ.get('CONTAINER_ID', '')
               or os.environ.get('VAST_CONTAINERLABEL', ''))
VAST_API_KEY = os.environ.get('VAST_API_KEY', '')

MAX_DISK_GB = 100
MIN_FREE_GB = 5
TOTAL_CHUNKS = 0  # Set from Redis metadata at startup


def _chunk_num(chunk_id: str) -> int:
    """Convert Redis chunk ID to 1-based number: 'chunk_0001' → 1."""
    return int(chunk_id.split('_')[1])


def _output_base(city: str, chunk_id: str) -> str:
    """Return '{city}_{N}.{total}' for output filenames."""
    return f"{city}_{_chunk_num(chunk_id)}.{TOTAL_CHUNKS}"


def _redis_retry(fn, *args, retries=5, delay=3, label="redis"):
    """Retry a Redis call with exponential backoff. Returns True on success."""
    for attempt in range(1, retries + 1):
        try:
            fn(*args)
            return True
        except Exception as e:
            print(f"[WARN] {label} attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(delay * attempt)
    print(f"[ERROR] {label} failed after {retries} attempts")
    return False

# ═══════════════════════════════════════════════════════════════════════════════
# Imports: Street View Downloader
# ═══════════════════════════════════════════════════════════════════════════════

from curl_cffi.requests import AsyncSession
import random
from gsv_thumbnail import fetch_thumbnail_view
from concurrent.futures import ThreadPoolExecutor

# ═══════════════════════════════════════════════════════════════════════════════
# Imports: Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════════

import torch
from torchvision import transforms

# ═══════════════════════════════════════════════════════════════════════════════
# R2 Storage
# ═══════════════════════════════════════════════════════════════════════════════

from r2_storage import R2Client

# ═══════════════════════════════════════════════════════════════════════════════
# Redis Task Queue
# ═══════════════════════════════════════════════════════════════════════════════

from redis_queue import TaskQueue


# ═══════════════════════════════════════════════════════════════════════════════
# View Item & Shared State
# ═══════════════════════════════════════════════════════════════════════════════

_SENTINEL = None


class _ThrottledPrinter:
    """Print a labelled message at most once per `interval` seconds.

    Used to compress log spam from per-pano failures (403, black-views,
    decode errors) so a long-running scrape doesn't generate gigabytes of
    duplicate stderr lines while a transient issue is happening.
    """
    def __init__(self, interval: float = 10.0):
        self.interval = interval
        self._last: Dict[str, float] = {}
        self._dropped: Dict[str, int] = {}
        self._lock = threading.Lock()

    def maybe(self, key: str, msg: str):
        now = time.time()
        with self._lock:
            last = self._last.get(key, 0.0)
            if now - last >= self.interval:
                dropped = self._dropped.pop(key, 0)
                self._last[key] = now
                suffix = f" (+{dropped} suppressed in last {self.interval:.0f}s)" if dropped else ""
                print(msg + suffix, flush=True)
            else:
                self._dropped[key] = self._dropped.get(key, 0) + 1


_throttle = _ThrottledPrinter(interval=10.0)


class IPBlockedError(RuntimeError):
    """Raised when the probe detects Google has 403-blocked this instance's
    source IP. The chunk should be unclaimed (not fail_task'd) and the worker
    should self-destruct so a new instance can try from a fresh IP."""


class ViewItem:
    __slots__ = ('panoid', 'view_data', 'lat', 'lng', 'sink')
    def __init__(self, panoid: str, view_data, lat: float, lng: float, sink=None):
        self.panoid = panoid
        self.view_data = view_data  # RGB numpy array (uint8 HWC)
        self.lat = lat
        self.lng = lng
        # SharedState this view belongs to. With cross-chunk overlap two
        # chunks' items share one queue — the extraction loop routes each
        # row to its own chunk's memmap/JSONL via this reference.
        self.sink = sink

class SharedState:
    """Thread-safe writing to memmap + metadata + failures."""
    def __init__(self, features_memmap, metadata_file_path, failed_file_path, start_idx=0):
        self.memmap = features_memmap
        self.write_idx = start_idx
        self.lock = threading.Lock()
        self.metadata_handle = open(metadata_file_path, 'w', encoding='utf-8')
        self.failed_handle = open(failed_file_path, 'w', encoding='utf-8')
        self._batch_count = 0
        # Views handed to the extractor but dropped (decode failure, OOM
        # skip, whole-batch error). Completion accounting for a chunk is:
        # downloader dead AND write_idx + dropped >= stats['views_produced'].
        self.dropped = 0

    def write_batch(self, features_batch: np.ndarray, metadata_batch: List[dict]):
        n = len(features_batch)
        if n == 0:
            return
        # HARD alignment check. If features and metadata ever desync, every
        # subsequent NPY row maps to the wrong panoid — this is exactly the
        # bug that produced the corrupt indexes. Fail loud, not silent.
        if len(metadata_batch) != n:
            raise AssertionError(
                f"SharedState.write_batch: features ({n}) != metadata "
                f"({len(metadata_batch)}) — refusing to write misaligned data"
            )
        with self.lock:
            start = self.write_idx
            end = start + n
            self.memmap[start:end] = features_batch
            for i, meta in enumerate(metadata_batch):
                meta['feature_index'] = start + i
                self.metadata_handle.write(json.dumps(meta) + '\n')
            self.metadata_handle.flush()
            self.write_idx = end
            self._batch_count += 1
            # Flush memmap to disk periodically to keep RSS low
            # Run in background to avoid blocking the GPU thread
            if self._batch_count % 100 == 0:
                mm = self.memmap
                threading.Thread(target=lambda: mm.flush(), daemon=True).start()

    def log_failure(self, panoid: str, reason: str):
        with self.lock:
            entry = {'panoid': panoid, 'reason': str(reason), 'timestamp': time.time()}
            self.failed_handle.write(json.dumps(entry) + '\n')
            self.failed_handle.flush()

    def close(self):
        with self.lock:
            if self.metadata_handle:
                self.metadata_handle.close()
                self.metadata_handle = None
            if self.failed_handle:
                self.failed_handle.close()
                self.failed_handle = None


# ═══════════════════════════════════════════════════════════════════════════════
# GPU Feature Extractor
# ═══════════════════════════════════════════════════════════════════════════════

GPU_INIT_TIMEOUT = int(os.environ.get('GPU_INIT_TIMEOUT', '300'))  # 5 min default


class _InitWatchdog:
    """Watchdog that kills the process if init hangs too long."""
    def __init__(self, timeout_sec: int, stage: str = "unknown"):
        self.timeout = timeout_sec
        self.stage = stage
        self._timer = None

    def start(self, stage: str = None):
        if stage:
            self.stage = stage
        self.cancel()
        self._timer = threading.Timer(self.timeout, self._on_timeout)
        self._timer.daemon = True
        self._timer.start()

    def cancel(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _on_timeout(self):
        msg = (f"[FATAL] Watchdog timeout after {self.timeout}s during: {self.stage}. "
               f"Process is stuck — forcing exit.")
        print(msg, flush=True)
        try:
            upload_logs_to_r2()
        except Exception:
            pass
        os._exit(1)


def _run_with_timeout(fn, timeout_sec: int, stage: str):
    """Run fn() in a thread with a timeout. Raises TimeoutError if it hangs."""
    result_container = [None]
    error_container = [None]

    def _target():
        try:
            result_container[0] = fn()
        except Exception as e:
            error_container[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        raise TimeoutError(
            f"{stage} timed out after {timeout_sec}s — process is hung. "
            f"This usually means a network stall (model download) or CUDA driver issue."
        )

    if error_container[0] is not None:
        raise error_container[0]

    return result_container[0]


class GpuExtractor:
    def __init__(self, gpu_id: int = 0, pool_mode: bool = False):
        """
        Args:
            gpu_id: cuda device index this extractor binds to.
                Used by the multi-GPU pool path so each child process
                pins to one specific GPU. Single-GPU path leaves this 0.
            pool_mode: when True, skip nn.DataParallel wrap. The pool spawns
                one extractor per device, so DataParallel would actively
                make things worse (round-trips through cuda:0). When False
                (legacy single-process path), the old DataParallel behavior
                is preserved for backwards compatibility.
        """
        self.gpu_id = gpu_id
        self._pool_mode = pool_mode
        t0 = time.time()
        self._watchdog = _InitWatchdog(GPU_INIT_TIMEOUT * 2 + 120, "gpu_init_overall")
        self._watchdog.start()

        try:
            self._init_gpu(t0)
        finally:
            self._watchdog.cancel()

    @staticmethod
    def _load_model():
        """Load MegaLoc model.

        Primary: baked safetensors at /app/models/megaloc/model.safetensors.
          Baked weights are deterministic, fast, and never depend on network.
        Fallback: torch.hub get_trained_model (HuggingFace) only when the
          baked file is missing — CI/dev environments that don't bake it in.

        Rationale for baked-first: torch.hub.load() stalls indefinitely on
        network blips, exhausting GPU_INIT_TIMEOUT (300s) before the fallback
        path runs. That kills the whole init and leaves CUDA in a bad state,
        producing SIGSEGV-139 exits at container teardown. The baked model
        works in <2s with zero network dependency, so we use it first when
        available.
        """
        errors = []

        # ── Primary: baked model ──
        model_path = Path('/app/models/megaloc/model.safetensors')
        if model_path.exists():
            try:
                from safetensors.torch import load_file

                size_mb = model_path.stat().st_size / 1e6
                print(f"[INIT]   Loading baked weights ({size_mb:.1f}MB) from {model_path}", flush=True)
                state_dict = load_file(str(model_path))

                hub_dir = Path(torch.hub.get_dir()) / 'gmberton_MegaLoc_main'
                if not hub_dir.exists():
                    raise FileNotFoundError(f"MegaLoc architecture not found at {hub_dir}")

                sys.path.insert(0, str(hub_dir))
                try:
                    import importlib
                    megaloc_module = importlib.import_module('megaloc_model')
                    model = megaloc_module.MegaLoc()
                    try:
                        model.load_state_dict(state_dict, strict=True)
                        print("[INIT]   Model loaded from baked weights (strict=True, all keys matched)", flush=True)
                    except RuntimeError:
                        # Reconcile baked weights against the current architecture.
                        # The baked safetensors was exported from an older gmberton
                        # /MegaLoc revision with two diffs vs current:
                        #   1. backbone wrapped in a container so keys are
                        #      `backbone.model.*` instead of `backbone.*`
                        #   2. extra `backbone.mask_token` from DINOv2 pretraining,
                        #      not used at inference and absent from current arch
                        DROPPED = {'backbone.mask_token'}
                        cleaned = {
                            ('backbone.' + k[len('backbone.model.'):]
                             if k.startswith('backbone.model.') else k): v
                            for k, v in state_dict.items()
                            if k not in DROPPED
                        }
                        # strict=False so unexpected keys are ignored, but check
                        # missing_keys ourselves — any missing key means a layer
                        # would run with random weights → silently wrong outputs.
                        result = model.load_state_dict(cleaned, strict=False)
                        if result.missing_keys:
                            raise RuntimeError(
                                f"Baked weights incompatible after cleanup — "
                                f"{len(result.missing_keys)} missing key(s), "
                                f"first few: {result.missing_keys[:5]}"
                            )
                        n_dropped = len(state_dict) - len(cleaned)
                        n_unexpected = len(result.unexpected_keys)
                        print(
                            f"[INIT]   Model loaded from baked weights "
                            f"(remapped backbone.model.*→backbone.*, "
                            f"dropped {n_dropped} pretraining key(s), "
                            f"{n_unexpected} unexpected key(s) ignored)",
                            flush=True,
                        )
                    return model
                finally:
                    sys.path.pop(0)
            except Exception as e:
                errors.append(f"baked model: {e}")
                # Hard-fail instead of falling through to torch.hub. The hub
                # fetches whatever HuggingFace serves at the time, which can
                # silently drift between extraction and query time and produces
                # features in a different embedding space — corrupting search
                # recall for any chunks indexed against a different version.
                # Pinning to baked safetensors keeps every worker (and the
                # server) on the same model checkpoint forever.
                raise RuntimeError(
                    f"Baked MegaLoc weights at {model_path} failed to load and "
                    f"torch.hub fallback is disabled to prevent encoder drift. "
                    f"Underlying error: {e}"
                )

        # No baked model present — refuse to start (would otherwise silently
        # use torch.hub and cause encoder mismatch with previously indexed cities).
        raise RuntimeError(
            f"No baked MegaLoc weights at {model_path}. "
            f"Workers must use a pinned safetensors file to keep all extracted "
            f"features in a single embedding space; torch.hub fallback was removed "
            f"because HuggingFace can serve different weights over time. "
            f"Bake models/megaloc/model.safetensors into the Docker image."
        )

    def _init_gpu(self, t0: float):
        # ── Step 1: CUDA check ──
        print(f"[INIT] Step 1/6: Checking CUDA availability...", flush=True)
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            raise RuntimeError(
                "CUDA is not available! Check nvidia-smi, CUDA drivers, and container GPU passthrough. "
                "torch.cuda.is_available() returned False."
            )
        # Pin to the assigned GPU so torch.cuda.* defaults to it.
        # Multi-GPU pool path passes gpu_id != 0; single-GPU path leaves
        # gpu_id=0 and behaviour matches the previous code exactly.
        try:
            torch.cuda.set_device(self.gpu_id)
        except Exception as e:
            raise RuntimeError(
                f"torch.cuda.set_device({self.gpu_id}) failed: {e}"
            )
        gpu_name = torch.cuda.get_device_name(self.gpu_id)
        props = torch.cuda.get_device_properties(self.gpu_id)
        gpu_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024**3)
        print(f"[INIT]   CUDA OK — cuda:{self.gpu_id} {gpu_name}, "
              f"VRAM: {gpu_mem:.1f}GB", flush=True)
        print(f"[INIT]   CUDA version: {torch.version.cuda}, PyTorch: {torch.__version__}", flush=True)

        torch.set_float32_matmul_precision('high')
        self.device = torch.device(f'cuda:{self.gpu_id}')

        # ── Step 2: Load MegaLoc model ──
        print(f"[INIT] Step 2/6: Loading MegaLoc model...", flush=True)
        self._watchdog.start("load_model")
        dl_start = time.time()
        try:
            model = _run_with_timeout(
                self._load_model,
                timeout_sec=GPU_INIT_TIMEOUT,
                stage="load_model"
            )
        except TimeoutError:
            raise RuntimeError(
                f"Model loading timed out after {GPU_INIT_TIMEOUT}s. "
                "Network download or model construction is hung."
            )
        print(f"[INIT]   Model ready in {time.time() - dl_start:.1f}s", flush=True)

        # ── Step 3: Move to GPU ──
        self._watchdog.start("model_to_cuda")
        print(f"[INIT] Step 3/6: Moving model to {self.device}...", flush=True)
        move_start = time.time()
        try:
            model = _run_with_timeout(
                lambda: model.to(self.device).eval(),
                timeout_sec=120,
                stage="model_to_cuda"
            )
        except TimeoutError:
            raise RuntimeError(
                "model.to(cuda) timed out after 120s. CUDA driver may be unresponsive. "
                "Check nvidia-smi and dmesg for GPU errors."
            )
        print(f"[INIT]   Model on GPU in {time.time() - move_start:.1f}s", flush=True)

        # ── Step 4: DataParallel / compile ──
        self._watchdog.start("torch_compile")
        gpu_count = torch.cuda.device_count()
        # In pool_mode the orchestrator spawns one process per GPU; each
        # process must NOT DataParallel-wrap (it would scatter through
        # cuda:0 and undo the partitioning). Legacy single-process path
        # keeps DataParallel for backwards compatibility on multi-GPU
        # machines that fall back to the legacy code path.
        if gpu_count > 1 and not self._pool_mode:
            print(f"[INIT] Step 4/6: Wrapping with DataParallel ({gpu_count} GPUs, "
                  f"legacy single-process path)...", flush=True)
            model = torch.nn.DataParallel(model)
        elif self._pool_mode:
            print(f"[INIT] Step 4/6: pool_mode — DataParallel skipped "
                  f"(one process per GPU)", flush=True)

        if hasattr(torch, 'compile'):
            print(f"[INIT] Step 4/6: torch.compile()...", flush=True)
            compile_start = time.time()
            try:
                model = _run_with_timeout(
                    lambda: torch.compile(model),
                    timeout_sec=120,
                    stage="torch_compile"
                )
                print(f"[INIT]   torch.compile() done in {time.time() - compile_start:.1f}s", flush=True)
            except TimeoutError:
                print(f"[WARN] torch.compile() timed out after 120s — running without compilation (this is OK)", flush=True)
            except Exception as e:
                print(f"[WARN] torch.compile() failed: {type(e).__name__}: {e} — running without compilation", flush=True)
        else:
            print(f"[INIT] Step 4/6: torch.compile not available (PyTorch < 2.0), skipping", flush=True)

        # ── Step 5: Warmup inference ──
        self._watchdog.start("warmup_inference")
        print(f"[INIT] Step 5/6: Warmup inference...", flush=True)
        warmup_start = time.time()
        try:
            dummy = torch.randn(1, 3, 322, 322, device=self.device)
            dummy = (dummy - torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)) / \
                    torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            with torch.no_grad():
                _ = model(dummy)
            del dummy
            torch.cuda.synchronize()
            print(f"[INIT]   Warmup done in {time.time() - warmup_start:.1f}s", flush=True)
        except Exception as e:
            print(f"[WARN] Compiled warmup failed: {type(e).__name__}: {e}", flush=True)
            print(f"[WARN] Falling back to eager mode (disabling torch.compile)...", flush=True)

            try:
                torch._dynamo.reset()
            except Exception:
                pass

            if hasattr(model, '_orig_mod'):
                model = model._orig_mod
                print(f"[WARN] Unwrapped compiled model to original module", flush=True)

            try:
                torch.cuda.empty_cache()
                dummy = torch.randn(1, 3, 322, 322, device=self.device)
                dummy = (dummy - torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)) / \
                        torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                with torch.no_grad():
                    _ = model(dummy)
                del dummy
                torch.cuda.synchronize()
                print(f"[INIT]   Eager warmup OK in {time.time() - warmup_start:.1f}s", flush=True)
            except Exception as e2:
                raise RuntimeError(
                    f"Warmup inference failed in both compiled AND eager mode: {type(e2).__name__}: {e2}. "
                    f"The model may be incompatible with this GPU or CUDA version."
                )

        self.model = model
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.executor = ThreadPoolExecutor(max_workers=16)

        # ── Step 6: Auto batch size probe ──
        self._watchdog.start("batch_size_probe")
        print(f"[INIT] Step 6/6: Probing optimal batch size via VRAM measurement...", flush=True)
        self.batch_size = self._probe_max_batch_size()

        vram_used = torch.cuda.memory_allocated(self.gpu_id) / (1024**3)
        vram_reserved = torch.cuda.memory_reserved(self.gpu_id) / (1024**3)
        # Single greppable success line — used by orchestrator + log scrapers
        # to confirm the worker came up correctly before any chunks are claimed.
        print(f"[INIT] MODEL READY — init={time.time() - t0:.1f}s "
              f"vram={vram_used:.2f}/{vram_reserved:.2f}GB "
              f"batch_size={self.batch_size}", flush=True)

    def _probe_max_batch_size(self) -> int:
        """Compute the safe batch size for this GPU.

        Method:
          1. Run a small batch (PROBE_BATCH=8) through the model under fp16
             autocast and measure peak VRAM delta vs baseline. Probing at 8
             instead of 1 amortizes the fixed per-batch activation overhead,
             producing a clean per-image marginal estimate.
          2. Bump that per-image cost by 10% (conservative pad — accounts for
             input variance and CUDA caching-allocator overhead between runs).
          3. Reserve a 1 GB SAFETY_PAD on top of (model + buffered activations)
             for kernel scratch space, IPC buffers, and fragmentation.
          4. batch = floor((total_vram - baseline - safety_pad) / per_image_buffered)
          5. Clamp to [MIN_BATCH, MAX_BATCH] and round down to power of 2.

        Empirically (RTX 3070 8GB, 100k-image stress test at bs=16):
          baseline ~0.86 GB, per-image marginal ~94 MB → formula picks bs=32
          which would peak at ~3.9 GB (51% of 8 GB) — safe headroom for
          fragmentation + decode pipelining.
        """
        PROBE_BATCH = 8
        SAFETY_PAD_GB = 1.0
        CONSERVATIVE_BUFFER = 1.10  # treat each image as 10% larger than measured
        MIN_BATCH = 8
        MAX_BATCH = 512

        try:
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated(self.gpu_id)
            torch.cuda.reset_peak_memory_stats(self.gpu_id)

            probe = torch.randn(PROBE_BATCH, 3, 322, 322, device=self.device)
            probe = (probe - self.mean) / self.std
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.float16):
                    _ = self.model(probe)
            torch.cuda.synchronize()

            peak = torch.cuda.max_memory_allocated(self.gpu_id)
            del probe
            torch.cuda.empty_cache()

            per_image_bytes = max((peak - baseline) // PROBE_BATCH, 1)
            per_image_buffered = int(per_image_bytes * CONSERVATIVE_BUFFER)

            total_vram = torch.cuda.get_device_properties(self.device).total_memory
            safety_pad = int(SAFETY_PAD_GB * 1024**3)
            free_for_act = max(total_vram - baseline - safety_pad, 0)

            max_batch = max(MIN_BATCH, int(free_for_act / per_image_buffered))
            max_batch = min(max_batch, MAX_BATCH)
            max_batch = 2 ** int(math.log2(max_batch))

            print(f"[INIT]   Auto batch size: {max_batch}", flush=True)
            print(f"[INIT]     vram total={total_vram/1024**3:.1f}GB  "
                  f"model+warmup={baseline/1024**3:.2f}GB  "
                  f"safety_pad={SAFETY_PAD_GB:.1f}GB  "
                  f"free_for_act={free_for_act/1024**3:.2f}GB", flush=True)
            print(f"[INIT]     per_image={per_image_bytes/1024**2:.0f}MB measured "
                  f"+ {int((CONSERVATIVE_BUFFER-1)*100)}% buffer "
                  f"= {per_image_buffered/1024**2:.0f}MB used for sizing "
                  f"(probe_batch={PROBE_BATCH})", flush=True)
            return max_batch
        except Exception as e:
            print(f"[WARN] Batch size probe failed ({e}), defaulting to 16", flush=True)
            return 16

    @staticmethod
    def _decode_item(item: ViewItem):
        """Convert ViewItem RGB numpy array → uint8 HWC CPU tensor.

        uint8 stays uint8 until it reaches the GPU: the float32 conversion
        used to happen on the CPU, making every PCIe transfer 4× larger
        (1.2MB vs 311KB per 322×322 view). /255 + normalize now run on GPU.
        """
        try:
            t = torch.from_numpy(np.ascontiguousarray(item.view_data))
            if t.ndim != 3 or t.shape[-1] != 3 or t.dtype != torch.uint8:
                raise ValueError(f"unexpected view tensor {t.dtype} {tuple(t.shape)}")
            return t
        except Exception as e:
            print(f"[WARN] Decode failed panoid={item.panoid}: {type(e).__name__}: {e}", flush=True)
            return None

    def start_decode(self, items: List[ViewItem]) -> list:
        """Non-blocking: submit numpy→tensor conversion for all items; return list of futures."""
        return [self.executor.submit(self._decode_item, item) for item in items]

    def _staging_buffer(self, n: int, h: int, w: int):
        """Reusable pinned uint8 staging buffer. cudaHostAlloc is expensive
        and serializes with the GPU — the old per-batch pin_memory() call
        allocated ~160MB of fresh pinned memory every batch."""
        buf = getattr(self, '_pin_buf', None)
        if (buf is None or buf.shape[0] < n
                or buf.shape[1] != h or buf.shape[2] != w):
            cap = max(n, self.batch_size)
            buf = torch.empty((cap, h, w, 3), dtype=torch.uint8,
                              pin_memory=True)
            self._pin_buf = buf
        return buf

    def _run_inference(self, items: List[ViewItem], valid_tensors: list, valid_indices: list):
        """GPU inference with pinned uint8 staging, fp16 autocast, and OOM auto-retry."""
        try:
            n = len(valid_tensors)
            h, w = valid_tensors[0].shape[0], valid_tensors[0].shape[1]
            try:
                buf = self._staging_buffer(n, h, w)
                for i, t in enumerate(valid_tensors):
                    buf[i].copy_(t)
                staged = buf[:n]
            except Exception:
                # Mixed shapes or pinned-alloc failure — stack copies instead.
                staged = torch.stack(valid_tensors).pin_memory()
            # uint8 HWC over PCIe (4× less traffic), float + scale on GPU.
            images = staged.to(self.device, non_blocking=True)
            images = images.permute(0, 3, 1, 2).float().div_(255.0)
            if images.shape[-2:] != (322, 322):
                images = torch.nn.functional.interpolate(
                    images, size=(322, 322), mode='bilinear', align_corners=False
                )
            images = (images - self.mean) / self.std

            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.float16):
                    feats = self.model(images)
            del images

            feats_np = feats.float().cpu().numpy()
            metadata_batch = [
                {'panoid': items[i].panoid, 'lat': items[i].lat, 'lng': items[i].lng}
                for i in valid_indices
            ]
            return feats_np, metadata_batch, valid_indices

        except torch.cuda.OutOfMemoryError:
            half = len(valid_tensors) // 2
            torch.cuda.empty_cache()
            if half == 0:
                print("[WARN] OOM on single image — skipping", flush=True)
                return None, [], []

            new_bs = max(8, half)
            print(f"[WARN] OOM on batch={len(valid_tensors)} → retrying as 2×{half}, "
                  f"shrinking batch_size {self.batch_size} → {new_bs}", flush=True)
            self.batch_size = new_bs

            f1, m1, vi1 = self._run_inference(items, valid_tensors[:half], valid_indices[:half])
            f2, m2, vi2 = self._run_inference(items, valid_tensors[half:], valid_indices[half:])

            if f1 is None and f2 is None:
                return None, [], []
            if f1 is None:
                return f2, m2, vi2
            if f2 is None:
                return f1, m1, vi1
            return np.concatenate([f1, f2], axis=0), m1 + m2, vi1 + vi2

    def infer_prefetched(self, items: List[ViewItem], futures: list):
        """Block on decode futures collected by start_decode(), then run GPU inference."""
        tensors_or_none = [f.result() for f in futures]
        valid_indices = [i for i, t in enumerate(tensors_or_none) if t is not None]
        valid_tensors = [tensors_or_none[i] for i in valid_indices]

        failures = len(items) - len(valid_tensors)
        if failures:
            print(f"[WARN] {failures}/{len(items)} images failed to decode in batch", flush=True)
        if not valid_tensors:
            return None, [], []
        return self._run_inference(items, valid_tensors, valid_indices)

    def extract_batch(self, items: List[ViewItem]):
        """Synchronous decode + GPU inference (no prefetch). Kept for compatibility."""
        tensors_or_none = list(self.executor.map(self._decode_item, items))
        valid_indices = [i for i, t in enumerate(tensors_or_none) if t is not None]
        valid_tensors = [tensors_or_none[i] for i in valid_indices]

        failures = len(items) - len(valid_tensors)
        if failures:
            print(f"[WARN] {failures}/{len(items)} images failed to decode in batch", flush=True)
        if not valid_tensors:
            return None, [], []
        return self._run_inference(items, valid_tensors, valid_indices)


# ═══════════════════════════════════════════════════════════════════════════════
# CSV Loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_csv(csv_path: str) -> Tuple[List[dict], Dict[str, Dict]]:
    records = []
    metadata = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = ',' if sample.count(',') >= sample.count(';') else ';'
        reader = csv.DictReader(f, delimiter=delimiter)

        col_map = {}
        if reader.fieldnames:
            for field in reader.fieldnames:
                clean = field.lower().strip().replace('_', '').replace('-', '')
                if clean == 'panoid':
                    col_map['panoid'] = field
                elif clean in ('lat', 'latitude'):
                    col_map['lat'] = field
                elif clean in ('lon', 'lng', 'longitude'):
                    col_map['lon'] = field
                elif clean in ('headingdeg', 'heading', 'yaw'):
                    col_map['heading'] = field

        if 'panoid' not in col_map:
            print(f"[ERROR] No panoid column in CSV. Columns: {reader.fieldnames}")
            sys.exit(1)

        for row in reader:
            panoid = row.get(col_map['panoid'], '').strip()
            if not panoid:
                continue
            record = {'panoid': panoid}
            if 'heading' in col_map and row.get(col_map['heading']):
                try:
                    record['heading_deg'] = float(row[col_map['heading']])
                except ValueError:
                    pass
            records.append(record)
            if 'lat' in col_map and 'lon' in col_map:
                try:
                    lat = float(row.get(col_map['lat'], '').strip())
                    lon = float(row.get(col_map['lon'], '').strip())
                    metadata[panoid] = {'lat': round(lat, 5), 'lng': round(lon, 5)}
                except (ValueError, AttributeError):
                    pass
    return records, metadata


# ═══════════════════════════════════════════════════════════════════════════════
# Async Downloader
# ═══════════════════════════════════════════════════════════════════════════════

def _blocking_put_items(item_queue, items, stats):
    """Runs in an executor thread: blocking-put items onto the stdlib queue.

    Checks the stop flag each second so shutdown can't deadlock on a full
    queue whose consumer (extraction loop) has already exited.
    Returns False if aborted by the stop flag.
    """
    for item in items:
        while True:
            if stats.get('stop_requested'):
                return False
            try:
                item_queue.put(item, timeout=1.0)
                break
            except queue.Full:
                continue
    return True


async def _queue_put_items(item_queue, items, stats):
    """Bridge asyncio → blocking stdlib queue without freezing the event loop.

    A direct item_queue.put(timeout=...) on the event loop blocks EVERY
    in-flight request whenever the queue is full — the whole downloader
    freezes until the GPU drains a batch. Run the blocking put in the
    default executor instead.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _blocking_put_items, item_queue, items, stats)


async def _download_single_pano(session, record, sem, config, item_queue, metadata, stats, shared_state):
    """Fetch all `num_views` server-rendered perspective views for one pano.

    Strict-alignment policy: if any single view fails (4xx, decode error,
    black-placeholder) after retries, the entire pano is dropped — partial
    panos would create row-count drift between NPY and JSONL across chunks.

    The async-pano-level retry covers transient network exceptions; per-view
    HTTP retries (5xx/429 + Retry-After) live inside fetch_thumbnail_view.
    """
    panoid_str = record['panoid']
    heading_deg = record.get('heading_deg')

    num_views = config['num_views']
    fov = config['view_fov']
    resolution = config['view_resolution']
    pitch = config.get('view_pitch', 0.0)
    offset = config.get('view_offset', 0.0)

    # Yaws: evenly spaced; aligned to street heading if CSV provided one,
    # otherwise oriented to true north.
    base_yaw = (heading_deg if heading_deg is not None else 0.0) + offset
    step = 360.0 / num_views
    yaws = [(base_yaw + step * i) % 360.0 for i in range(num_views)]

    # Jitter each pano's start so the sem doesn't release in uniform bursts.
    jitter_max = config.get('pano_jitter_max', 0.0)
    if jitter_max > 0:
        await asyncio.sleep(random.uniform(0, jitter_max))

    retries = 3
    # Accumulate views across attempts so retries only refetch the yaws
    # that actually failed — refetching all num_views on every retry
    # multiplies per-IP request cost for no benefit.
    view_results = [None] * num_views
    for attempt in range(1, retries + 1):
        # Short-circuit if the probe flagged this IP as blocked or the
        # extraction loop requested shutdown — don't waste requests.
        if stats.get('ip_blocked_403') or stats.get('stop_requested'):
            return
        try:
            async with sem:
                # Fan out parallel thumbnail fetches for the still-missing
                # yaws only. Each call handles its own 5xx/429 retry; we
                # get back numpy RGB arrays or None.
                missing_idx = [i for i, v in enumerate(view_results)
                               if v is None]
                tasks = [
                    fetch_thumbnail_view(
                        session, panoid_str, yaws[i], pitch, fov,
                        resolution, resolution, stats=stats,
                    )
                    for i in missing_idx
                ]
                fetched = await asyncio.gather(*tasks)
                for i, arr in zip(missing_idx, fetched):
                    view_results[i] = arr

            # Re-check 403 flag — only the probe machinery sets it now
            # (a single pano's persistent 403 no longer flips it).
            if stats.get('ip_blocked_403'):
                _throttle.maybe(
                    'ip_blocked',
                    f"[DL] HTTP 403 from streetviewpixels-pa — IP blocked, "
                    f"winding down (http_403={stats.get('http_403', 0)})"
                )
                return

            missing = sum(1 for v in view_results if v is None)
            if missing:
                if attempt < retries:
                    # Back off OUTSIDE the semaphore so a sleeping pano
                    # doesn't hold one of the per-IP download slots.
                    await asyncio.sleep(2 ** attempt)
                    continue
                # All-N-or-drop gate: partial panos would create row-count
                # drift between NPY and JSONL across chunks.
                stats['dl_fail'] += 1
                stats.setdefault('fail_partial_views', 0)
                stats['fail_partial_views'] += 1
                _throttle.maybe(
                    'partial_views',
                    f"[DL] partial_views panoid={panoid_str} "
                    f"missing={missing}/{num_views} "
                    f"(total fail_partial_views={stats['fail_partial_views']}, "
                    f"black={stats.get('black_views', 0)}, "
                    f"suspect_403={stats.get('suspect_403', 0)})"
                )
                shared_state.log_failure(
                    panoid_str, f"partial_views:{missing}/{num_views}"
                )
                return

            # All N views good — push as ViewItems sharing the same
            # panoid + lat/lng. SharedState.write_batch keeps memmap
            # rows and JSONL lines aligned under a lock. Each item carries
            # its chunk's sink so overlapped chunks can share the queue.
            meta = metadata.get(panoid_str, {'lat': 0.0, 'lng': 0.0})
            items = [
                ViewItem(panoid_str, view_arr, meta['lat'], meta['lng'],
                         sink=shared_state)
                for view_arr in view_results
            ]
            if not await _queue_put_items(item_queue, items, stats):
                return  # shutdown requested while queue was full

            stats['dl_ok'] += 1
            stats['views_produced'] += num_views
            return

        except Exception as e:
            if attempt < retries:
                await asyncio.sleep(2 ** attempt)
            else:
                stats['dl_fail'] += 1
                stats.setdefault('fail_exc', 0)
                stats['fail_exc'] += 1
                _throttle.maybe(
                    'pano_exc',
                    f"[DL] pano exception {type(e).__name__}: {str(e)[:200]} "
                    f"(total fail_exc={stats['fail_exc']})"
                )
                shared_state.log_failure(panoid_str, f"exception: {e}")

async def _run_downloader(records, config, item_queue, metadata, stats, shared_state):
    print(f"[DL] entered records={len(records)} max_threads={config.get('max_threads')}",
          flush=True)
    sem = asyncio.Semaphore(config['max_threads'])

    try:
        # curl_cffi AsyncSession with Chrome 142 impersonation: full Chrome
        # network signature (TLS JA3/JA4, HTTP/2 + HTTP/3 + ALPN, every
        # Sec-Ch-Ua/Sec-Fetch header, brotli-capable Accept-Encoding) plus a
        # built-in cookie jar that survives the maps.google.com warmup visit
        # below. max_clients caps the underlying libcurl multi-handle's pool;
        # 400 is high enough to feed the sem semaphore-controlled fan-out
        # without thrashing connections.
        async with AsyncSession(
            impersonate='chrome142',
            max_clients=400,
            timeout=15,
        ) as session:

            # ── Warmup: grab NID/CONSENT from maps.google.com before tiling ──
            # Real browsers always have these cookies by the time they touch
            # cbk*. Going straight to cbk* with zero cookies is a small bot
            # tell. Cost: 1 extra GET per chunk.
            try:
                wr = await session.get(
                    'https://www.google.com/maps/',
                    allow_redirects=True,
                )
                cookie_names = sorted(list(session.cookies.keys()))
                print(f"[DL][WARMUP] google.com/maps → status={wr.status_code} "
                      f"cookies={cookie_names}", flush=True)
            except Exception as e:
                print(f"[DL][WARMUP] maps.google.com FAILED: "
                      f"{type(e).__name__}: {e} — continuing without cookies",
                      flush=True)

            # ── Connectivity probe: fetch 1 thumbnail from the first real pano ──
            if records:
                probe_pid = records[0].get('panoid', '')
                probe_resolution = config.get('view_resolution', 322)
                probe_fov = int(round(config.get('view_fov', 70.0)))
                probe_url = (
                    f"https://streetviewpixels-pa.googleapis.com/v1/thumbnail"
                    f"?panoid={probe_pid}&cb_client=maps_sv.tactile.gps"
                    f"&w={probe_resolution}&h={probe_resolution}"
                    f"&yaw=0.00&pitch=0.00&thumbfov={probe_fov}"
                )
                try:
                    r = await session.get(probe_url)
                    body = r.content
                    print(f"[DL][PROBE] GET thumbnail panoid={probe_pid} → "
                          f"status={r.status_code} len={len(body)} "
                          f"ct={r.headers.get('content-type','')} "
                          f"cl={r.headers.get('content-length','')}",
                          flush=True)
                    stats['probe_status'] = r.status_code
                    if r.status_code == 403:
                        # Google sorry.google.com page = IP is blocked.
                        # Flag so the outer loop can unclaim + self-destruct
                        # instead of burning API calls and poisoning the queue.
                        stats['ip_blocked_403'] = True
                        print(f"[DL][PROBE] IP appears 403-blocked by Google. "
                              f"Will unclaim chunk + self-destruct.", flush=True)
                except Exception as e:
                    print(f"[DL][PROBE] GET thumbnail FAILED: "
                          f"{type(e).__name__}: {e}", flush=True)
                    stats['probe_status'] = -1

            # Short-circuit the whole gather if probe says IP is 403-blocked —
            # no point firing 1000 requests that are all going to return the
            # sorry.google.com page.
            if stats.get('ip_blocked_403'):
                print("[DL] Skipping pano dispatch — probe detected 403 block.",
                      flush=True)
                return

            # ── Background periodic re-probe to detect mid-chunk blocks ──
            # A chunk can start fine (probe=200) but hit the block partway
            # through. Re-probe every 15s in the background so we catch the
            # transition quickly and abandon the chunk cleanly instead of
            # writing half a feature file.
            async def _reprobe_loop():
                while not stats.get('ip_blocked_403') and not stats.get('dl_done'):
                    await asyncio.sleep(15)
                    if stats.get('dl_done'):
                        return
                    try:
                        rr = await session.get(probe_url)
                        _ = rr.content
                        if rr.status_code == 403:
                            stats['ip_blocked_403'] = True
                            print(f"[DL][REPROBE] Mid-chunk 403 detected "
                                  f"(dl_ok={stats['dl_ok']}, "
                                  f"dl_fail={stats['dl_fail']}) — "
                                  f"abandoning chunk.", flush=True)
                            return
                    except Exception as e:
                        # Transient — don't flip the flag on a single failure
                        print(f"[DL][REPROBE] transient error: "
                              f"{type(e).__name__}: {e}", flush=True)

            reprobe_task = asyncio.create_task(_reprobe_loop())

            CHUNK = 5000
            for i in range(0, len(records), CHUNK):
                if stats.get('ip_blocked_403'):
                    print("[DL] Mid-chunk 403 — skipping remaining panos.",
                          flush=True)
                    break
                if stats.get('stop_requested'):
                    print("[DL] Stop requested — skipping remaining panos.",
                          flush=True)
                    break
                chunk = records[i:i + CHUNK]
                print(f"[DL] Dispatching {len(chunk)} pano tasks "
                      f"(offset={i}/{len(records)})", flush=True)
                tasks = [
                    _download_single_pano(session, rec, sem, config,
                                          item_queue, metadata, stats, shared_state)
                    for rec in chunk
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                # Surface any per-task exceptions that would otherwise be silent
                exc_counts = {}
                for r in results:
                    if isinstance(r, BaseException):
                        k = type(r).__name__
                        exc_counts[k] = exc_counts.get(k, 0) + 1
                if exc_counts:
                    print(f"[DL] Silent per-task exceptions: {exc_counts} "
                          f"(dl_ok={stats['dl_ok']}, dl_fail={stats['dl_fail']})",
                          flush=True)

            reprobe_task.cancel()
            try:
                await reprobe_task
            except (asyncio.CancelledError, Exception):
                pass
    finally:
        item_queue.put(_SENTINEL)
        stats['dl_done'] = True
        print(f"[DL] exit dl_ok={stats['dl_ok']} dl_fail={stats['dl_fail']} "
              f"views={stats['views_produced']} "
              f"partial={stats.get('fail_partial_views', 0)} "
              f"black={stats.get('black_views', 0)} "
              f"http403={stats.get('http_403', 0)} "
              f"suspect403={stats.get('suspect_403', 0)} "
              f"fetch_exc={stats.get('fetch_exc', 0)} "
              f"exc={stats.get('fail_exc', 0)}", flush=True)


def downloader_thread(records, config, item_queue, metadata, stats, shared_state):
    try:
        asyncio.run(_run_downloader(records, config, item_queue, metadata, stats, shared_state))
    except BaseException as e:
        import traceback
        print(f"[DL][FATAL] downloader_thread crashed: {type(e).__name__}: {e}",
              flush=True)
        traceback.print_exc()
        # Ensure main loop unblocks even after a crash
        try:
            item_queue.put(_SENTINEL, timeout=5.0)
        except Exception:
            pass
        stats['dl_done'] = True


# ═══════════════════════════════════════════════════════════════════════════════
# Disk Space Management
# ═══════════════════════════════════════════════════════════════════════════════

def get_free_gb(path: str = '/') -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)

def wait_for_disk_space(path: str = '/', min_gb: float = MIN_FREE_GB,
                        max_wait_sec: int = 600):
    """Block until min_gb is free, or raise after max_wait_sec.

    Bounded so a stuck cleanup or runaway log file can't hang a worker
    forever during a multi-day scrape — the outer chunk handler treats the
    raise as a fail_task, which lets another worker retry the chunk.
    """
    waited = 0
    poll = 60
    while get_free_gb(path) < min_gb:
        if waited >= max_wait_sec:
            raise RuntimeError(
                f"Disk-space wait exceeded {max_wait_sec}s "
                f"({get_free_gb(path):.1f}GB free, need {min_gb}GB at {path}). "
                f"Aborting chunk so it can be retried elsewhere."
            )
        print(f"[WARN] Only {get_free_gb(path):.1f}GB free, waiting for space "
              f"(need {min_gb}GB, waited {waited}s/{max_wait_sec}s)...")
        time.sleep(poll)
        waited += poll


# ═══════════════════════════════════════════════════════════════════════════════
# Log Capture & Upload
# ═══════════════════════════════════════════════════════════════════════════════

LOG_FILE = f"/tmp/worker_{INSTANCE_ID or 'unknown'}.log"


class TeeWriter:
    """Writes to both the original stream and a log file."""
    def __init__(self, original, log_file_handle):
        self.original = original
        self.log_file = log_file_handle

    def write(self, data):
        self.original.write(data)
        try:
            self.log_file.write(data)
        except Exception:
            pass

    def flush(self):
        self.original.flush()
        try:
            self.log_file.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self.original, name)


def _start_log_capture():
    """Tee stdout and stderr to a log file."""
    try:
        fh = open(LOG_FILE, "w", encoding="utf-8", errors="replace")
        sys.stdout = TeeWriter(sys.__stdout__, fh)
        sys.stderr = TeeWriter(sys.__stderr__, fh)
        return fh
    except Exception as e:
        print(f"[WARN] Could not start log capture: {e}")
        return None


def upload_logs_to_r2():
    """Upload the captured log file to R2."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()

        r2 = R2Client()
        log_key = f"Logs/{FEATURES_BUCKET_PREFIX}/worker_{INSTANCE_ID}.log"
        r2.upload_file(LOG_FILE, log_key)
        print(f"[INFO] Uploaded logs to R2: {log_key}")
    except Exception as e:
        print(f"[WARN] Failed to upload logs to R2: {e}")


MAX_LOG_BYTES = 200 * 1024 * 1024  # truncate tee'd log past this size


def _truncate_log_if_huge():
    """Upload-then-truncate the tee'd log once it exceeds MAX_LOG_BYTES.

    A multi-day scrape's stdout can otherwise fill the disk. Called from
    the 30s heartbeat block in the extraction loop.
    """
    try:
        if os.path.getsize(LOG_FILE) < MAX_LOG_BYTES:
            return
    except OSError:
        return
    print(f"[INFO] Log file exceeded {MAX_LOG_BYTES // (1024**2)}MB — "
          f"uploading to R2 then truncating", flush=True)
    try:
        upload_logs_to_r2()
    except Exception:
        pass
    try:
        old_fh = sys.stdout.log_file if isinstance(sys.stdout, TeeWriter) else None
        new_fh = open(LOG_FILE, "w", encoding="utf-8", errors="replace")
        for stream in (sys.stdout, sys.stderr):
            if isinstance(stream, TeeWriter):
                stream.log_file = new_fh
        if old_fh is not None:
            try:
                old_fh.close()
            except Exception:
                pass
        print("[INFO] Log file truncated after upload", flush=True)
    except Exception as e:
        print(f"[WARN] Log truncate failed: {e}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Destruct
# ═══════════════════════════════════════════════════════════════════════════════

def self_destruct():
    """Destroy this Vast.ai instance — retries forever until the instance is gone."""
    instance_id = INSTANCE_ID

    if not instance_id:
        # Fallback: try to detect from vastai CLI
        if VAST_API_KEY:
            try:
                result = subprocess.run(
                    ["vastai", "--api-key", VAST_API_KEY, "show", "instances", "--raw"],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0 and result.stdout.strip():
                    instances = json.loads(result.stdout.strip())
                    if instances and len(instances) == 1:
                        instance_id = str(instances[0].get('id', ''))
            except Exception as e:
                print(f"[WARN] Instance ID auto-detect failed: {e}")

    # Missing creds: sleeping forever would burn money on a paid GPU.
    # Upload logs and exit(1) so the entrypoint's failure path (retry →
    # API self-destruct) takes over.
    if not instance_id:
        print("[ERROR] Cannot self-destruct: unable to determine INSTANCE_ID — "
              "uploading logs and exiting(1) for entrypoint to handle", flush=True)
        try:
            upload_logs_to_r2()
        except Exception:
            pass
        os._exit(1)
    if not VAST_API_KEY:
        print("[ERROR] Cannot self-destruct: VAST_API_KEY not set — "
              "uploading logs and exiting(1) for entrypoint to handle", flush=True)
        try:
            upload_logs_to_r2()
        except Exception:
            pass
        os._exit(1)

    # Use the plural form `destroy instances <id>` — singular prompts for
    # interactive [y/N] confirmation, reports exit=0 on empty stdin but leaves
    # the instance alive. Plural form takes a list and does not prompt.
    # Also pipe an empty stdin defensively and auto-confirm via `yes`-style
    # input in case the plural form ever starts prompting too.
    cmd = ["vastai", "--api-key", VAST_API_KEY, "destroy", "instances", str(instance_id)]
    attempt = 0
    while True:
        attempt += 1
        try:
            print(f"[INFO] Self-destruct attempt {attempt} for instance {instance_id}...")
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
                input="y\n",
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            print(f"[INFO] Self-destruct response: exit={result.returncode} stdout='{stdout}' stderr='{stderr}'")

            # Only treat as successful if stdout matches the expected
            # "destroying instance N." pattern — exit=0 can be lying when the
            # CLI aborts a prompt.
            if result.returncode == 0 and f"destroying instance {instance_id}" in stdout.lower():
                print(f"[INFO] Instance {instance_id} destroyed successfully.")
                return
            print(f"[WARN] Self-destruct attempt {attempt} did not confirm destruction — retrying in 30s")
        except Exception as e:
            print(f"[WARN] Self-destruct attempt {attempt} exception: {e} — retrying in 30s")
        time.sleep(30)


# ═══════════════════════════════════════════════════════════════════════════════
# Upload with Retry
# ═══════════════════════════════════════════════════════════════════════════════

MAX_EXTENDED_RETRIES = 30
# Total wall-clock budget for one file's upload (initial + extended retries).
# Without it a dead network parks the worker for ~31 min per file; with it
# the failure surfaces to the caller, which unclaims the chunk so a
# (possibly healthier) worker retries.
UPLOAD_MAX_WALL_S = int(os.environ.get('UPLOAD_MAX_WALL_S', '600'))


def upload_with_retry(r2, local_path, bucket_key, label="FILE", max_attempts=5):
    """Upload a file to R2 with exponential backoff, extended retry, and a
    wall-clock deadline (UPLOAD_MAX_WALL_S). Returns False on deadline."""
    file_size = os.path.getsize(local_path)
    deadline = time.time() + UPLOAD_MAX_WALL_S

    def _already_landed():
        # Upload may succeed server-side but the connection resets before we
        # see the response. Size must match — a stale/partial object on the
        # key must NOT pass as success.
        try:
            if r2.file_exists(bucket_key, expected_size=file_size):
                print(f"[INFO] File already on R2 with matching size "
                      f"({file_size:,} bytes): {bucket_key}")
                return True
        except Exception:
            pass
        return False

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            if _already_landed():
                return True
            r2.abort_pending_multipart(bucket_key)
        print(f"[INFO] Uploading {label} ({file_size / (1024**2):.1f} MB), attempt {attempt}/{max_attempts}...")
        if r2.upload_file(local_path, bucket_key, max_retries=1):
            return True
        print(f"[WARN] Upload attempt {attempt}/{max_attempts} failed for {bucket_key}")
        if time.time() >= deadline:
            print(f"[ERROR] Upload deadline ({UPLOAD_MAX_WALL_S}s) exceeded for "
                  f"{bucket_key} — giving up so the chunk can be retried elsewhere")
            r2.abort_pending_multipart(bucket_key)
            return False
        if attempt < max_attempts:
            wait = min(2 ** attempt, 120)
            time.sleep(wait)

    # Extended retry with client reset
    for retry_count in range(1, MAX_EXTENDED_RETRIES + 1):
        if _already_landed():
            return True
        if time.time() >= deadline:
            print(f"[ERROR] Upload deadline ({UPLOAD_MAX_WALL_S}s) exceeded for "
                  f"{bucket_key} after {retry_count - 1} extended retries — giving up")
            break
        print(f"[WARN] Extended retry #{retry_count}/{MAX_EXTENDED_RETRIES} for {bucket_key}")
        r2.reset_client()
        r2.abort_pending_multipart(bucket_key)
        time.sleep(60)
        if r2.upload_file(local_path, bucket_key, max_retries=1):
            return True

    # Final existence check before declaring permanent failure
    if _already_landed():
        return True

    r2.abort_pending_multipart(bucket_key)
    print(f"[ERROR] Upload permanently failed for {bucket_key}")
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Process a Single Chunk
# ═══════════════════════════════════════════════════════════════════════════════

# Cross-chunk overlap: when the current chunk has fewer than this fraction of
# its panos still downloading (its straggler tail), the prefetched next
# chunk's downloader is started into the SAME item queue so the GPU and the
# network never sit idle at chunk boundaries. CHUNK_OVERLAP=0 disables.
CHUNK_OVERLAP = os.environ.get('CHUNK_OVERLAP', '1') == '1'
OVERLAP_TAIL_FRACTION = float(os.environ.get('OVERLAP_TAIL_FRACTION', '0.02'))


class _ChunkRun:
    """Live extraction state for one chunk: its memmap sink, stats dict and
    downloader thread. Created either at the top of process_chunk (normal) or
    mid-extraction of the PREVIOUS chunk (overlap), in which case it is handed
    off to the next process_chunk call with its downloader already running."""
    __slots__ = ('chunk_id', 'redis_cid', 'local_csv', 'total_records',
                 'features_file', 'metadata_file', 'failed_file', 'out_base',
                 'sink', 'stats', 'dl_thread', 'total_views_est', 'bmeta',
                 'pending_batch', 'pending_futures')

    def is_complete(self) -> bool:
        """All downloader-produced views are written or dropped, and the
        downloader can produce no more. Order matters: check the thread
        FIRST so views_produced is final when the counters are read."""
        if self.dl_thread.is_alive():
            return False
        return (self.sink.write_idx + self.sink.dropped
                >= self.stats.get('views_produced', 0))


def _start_chunk_run(work_dir: Path, item_queue, chunk_id: str, redis_cid: str,
                     local_csv: str, records, metadata_map, city: str,
                     bmeta=None, total_chunks=None) -> _ChunkRun:
    """Allocate output files + memmap and start the downloader for a chunk.

    total_chunks: the chunk's CITY total for output naming. Must be passed
    explicitly for overlap runs — the global TOTAL_CHUNKS still holds the
    CURRENT chunk's city total at overlap-start time (batch mode)."""
    total_records = len(records)
    views_per_pano = HARDCODED_CONFIG['num_views']
    total_views_est = total_records * views_per_pano

    if total_chunks is None:
        total_chunks = TOTAL_CHUNKS
    out_base = f"{city}_{_chunk_num(chunk_id)}.{total_chunks}"
    features_file = str(work_dir / f"{out_base}.npy")
    metadata_file = str(work_dir / f"Metadata_{out_base}.jsonl")
    failed_file = str(work_dir / f"failed_{chunk_id}.jsonl")

    features_memmap = np.lib.format.open_memmap(
        features_file, mode='w+', dtype='float32',
        shape=(total_views_est, 8448)
    )
    sink = SharedState(features_memmap, metadata_file, failed_file, start_idx=0)

    dl_config = dict(HARDCODED_CONFIG)
    stats = {'dl_ok': 0, 'dl_fail': 0, 'ext_ok': 0, 'views_produced': 0,
             'dl_done': False}
    # Back-reference so the extraction loop can credit ext_ok to the right
    # chunk when overlapped batches mix two chunks' views.
    sink.owner_stats = stats
    dl_thread = threading.Thread(
        target=downloader_thread,
        args=(records, dl_config, item_queue, metadata_map, stats, sink)
    )
    dl_thread.start()

    run = _ChunkRun()
    run.chunk_id = chunk_id
    run.redis_cid = redis_cid
    run.local_csv = local_csv
    run.total_records = total_records
    run.features_file = features_file
    run.metadata_file = metadata_file
    run.failed_file = failed_file
    run.out_base = out_base
    run.sink = sink
    run.stats = stats
    run.dl_thread = dl_thread
    run.total_views_est = total_views_est
    run.bmeta = bmeta
    run.pending_batch = []
    run.pending_futures = None
    return run


def _abort_run(run: '_ChunkRun', tq, item_queue, work_dir: Path,
               reason: str, hb=None):
    """Wind down an overlap run that can't continue (current chunk raised,
    IP block, shutdown): stop its downloader, unclaim its chunk, delete its
    partial files. Never raises."""
    try:
        run.stats['stop_requested'] = True
        deadline = time.time() + 60
        while run.dl_thread.is_alive() and time.time() < deadline:
            try:
                while True:
                    item_queue.get_nowait()
            except queue.Empty:
                pass
            run.dl_thread.join(timeout=1.0)
    except Exception:
        pass
    try:
        run.sink.close()
    except Exception:
        pass
    try:
        tq.unclaim_task(REGION, run.redis_cid, INSTANCE_ID,
                        reason=reason, back=True)
    except Exception as e:
        print(f"[OVERLAP] unclaim of {run.redis_cid} failed: {e}", flush=True)
    if hb is not None:
        try:
            hb.unregister(run.redis_cid)
        except Exception:
            pass
    _cleanup_chunk_files(work_dir, run.chunk_id, run.local_csv,
                         out_base=run.out_base)


def _try_start_overlap(overlap_pf, r2, tq, work_dir: Path, item_queue, hb):
    """Consume the prefetched next chunk and start its downloader into the
    shared queue. Returns the new _ChunkRun, or None with the prefetch entry
    left intact for the main loop (empty chunk, low disk, errors) — except
    the already-on-R2 case, which is completed and consumed here."""
    pf = overlap_pf[0]
    try:
        next_id, pf_csv, pf_records, pf_meta = pf[0], pf[1], pf[2], pf[3]
        pf_bmeta = pf[4] if len(pf) > 4 else None
    except Exception:
        return None
    if not pf_records:
        return None  # empty chunk — cheap for the main loop to complete

    if pf_bmeta:
        n_city = pf_bmeta['city_name']
        file_cid = f"chunk_{pf_bmeta['chunk_num']:04d}"
        feat_prefix = pf_bmeta['features_prefix']
        n_total = pf_bmeta['city_total']
    else:
        n_city = CITY_NAME
        file_cid = next_id
        feat_prefix = FEATURES_BUCKET_PREFIX
        n_total = TOTAL_CHUNKS

    # Same skip check the main loop would do — never re-extract a chunk
    # whose NPY + metadata already landed on R2.
    out_b = f"{n_city}_{_chunk_num(file_cid)}.{n_total}"
    try:
        if (r2.object_size(f"{feat_prefix}/{out_b}.npy") > 0
                and r2.object_size(f"{feat_prefix}/Metadata_{out_b}.jsonl") > 0):
            print(f"[OVERLAP] {next_id} output already on R2 — completing "
                  f"instead of overlapping", flush=True)
            overlap_pf[0] = None
            try:
                tq.complete_task(REGION, next_id, INSTANCE_ID)
            except Exception:
                pass
            if hb is not None:
                hb.unregister(next_id)
            try:
                os.remove(pf_csv)
            except OSError:
                pass
            return None
    except Exception:
        pass  # can't check — proceed with the overlap

    # A second ~270MB memmap comes alive — make sure the disk can take it
    # without blocking the extraction thread for long.
    try:
        wait_for_disk_space(str(work_dir), MIN_FREE_GB, max_wait_sec=5)
    except Exception:
        print("[OVERLAP] low disk — leaving next chunk to the main loop",
              flush=True)
        return None

    print(f"[OVERLAP] Starting next chunk {next_id} ({len(pf_records)} panos) "
          f"during current chunk's straggler tail", flush=True)
    try:
        new_run = _start_chunk_run(work_dir, item_queue, file_cid, next_id,
                                   pf_csv, pf_records, pf_meta, n_city,
                                   bmeta=pf_bmeta, total_chunks=n_total)
    except Exception as e:
        # Setup failed (e.g. memmap allocation) — leave the prefetch entry
        # for the main loop's normal path rather than losing the claim.
        print(f"[OVERLAP] start failed ({type(e).__name__}: {e}) — "
              f"falling back to sequential processing", flush=True)
        return None
    overlap_pf[0] = None  # consumed — main loop must not double-process it
    return new_run


def _truncate_npy(path: str, final_count: int):
    """Shrink an NPY to its first `final_count` rows IN PLACE by patching the
    header's shape and truncating the file — replaces the old read-copy-
    rewrite which pushed ~260MB through RAM and doubled disk I/O per chunk.
    Falls back to the copy method if the header isn't exactly as
    np.lib.format wrote it."""
    import ast
    try:
        with open(path, 'r+b') as f:
            magic = f.read(6)
            if magic != b'\x93NUMPY':
                raise ValueError("bad NPY magic")
            major, _minor = f.read(2)
            if major == 1:
                hlen = int.from_bytes(f.read(2), 'little')
                hstart = 10
            else:
                hlen = int.from_bytes(f.read(4), 'little')
                hstart = 12
            header = f.read(hlen).decode('latin1')
            d = ast.literal_eval(header)
            old_shape = tuple(d['shape'])
            if final_count > old_shape[0]:
                raise ValueError("final_count larger than allocated shape")
            new_shape = (final_count,) + old_shape[1:]
            new_header = (
                "{'descr': %r, 'fortran_order': %r, 'shape': %r, }"
                % (d['descr'], d['fortran_order'], new_shape)
            )
            # Same padding scheme as np.lib.format: spaces, trailing \n,
            # total header length unchanged (the shape only shrinks).
            if len(new_header) + 1 > hlen:
                raise ValueError("new header does not fit in existing header")
            padded = new_header + ' ' * (hlen - 1 - len(new_header)) + '\n'
            f.seek(hstart)
            f.write(padded.encode('latin1'))
        row_bytes = int(np.dtype(d['descr']).itemsize)
        for dim in old_shape[1:]:
            row_bytes *= int(dim)
        os.truncate(path, hstart + hlen + final_count * row_bytes)
        check = np.load(path, mmap_mode='r')
        if check.shape != new_shape:
            raise ValueError(f"post-truncate shape {check.shape} != {new_shape}")
        del check
    except Exception as e:
        print(f"[WARN] In-place NPY truncate failed ({type(e).__name__}: {e}) "
              f"— falling back to copy", flush=True)
        mm = np.lib.format.open_memmap(path, mode='r+')
        truncated = mm[:final_count].copy()
        del mm
        np.save(path, truncated)
        del truncated


def process_chunk(r2, tq: TaskQueue, extractor, chunk_id: str, work_dir: Path,
                  preloaded=None, chunks_done_so_far: int = 0, redis_chunk_id: str = None,
                  gpu_pool: 'GpuWorkerPool' = None, item_queue=None,
                  overlap_pf=None, handoff: '_ChunkRun' = None,
                  handoff_out=None, hb=None):
    """
    Extract features for a chunk. Returns (features_file, metadata_file, local_csv)
    for async upload, or None if chunk was empty.

    preloaded: optional (local_csv, records, metadata_map) to skip CSV download.
    redis_chunk_id: global chunk ID for Redis ops (batch mode). Defaults to chunk_id.
    gpu_pool: when provided (n_gpus > 1), partitions records across the pool's
        child processes and stitches partial NPYs. When None, runs the legacy
        single-process extraction loop. `extractor` is ignored in pool mode.
    item_queue: shared download→extract queue (single-GPU path). Must be the
        SAME object across calls for cross-chunk overlap; created locally if
        omitted (overlap disabled in that case).
    overlap_pf: the main loop's prefetch result ref ([None] or
        [(next_id, csv, records, meta, bmeta)]). When the current chunk
        enters its straggler tail, the next chunk's downloader is started
        into the shared queue and the entry is consumed (set to None).
    handoff: a _ChunkRun for THIS chunk whose downloader was already started
        by the previous call's overlap — skips CSV/setup entirely.
    handoff_out: single-element list; on return it holds the next chunk's
        live _ChunkRun (or None) for the caller to pass back as `handoff`.
    """
    _rcid = redis_chunk_id or chunk_id
    city = CITY_NAME

    run = None
    if handoff is not None:
        # Overlap handoff: downloader already running into item_queue; files,
        # sink and stats were created at overlap-start with this chunk's own
        # city naming. The prefetch thread already downloaded its CSV.
        run = handoff
        local_csv = run.local_csv
        records = None
        metadata_map = None
        total_records = run.total_records
        print(f"[CHUNK {chunk_id}] Resuming overlapped run: {total_records} "
              f"panos, {run.stats['ext_ok']} views already extracted", flush=True)
    elif preloaded:
        local_csv, records, metadata_map = preloaded
        total_records = len(records)
        print(f"[CHUNK {chunk_id}] Using pre-fetched CSV ({total_records} panos)")
    else:
        csv_filename = f"{city}_{chunk_id}.csv"
        csv_key = f"{CSV_BUCKET_PREFIX}/{csv_filename}"
        local_csv = str(work_dir / csv_filename)
        print(f"[CHUNK {chunk_id}] Downloading {csv_key}...")
        try:
            tq.report_status(REGION, INSTANCE_ID, "DOWNLOADING", chunk_id=_rcid)
        except Exception:
            pass
        if not r2.download_file(csv_key, local_csv, max_retries=3):
            raise RuntimeError(f"Failed to download {csv_key}")
        records, metadata_map = load_csv(local_csv)
        total_records = len(records)

    views_per_pano = HARDCODED_CONFIG['num_views']
    total_views_est = total_records * views_per_pano
    feature_dim = 8448

    print(f"[CHUNK {chunk_id}] {total_records} panos, ~{total_views_est} views expected")

    if total_records == 0:
        print(f"[CHUNK {chunk_id}] Empty chunk, skipping")
        _cleanup_chunk_files(work_dir, chunk_id, local_csv)
        return None

    # ── Step 3: Setup output files ──
    out_base = _output_base(city, chunk_id)
    features_file = str(work_dir / f"{out_base}.npy")
    metadata_file = str(work_dir / f"Metadata_{out_base}.jsonl")
    failed_file = str(work_dir / f"failed_{chunk_id}.jsonl")

    # ── Multi-GPU fan-out path ──
    # When a pool is provided, dispatch the chunk's records across N child
    # processes (one per GPU). Each child runs its own downloader+extractor
    # on its partition; parent stitches the partials into the final NPY.
    # Single-GPU path (gpu_pool=None) falls through to the legacy loop below.
    if gpu_pool is not None and gpu_pool.n_gpus > 1:
        # Each child over-allocates a memmap for its partition + writes
        # metadata/failed jsonl. Worst-case sum ≈ chunk_size_npy across all
        # children (it's still the same panos, just split). Plus the final
        # stitched NPY. Make sure we have headroom before dispatching.
        wait_for_disk_space(str(work_dir), MIN_FREE_GB)
        try:
            tq.report_status(REGION, INSTANCE_ID, "EXTRACTING", chunk_id=_rcid,
                             total=total_views_est)
        except Exception:
            pass
        t_pool_start = time.time()
        print(f"[CHUNK {chunk_id}] Dispatching to {gpu_pool.n_gpus}-GPU pool "
              f"({total_records} panos)", flush=True)
        try:
            partials = gpu_pool.process_chunk_partitioned(
                chunk_id, records, work_dir,
            )
        except Exception as e:
            # Pool dispatch failed (worker died, timeout, etc.). Let the
            # caller fail_task this chunk; pool itself may still be healthy
            # enough to handle the next chunk via its surviving workers.
            print(f"[CHUNK {chunk_id}] Pool dispatch failed: "
                  f"{type(e).__name__}: {e}", flush=True)
            _cleanup_chunk_files(work_dir, chunk_id, local_csv)
            raise

        # Check for IP block reported by any worker. Same semantics as
        # single-process path: discard partial output, raise IPBlockedError.
        if any((p.get('stats') or {}).get('ip_blocked_403') for p in partials):
            dl_ok = sum((p.get('stats') or {}).get('dl_ok', 0) for p in partials)
            dl_fail = sum((p.get('stats') or {}).get('dl_fail', 0) for p in partials)
            print(f"[ERROR] Chunk {chunk_id}: IP 403-blocked by Google in pool "
                  f"(dl_ok={dl_ok}, dl_fail={dl_fail}). Discarding partial.",
                  flush=True)
            # Best-effort cleanup of partial files
            for p in partials:
                for k in ('partial_npy', 'partial_meta', 'partial_failed'):
                    v = p.get(k)
                    if v and os.path.exists(v):
                        try:
                            os.remove(v)
                        except OSError:
                            pass
            _cleanup_chunk_files(work_dir, chunk_id, local_csv)
            raise IPBlockedError(
                f"Chunk {chunk_id}: IP 403-blocked by Google. "
                f"Partial output discarded — unclaim + self-destruct."
            )

        try:
            final_count, agg_stats = _stitch_partials(
                partials, features_file, metadata_file, failed_file,
                feature_dim=feature_dim,
            )
        except Exception as e:
            print(f"[CHUNK {chunk_id}] Stitch failed: {type(e).__name__}: {e}",
                  flush=True)
            _cleanup_chunk_files(work_dir, chunk_id, local_csv)
            raise

        elapsed = time.time() - t_pool_start
        speed = final_count / elapsed if elapsed > 0 else 0
        print(f"[CHUNK {chunk_id}] Pool extraction complete: "
              f"features={final_count} dl_ok={agg_stats['dl_ok']} "
              f"dl_fail={agg_stats['dl_fail']} elapsed={elapsed:.1f}s "
              f"speed={speed:.1f} views/s", flush=True)

        if final_count == 0 and total_records > 0:
            print(f"[ERROR] Chunk {chunk_id}: 0 features from {total_records} panos (pool)")
            _cleanup_chunk_files(work_dir, chunk_id, local_csv)
            raise RuntimeError(f"Zero features extracted from chunk {chunk_id} (pool)")

        return features_file, metadata_file, local_csv

    # ── Step 4: Download + extract (single-GPU path) ──
    # Thumbnail mode: no tile-row pre-computation needed (server renders
    # the perspective view directly from yaw/pitch/fov).
    if item_queue is None:
        # Legacy caller without a shared queue — overlap can't span calls.
        item_queue = queue.Queue(maxsize=HARDCODED_CONFIG['queue_size'])
        overlap_pf = None

    if run is None:
        # ~270MB memmap for 1K panos × 8 views × 8448 dim × 4 bytes,
        # allocated inside _start_chunk_run together with the sink + stats.
        run = _start_chunk_run(work_dir, item_queue, chunk_id, _rcid,
                               local_csv, records, metadata_map, city)

    shared_state = run.sink
    stats = run.stats
    dl_thread = run.dl_thread
    features_file = run.features_file
    metadata_file = run.metadata_file
    total_views_est = run.total_views_est
    next_run: Optional[_ChunkRun] = None
    overlap_attempted = False

    # ── Extraction loop ──
    loop_start = time.time()
    last_progress_time = time.time()
    last_progress_count = 0
    last_log_time = time.time()
    last_heartbeat_time = time.time()
    batch_times = []
    STALL_TIMEOUT = int(os.environ.get('STALL_TIMEOUT', '600'))
    LOG_INTERVAL = 30
    HEARTBEAT_INTERVAL = 30

    print(f"[CHUNK {chunk_id}] Starting extraction (batch_size={extractor.batch_size})", flush=True)
    try:
        tq.report_status(REGION, INSTANCE_ID, "EXTRACTING", chunk_id=_rcid,
                         total=total_views_est)
    except Exception:
        pass

    pending_batch: List[ViewItem] = run.pending_batch or []
    pending_futures = run.pending_futures
    run.pending_batch = []
    run.pending_futures = None
    last_gc = 0
    if stats['ext_ok']:
        # Handed-off run: progress already happened — seed the stall
        # detector so the head start isn't misread as a stall window.
        last_progress_count = stats['ext_ok']

    def _flush_pending():
        """Run inference on the pending batch, routing each row to its own
        chunk's sink (overlap can mix two chunks in one batch)."""
        nonlocal pending_batch, pending_futures, last_progress_time, \
            last_progress_count, last_gc
        if not pending_batch or pending_futures is None:
            pending_batch = []
            pending_futures = None
            return
        batch_start = time.time()
        try:
            feats_np, meta_batch, valid_indices = extractor.infer_prefetched(
                pending_batch, pending_futures)
            if feats_np is None:
                for it in pending_batch:
                    it.sink.dropped += 1
            else:
                # feats_np rows align with valid_indices order. Group rows
                # by sink; single-sink batches skip the fancy-index copy.
                valid_set = set(valid_indices)
                for i, it in enumerate(pending_batch):
                    if i not in valid_set:
                        it.sink.dropped += 1
                groups = {}
                for row, vi in enumerate(valid_indices):
                    s = pending_batch[vi].sink
                    g = groups.get(id(s))
                    if g is None:
                        g = (s, [], [])
                        groups[id(s)] = g
                    g[1].append(row)
                    g[2].append(meta_batch[row])
                for s, rows, metas in groups.values():
                    if len(rows) == feats_np.shape[0]:
                        s.write_batch(feats_np, metas)
                    else:
                        s.write_batch(feats_np[rows], metas)
                    owner = getattr(s, 'owner_stats', None)
                    if owner is not None:
                        owner['ext_ok'] += len(metas)
                del feats_np, meta_batch
                last_progress_time = time.time()
                last_progress_count = stats['ext_ok']
            batch_times.append(time.time() - batch_start)
            # Watermark, not modulo — ext_ok grows by batch-size steps and
            # rarely lands exactly on a 5000 multiple.
            if stats['ext_ok'] - last_gc >= 5000:
                gc.collect()
                last_gc = stats['ext_ok']
        except Exception as e:
            print(f"[ERROR] Batch extraction failed: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            for it in pending_batch:
                it.sink.dropped += 1
        finally:
            pending_batch = []
            pending_futures = None

    loop_error = None
    interrupted = False
    try:
        while True:
            now = time.time()

            # ── Redis heartbeat + status report + housekeeping (30s) ──
            if now - last_heartbeat_time >= HEARTBEAT_INTERVAL:
                # Disk check + log rotation moved here from the hot loop —
                # a statvfs + getsize per batch iteration is pure overhead.
                wait_for_disk_space(str(work_dir), MIN_FREE_GB)
                _truncate_log_if_huge()
                try:
                    tq.heartbeat(REGION, INSTANCE_ID, _rcid)
                    elapsed = now - loop_start
                    _spd = stats['ext_ok'] / elapsed if elapsed > 0 else 0
                    _rem = total_views_est - stats['ext_ok']
                    _eta = _rem / _spd if _spd > 0 else 0
                    tq.report_status(
                        REGION, INSTANCE_ID, "EXTRACTING",
                        chunk_id=_rcid, chunks_done=chunks_done_so_far,
                        processed=stats['ext_ok'], total=total_views_est,
                        speed=_spd, eta=_eta,
                    )
                except Exception as e:
                    print(f"[WARN] Heartbeat failed: {e}", flush=True)
                last_heartbeat_time = now

            # ── Chunk complete? (downloader dead + every produced view
            # written or dropped). With overlap the queue is rarely empty,
            # so this must be checked here, not only on a dry queue. ──
            if run.is_complete():
                if pending_batch and any(it.sink is shared_state
                                         for it in pending_batch):
                    _flush_pending()
                    continue
                break

            # ── Cross-chunk overlap: current chunk is in its straggler
            # tail and the next chunk's CSV is prefetched — start its
            # downloader into the SAME queue so neither the network nor
            # the GPU idles across the chunk boundary. ──
            if (CHUNK_OVERLAP and next_run is None and not overlap_attempted
                    and gpu_pool is None
                    and overlap_pf is not None and overlap_pf[0] is not None
                    and not stats.get('ip_blocked_403')):
                remaining_panos = (total_records - stats['dl_ok']
                                   - stats['dl_fail'])
                if remaining_panos <= max(
                        8, int(total_records * OVERLAP_TAIL_FRACTION)):
                    overlap_attempted = True
                    next_run = _try_start_overlap(
                        overlap_pf, r2, tq, work_dir, item_queue, hb)

            # ── Stall detection ──
            since_progress = now - last_progress_time
            if since_progress > STALL_TIMEOUT and stats['ext_ok'] == last_progress_count:
                msg = (f"[FATAL] Chunk {chunk_id} stalled — no progress for {since_progress:.0f}s")
                print(msg, flush=True)
                raise RuntimeError(msg)

            # ── Periodic status log ──
            if now - last_log_time >= LOG_INTERVAL:
                elapsed = now - loop_start
                speed = stats['ext_ok'] / elapsed if elapsed > 0 else 0
                remaining = total_views_est - stats['ext_ok']
                eta = remaining / speed if speed > 0 else 0
                pct = int(stats['ext_ok'] / total_views_est * 100) if total_views_est > 0 else 0
                print(f"PROGRESS|{INSTANCE_ID}|{chunk_id}|{stats['ext_ok']}|{total_views_est}|EXTRACTING", flush=True)
                print(f"[STATS] chunk={chunk_id} | "
                      f"views={stats['ext_ok']:,}/{total_views_est:,} ({pct}%) | "
                      f"speed={speed:.1f} views/s | eta={eta/60:.1f}min | "
                      f"dl_ok={stats['dl_ok']} | dl_fail={stats['dl_fail']} | "
                      f"queue={item_queue.qsize()}"
                      + (f" | overlap={next_run.chunk_id}" if next_run else ""),
                      flush=True)
                last_log_time = now

            # ── Fill next batch ──
            current_batch: List[ViewItem] = []
            fill_deadline = None
            dl_feeding = (dl_thread.is_alive()
                          or (next_run is not None
                              and next_run.dl_thread.is_alive()))
            while len(current_batch) < extractor.batch_size:
                try:
                    item = item_queue.get(timeout=0.01)
                    if item is _SENTINEL:
                        continue
                    current_batch.append(item)
                except queue.Empty:
                    # Coalesce: if the batch is still small and the queue
                    # only momentarily dried up, keep polling up to ~75ms
                    # before dispatching an inefficient tiny GPU batch.
                    if (len(current_batch) >= extractor.batch_size // 2
                            or not dl_feeding):
                        break
                    if fill_deadline is None:
                        fill_deadline = time.time() + 0.075
                    if time.time() >= fill_deadline:
                        break

            # ── No new items: drain pending then loop (completion is
            # detected at the top of the loop) ──
            if not current_batch:
                _flush_pending()
                continue

            # ── Submit decode of current batch ──
            current_futures = extractor.start_decode(current_batch)

            # ── GPU inference on pending batch ──
            _flush_pending()

            # ── Promote current batch to pending ──
            pending_batch = current_batch
            pending_futures = current_futures

    except KeyboardInterrupt:
        print("[WARN] Interrupted", flush=True)
        interrupted = True
    except BaseException as e:
        loop_error = e

    # ── Wind down ──
    # Normal exit: the current downloader is already dead (run.is_complete).
    # Abnormal exit (stall, interrupt, batch crash): stop BOTH downloaders
    # and drain the shared queue while joining — a full queue with no
    # consumer would deadlock the join forever. The overlap run cannot
    # survive an abnormal exit (the drain throws its items away), so it is
    # aborted and unclaimed.
    if dl_thread.is_alive() or loop_error is not None or interrupted:
        stats['stop_requested'] = True
        if next_run is not None:
            next_run.stats['stop_requested'] = True
        _join_deadline = time.time() + 120
        while dl_thread.is_alive() and time.time() < _join_deadline:
            try:
                while True:
                    item_queue.get_nowait()
            except queue.Empty:
                pass
            dl_thread.join(timeout=1.0)
        if dl_thread.is_alive():
            print("[WARN] Downloader thread did not exit within 120s — "
                  "proceeding without it", flush=True)
        if next_run is not None:
            _abort_run(next_run, tq, item_queue, work_dir,
                       reason="sibling_chunk_aborted", hb=hb)
            next_run = None

    # Leftover pending items at a clean break can only belong to the overlap
    # run (the current chunk's accounting closed) — carry them across the
    # handoff or their views would never be written and the next chunk's
    # completion accounting would never close.
    if next_run is not None and pending_batch:
        next_run.pending_batch = pending_batch
        next_run.pending_futures = pending_futures
        pending_batch = []
        pending_futures = None

    final_count = shared_state.write_idx
    shared_state.close()

    if loop_error is not None:
        _cleanup_chunk_files(work_dir, chunk_id, local_csv,
                             out_base=run.out_base)
        raise loop_error

    # ── Truncate memmap to actual size ──
    mm_ref = shared_state.memmap
    shared_state.memmap = None
    del mm_ref
    gc.collect()

    # ── If IP got 403-blocked at any point, this chunk is unfinished. ──
    # DISCARD any partial features — uploading a truncated NPY with
    # dl_fail > dl_ok would mark the chunk done in Redis and leave those
    # panos unprocessed forever. Raise IPBlockedError so the main loop
    # unclaims the chunk (no fcnt bump) and self-destructs.
    if stats.get('ip_blocked_403'):
        print(f"[ERROR] Chunk {chunk_id}: IP 403-blocked mid-chunk "
              f"(dl_ok={stats['dl_ok']}, dl_fail={stats['dl_fail']}, "
              f"partial_features={final_count}). Discarding partial output.",
              flush=True)
        # The overlap run shares this instance's IP — same block. Abort it
        # so its chunk is unclaimed before we self-destruct.
        if next_run is not None:
            _abort_run(next_run, tq, item_queue, work_dir,
                       reason="ip_blocked_403_overlap", hb=hb)
            next_run = None
        _cleanup_chunk_files(work_dir, chunk_id, local_csv,
                             out_base=run.out_base)
        raise IPBlockedError(
            f"Chunk {chunk_id}: IP 403-blocked by Google (probe returned "
            "sorry.google.com). Partial output discarded — unclaim + "
            "self-destruct."
        )

    if final_count == 0 and total_records > 0:
        print(f"[ERROR] Chunk {chunk_id}: 0 features from {total_records} panos!")
        # Exception paths in the caller never consume handoffs — abort the
        # overlap run (unclaims its chunk) rather than leaking it.
        if next_run is not None:
            _abort_run(next_run, tq, item_queue, work_dir,
                       reason="sibling_chunk_failed", hb=hb)
            next_run = None
        _cleanup_chunk_files(work_dir, chunk_id, local_csv,
                             out_base=run.out_base)
        raise RuntimeError(f"Zero features extracted from chunk {chunk_id}")

    if final_count > 0 and final_count < total_views_est:
        print(f"[CHUNK {chunk_id}] Truncating features: {total_views_est} → {final_count}")
        _truncate_npy(features_file, final_count)

    print(f"[CHUNK {chunk_id}] Extraction complete: {final_count} features")

    # Hand the live overlap run (if any) back to the caller, which passes it
    # to the next process_chunk call as `handoff`.
    if handoff_out is not None:
        handoff_out[0] = next_run
    elif next_run is not None:
        # No caller slot to receive it — should not happen, but never leak a
        # claimed chunk.
        _abort_run(next_run, tq, item_queue, work_dir,
                   reason="no_handoff_slot", hb=hb)

    # Return file paths for async upload (caller handles upload + cleanup)
    return features_file, metadata_file, local_csv


def _cleanup_chunk_files(work_dir: Path, chunk_id: str, local_csv: str = None,
                         out_base: str = None):
    """Delete local files for a processed chunk to free disk space.

    `out_base` should be passed by callers that hold the upload-thread
    snapshot, to avoid reading the mutable global CITY_NAME / TOTAL_CHUNKS
    from a different city's iteration (batch-mode race).
    """
    if out_base is None:
        # Fall back to globals for legacy/main-thread callers (single-city
        # only — batch-mode callers MUST pass out_base explicitly).
        out_base = _output_base(CITY_NAME, chunk_id)
    for f in [
        str(work_dir / f"{out_base}.npy"),
        str(work_dir / f"Metadata_{out_base}.jsonl"),
        str(work_dir / f"failed_{chunk_id}.jsonl"),
    ]:
        try:
            os.remove(f)
        except OSError:
            pass
    # Pool-mode partials: a failed pool chunk leaves partial_g*_{chunk_id}*
    # files (~258MB each) that would otherwise accumulate and fill the disk.
    try:
        for pf in Path(work_dir).glob(f"partial_g*_{chunk_id}*"):
            try:
                os.remove(pf)
            except OSError:
                pass
    except Exception:
        pass
    if local_csv:
        try:
            os.remove(local_csv)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Async Upload (parallel NPY + metadata)
# ═══════════════════════════════════════════════════════════════════════════════

def upload_chunk_files(r2, chunk_id: str, features_file: str, metadata_file: str,
                       local_csv: str, work_dir: Path,
                       npy_key: str, meta_key: str, out_base: str):
    """Upload NPY + metadata to R2 in parallel, then cleanup local files.

    The destination keys are computed by the caller (main thread) so that
    the background upload thread doesn't read mutable globals
    (CITY_NAME / FEATURES_BUCKET_PREFIX / TOTAL_CHUNKS) — which the next
    chunk's claim could overwrite mid-upload in batch mode and silently
    misroute the file to another city's prefix.
    """
    errors = []

    def _upload_npy():
        if os.path.exists(features_file) and os.path.getsize(features_file) > 0:
            if not upload_with_retry(r2, features_file, npy_key, label="NPY"):
                errors.append(f"Failed to upload NPY for chunk {chunk_id}")

    def _upload_meta():
        if os.path.exists(metadata_file) and os.path.getsize(metadata_file) > 0:
            if not upload_with_retry(r2, metadata_file, meta_key, label="META"):
                errors.append(f"Failed to upload metadata for chunk {chunk_id}")

    npy_t = threading.Thread(target=_upload_npy)
    meta_t = threading.Thread(target=_upload_meta)
    npy_t.start()
    meta_t.start()
    npy_t.join()
    meta_t.join()

    if errors:
        raise RuntimeError("; ".join(errors))

    _cleanup_chunk_files(work_dir, chunk_id, local_csv, out_base=out_base)
    print(f"[CHUNK {chunk_id}] Upload + cleanup complete")


def _do_background_upload(error_ref, r2, chunk_id, features_file, metadata_file,
                          local_csv, work_dir, npy_key, meta_key, out_base):
    """Thread target: upload chunk files and store any error in error_ref[0]."""
    try:
        upload_chunk_files(r2, chunk_id, features_file, metadata_file, local_csv,
                           work_dir, npy_key, meta_key, out_base)
    except Exception as e:
        error_ref[0] = e


# ═══════════════════════════════════════════════════════════════════════════════
# Async Upload Manager — bounded queue with backpressure
# ═══════════════════════════════════════════════════════════════════════════════

class _UploadJob:
    """One chunk's worth of files to upload. Decouples extraction from upload."""
    __slots__ = ('redis_chunk_id', 'chunk_id', 'features_file', 'metadata_file',
                 'local_csv', 'work_dir', 'npy_key', 'meta_key', 'out_base',
                 'submit_time')

    def __init__(self, redis_chunk_id, chunk_id, features_file, metadata_file,
                 local_csv, work_dir, npy_key, meta_key, out_base):
        self.redis_chunk_id = redis_chunk_id
        self.chunk_id = chunk_id
        self.features_file = features_file
        self.metadata_file = metadata_file
        self.local_csv = local_csv
        self.work_dir = work_dir
        self.npy_key = npy_key
        self.meta_key = meta_key
        self.out_base = out_base
        self.submit_time = time.time()


class AsyncUploadManager:
    """Bounded-queue uploader so extraction never blocks on a slow R2 PUT.

    Producer (extractor main loop) calls submit(job). If the queue is full
    the call blocks — that's the backpressure: when uploads can't keep up,
    extraction pauses instead of letting local disk fill or RSS balloon.

    Consumer threads pop jobs, run upload_chunk_files() (which already does
    NPY+metadata in parallel internally), and push (redis_chunk_id, error)
    onto a completion queue. The main loop drains completions every
    iteration and only marks chunks done in Redis after the upload landed
    — so a failed PUT never poisons the queue with a falsely-completed
    chunk.

    Sized for the failure modes:
      * max_pending=5 — caps local disk pressure. Each chunk's npy is up
        to ~270 MB; 5 outstanding = ~1.4 GB worst case, well under
        MIN_FREE_GB on a typical 100 GB instance.
      * num_workers=2 — each upload_chunk_files internally fires 2 PUTs
        (NPY + metadata) in parallel, so 2 workers = ~4 concurrent PUTs.
        More workers don't help once R2 throughput saturates.
    """

    def __init__(self, r2, max_pending: int = 5, num_workers: int = 2,
                 on_completion=None):
        self.r2 = r2
        self.max_pending = max_pending
        self._upload_q: queue.Queue = queue.Queue(maxsize=max_pending)
        self._completion_q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        # Total successful + failed uploads observed by this manager (lifetime).
        # Useful for stall-detection in the main loop ("nothing finished for 5 min").
        self._lifetime_completed = 0
        self._lifetime_failed = 0
        self._stats_lock = threading.Lock()
        self._on_completion = on_completion  # optional callback(redis_cid, error)
        self._workers = []
        for i in range(max(1, num_workers)):
            t = threading.Thread(target=self._worker_loop, name=f"upload-{i}",
                                 daemon=True)
            t.start()
            self._workers.append(t)
        print(f"[UPLOAD] AsyncUploadManager started: "
              f"max_pending={max_pending}, num_workers={num_workers}", flush=True)

    # ── Producer API ──

    def submit(self, job: _UploadJob, timeout: float = None) -> None:
        """Enqueue a job. Blocks (backpressure) if the queue is full.

        Raises queue.Full only if timeout is set and exceeded — main loop
        passes timeout=None so it just waits, which is the desired
        backpressure semantics.
        """
        self._upload_q.put(job, timeout=timeout)
        print(f"[UPLOAD] queued chunk={job.chunk_id} "
              f"(pending={self._upload_q.qsize()}/{self.max_pending})", flush=True)

    def pending_count(self) -> int:
        return self._upload_q.qsize()

    # ── Consumer API (main loop) ──

    def drain_completions(self):
        """Pop all available completions without blocking.

        Returns a list of (redis_chunk_id, error_or_None) tuples. The main
        loop calls this every iteration to translate finished uploads into
        Redis complete/fail calls.
        """
        results = []
        while True:
            try:
                results.append(self._completion_q.get_nowait())
            except queue.Empty:
                break
        return results

    def wait_one_completion(self, timeout: float = None):
        """Block until at least one upload finishes. Returns all completions
        accumulated by then (including the one we waited for)."""
        try:
            first = self._completion_q.get(timeout=timeout)
        except queue.Empty:
            return []
        results = [first]
        results.extend(self.drain_completions())
        return results

    def drain_all(self, log_interval: float = 30.0):
        """Block until every queued job has been processed. Returns the
        accumulated completions for the caller to mark done/failed in Redis.

        Called at shutdown so we never leave a healthy chunk's NPY un-uploaded.
        """
        results = []
        # Phase 1: wait for queue itself to empty (every get() must be paired
        # with task_done() — see _worker_loop).
        last_log = time.time()
        while self._upload_q.unfinished_tasks > 0:
            # Pull any completions that have piled up while we wait
            results.extend(self.drain_completions())
            if time.time() - last_log >= log_interval:
                print(f"[UPLOAD] drain_all: still {self._upload_q.unfinished_tasks} "
                      f"task(s) in flight", flush=True)
                last_log = time.time()
            time.sleep(0.5)
        # Phase 2: collect any final completions emitted after the queue cleared.
        # task_done() is called BEFORE we put on completion_q, so a brief drain
        # window is enough.
        for _ in range(20):
            results.extend(self.drain_completions())
            time.sleep(0.1)
        return results

    def shutdown(self, drain: bool = True, timeout: float = 600.0):
        """Stop workers. If drain=True (default), wait for queued jobs first.

        Returns any final completions so the caller can mark them in Redis.
        """
        final = []
        if drain:
            final = self.drain_all()
        self._stop.set()
        # Wake any worker blocked on queue.get(timeout=...)
        for _ in self._workers:
            try:
                self._upload_q.put_nowait(None)
            except queue.Full:
                pass
        deadline = time.time() + timeout
        for t in self._workers:
            remaining = max(0.5, deadline - time.time())
            t.join(timeout=remaining)
        return final

    # ── Stats ──

    def stats(self):
        with self._stats_lock:
            return {
                'pending': self._upload_q.qsize(),
                'completed': self._lifetime_completed,
                'failed': self._lifetime_failed,
                'max_pending': self.max_pending,
            }

    # ── Internals ──

    def _worker_loop(self):
        while not self._stop.is_set():
            try:
                job = self._upload_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if job is None:
                # Sentinel — wake up and exit. task_done so drain_all() unblocks.
                self._upload_q.task_done()
                break
            err = None
            t_start = time.time()
            try:
                upload_chunk_files(
                    self.r2, job.chunk_id, job.features_file, job.metadata_file,
                    job.local_csv, job.work_dir,
                    job.npy_key, job.meta_key, job.out_base,
                )
            except Exception as e:
                err = e
                import traceback
                print(f"[UPLOAD] FAILED chunk={job.chunk_id}: "
                      f"{type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                # The chunk gets unclaimed for another (possibly healthier)
                # worker — clean local files so this worker's disk doesn't
                # fill with orphans it will never upload.
                try:
                    _cleanup_chunk_files(job.work_dir, job.chunk_id,
                                         job.local_csv, out_base=job.out_base)
                except Exception as ce:
                    print(f"[UPLOAD] cleanup after failure failed: {ce}",
                          flush=True)
            finally:
                # task_done BEFORE pushing completion so drain_all sees a clean
                # state before the main loop reads the completion. Reverse
                # order would risk a race where drain_all returns "queue empty"
                # but completions are still being drained one-by-one.
                self._upload_q.task_done()
                with self._stats_lock:
                    if err is None:
                        self._lifetime_completed += 1
                    else:
                        self._lifetime_failed += 1
                self._completion_q.put((job.redis_chunk_id, err))
                if err is None:
                    print(f"[UPLOAD] done chunk={job.chunk_id} "
                          f"({time.time() - t_start:.1f}s)", flush=True)
                if self._on_completion:
                    try:
                        self._on_completion(job.redis_chunk_id, err)
                    except Exception as cb_e:
                        print(f"[UPLOAD] on_completion callback failed: {cb_e}",
                              flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Background Heartbeat — keeps every claimed chunk fresh in Redis
# ═══════════════════════════════════════════════════════════════════════════════

class HeartbeatThread(threading.Thread):
    """Daemon that heartbeats every registered chunk every `interval` seconds.

    The in-loop heartbeat only covers single-GPU extraction. Pool mode
    (process_chunk_partitioned blocks the main thread up to 7200s), the
    upload phase, and prefetched chunks all exceed STALE_TIMEOUT=300s with
    no heartbeat — reclaim_stale requeues them and the fleet does duplicate
    work. Register a chunk at claim time, unregister on complete/fail/unclaim.
    The in-loop heartbeat stays (harmless duplicate).
    """

    def __init__(self, tq: TaskQueue, region: str, worker_id: str,
                 interval: float = 30.0):
        super().__init__(name="heartbeat", daemon=True)
        self.tq = tq
        self.region = region
        self.worker_id = worker_id
        self.interval = interval
        self._chunks: Set[str] = set()
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()

    def register(self, chunk_id: str):
        if not chunk_id:
            return
        with self._lock:
            self._chunks.add(chunk_id)

    def unregister(self, chunk_id: str):
        if not chunk_id:
            return
        with self._lock:
            self._chunks.discard(chunk_id)

    def stop(self):
        self._stop_evt.set()

    def run(self):
        while not self._stop_evt.wait(self.interval):
            with self._lock:
                chunks = list(self._chunks)
            for cid in chunks:
                try:
                    # Conditional heartbeat: only refreshes chunks still in
                    # the active hash, so a racing complete/unclaim can't be
                    # undone by a late HSET from this thread.
                    self.tq.heartbeat_if_active(self.region, self.worker_id, cid)
                except Exception as e:
                    print(f"[WARN] Background heartbeat for {cid} failed: {e}",
                          flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-GPU Worker Pool (Option C: one process per GPU)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Architecture:
#   * Parent process: claims chunks, downloads+parses CSV, partitions records,
#     dispatches to per-GPU child workers, merges partial NPYs back into the
#     final NPY, submits to AsyncUploadManager. Handles Redis state.
#   * Child process (one per GPU): owns a long-lived GpuExtractor pinned to
#     cuda:gpu_id. On each "PROCESS" command it spawns a downloader thread
#     for its partition's records, extracts features, writes
#     partial_g{N}_{chunk}.npy + partial_g{N}_meta_{chunk}.jsonl.
#
# Why multi-process, not multi-thread:
#   Python GIL serialises decode + memmap writes; multi-process gets every
#   GPU's downloader and inference loop on its own interpreter so the only
#   shared resource is the (network-bound) Google Street View server.
#
# Single-GPU detection:
#   If torch.cuda.device_count() <= 1, the pool is NOT spawned and the
#   pipeline runs the original single-process path unchanged. The env var
#   DISABLE_MULTI_GPU=1 forces the legacy path even on multi-GPU machines
#   (emergency fallback for debugging).

# Spawn-context multiprocessing — fork would corrupt CUDA contexts. We
# import lazily inside the helpers so module import doesn't drag in heavy
# state in the child process before it has a chance to pin the GPU.


def _child_extract_partition(extractor: 'GpuExtractor', records: List[dict],
                              partition_id: int, work_dir: Path, chunk_id: str,
                              gpu_id: int) -> dict:
    """Run inside a child process: download + extract one partition.

    Writes per-partition NPY + metadata to disk. Returns a dict the parent
    uses to stitch the final outputs. Never raises into the parent — the
    caller (_gpu_worker_main) wraps this in try/except and sends ERROR
    over the result queue.
    """
    result = {
        'gpu_id': gpu_id,
        'partition_id': partition_id,
        'partial_npy': None,
        'partial_meta': None,
        'partial_failed': None,
        'feature_count': 0,
        'stats': {'dl_ok': 0, 'dl_fail': 0, 'ext_ok': 0,
                  'views_produced': 0, 'dl_done': False,
                  'ip_blocked_403': False},
    }

    if not records:
        return result

    total_records = len(records)
    views_per_pano = HARDCODED_CONFIG['num_views']
    total_views_est = max(1, total_records * views_per_pano)
    feature_dim = 8448

    base = f"partial_g{partition_id}_{chunk_id}"
    partial_npy = str(work_dir / f"{base}.npy")
    partial_meta = str(work_dir / f"{base}_meta.jsonl")
    partial_failed = str(work_dir / f"{base}_failed.jsonl")

    features_memmap = np.lib.format.open_memmap(
        partial_npy, mode='w+', dtype='float32',
        shape=(total_views_est, feature_dim),
    )
    shared_state = SharedState(features_memmap, partial_meta, partial_failed, start_idx=0)

    # downloader_thread expects {panoid: row_dict}; build locally so we don't
    # have to ship metadata_map across processes (would just be derived from
    # records anyway).
    metadata_map = {}
    for r in records:
        pid = r.get('panoid') if isinstance(r, dict) else None
        if pid:
            metadata_map[pid] = r

    item_queue = queue.Queue(maxsize=HARDCODED_CONFIG['queue_size'])
    stats = result['stats']

    # Per-IP request budget: every pool child shares this instance's single
    # public IP. Divide MAX_THREADS across the children instead of giving
    # each the full budget — n_gpus × 30 concurrent panos from one IP trips
    # Google's per-IP 403 gate. The total stays at MAX_THREADS (sacred).
    child_config = dict(HARDCODED_CONFIG)
    try:
        _n_dev = max(1, torch.cuda.device_count())
    except Exception:
        _n_dev = 1
    child_config['max_threads'] = max(4, HARDCODED_CONFIG['max_threads'] // _n_dev)

    dl_thread = threading.Thread(
        target=downloader_thread,
        args=(records, child_config, item_queue, metadata_map, stats, shared_state),
        name=f"dl-g{partition_id}",
    )
    dl_thread.start()

    pending_batch: List[ViewItem] = []
    pending_futures = None
    last_log = time.time()
    started = time.time()

    try:
        while True:
            now = time.time()
            if now - last_log >= 30:
                elapsed = now - started
                spd = stats['ext_ok'] / elapsed if elapsed > 0 else 0
                # Per-worker progress line. Parent aggregates these into the
                # overall PROGRESS line consumed by the orchestrator.
                print(f"[GPU-{gpu_id}/{chunk_id}] views={stats['ext_ok']:,} "
                      f"speed={spd:.1f}/s dl_ok={stats['dl_ok']} "
                      f"dl_fail={stats['dl_fail']} queue={item_queue.qsize()}",
                      flush=True)
                last_log = now

            current_batch: List[ViewItem] = []
            while len(current_batch) < extractor.batch_size:
                try:
                    item = item_queue.get(timeout=0.01)
                    if item is _SENTINEL:
                        continue
                    current_batch.append(item)
                except queue.Empty:
                    break

            if not current_batch:
                if pending_batch and pending_futures is not None:
                    try:
                        feats_np, meta_batch, _ = extractor.infer_prefetched(
                            pending_batch, pending_futures)
                        if feats_np is not None and len(meta_batch) > 0:
                            shared_state.write_batch(feats_np, meta_batch)
                            stats['ext_ok'] += len(meta_batch)
                            del feats_np, meta_batch
                    except Exception as e:
                        print(f"[GPU-{gpu_id}] drain batch fail: "
                              f"{type(e).__name__}: {e}", flush=True)
                    finally:
                        pending_batch, pending_futures = [], None
                if not dl_thread.is_alive():
                    break
                continue

            current_futures = extractor.start_decode(current_batch)
            if pending_batch and pending_futures is not None:
                try:
                    feats_np, meta_batch, _ = extractor.infer_prefetched(
                        pending_batch, pending_futures)
                    if feats_np is not None and len(meta_batch) > 0:
                        shared_state.write_batch(feats_np, meta_batch)
                        stats['ext_ok'] += len(meta_batch)
                        del feats_np, meta_batch
                except Exception as e:
                    print(f"[GPU-{gpu_id}] batch fail: "
                          f"{type(e).__name__}: {e}", flush=True)
            pending_batch, pending_futures = current_batch, current_futures
    finally:
        # Make absolutely sure the downloader thread exits before we touch
        # the memmap close path — otherwise it could write past write_idx
        # while we're truncating. Set the stop flag + drain the queue while
        # joining so a full queue with no consumer can't deadlock the join.
        stats['stop_requested'] = True
        _join_deadline = time.time() + 120
        while dl_thread.is_alive() and time.time() < _join_deadline:
            try:
                while True:
                    item_queue.get_nowait()
            except queue.Empty:
                pass
            dl_thread.join(timeout=1.0)
        final_count = shared_state.write_idx
        shared_state.close()
        # Release memmap before parent stitches it on its side.
        try:
            del features_memmap
        except Exception:
            pass
        gc.collect()

    result['partial_npy'] = partial_npy
    result['partial_meta'] = partial_meta
    result['partial_failed'] = partial_failed
    result['feature_count'] = int(final_count)
    return result


def _gpu_worker_main(gpu_id: int, in_q, out_q):
    """Child-process entry point. Long-lived: serves multiple chunks until STOP.

    The parent spawns one of these per GPU. We pin to cuda:gpu_id BEFORE
    instantiating GpuExtractor so the model lands on the right device and
    the OOM probe measures the right card.

    Communication protocol:
      Parent → child: ('PROCESS', chunk_id, records, work_dir_str, partition_id)
                      ('STOP',)
      Child → parent (out_q): ('READY', gpu_id)         # once at startup
                              ('DONE', result_dict)     # per chunk
                              ('ERROR', error_string)   # init or per-chunk failure
    """
    # Re-establish flush=True print behaviour in the child.
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True) if hasattr(_sys.stdout, 'reconfigure') else None
    print(f"[GPU-{gpu_id}] child started pid={os.getpid()}", flush=True)

    try:
        # Pin device early so all subsequent torch.* calls default to it.
        torch.cuda.set_device(gpu_id)
        extractor = GpuExtractor(gpu_id=gpu_id, pool_mode=True)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            out_q.put(('ERROR', f"gpu={gpu_id} init: {type(e).__name__}: {e}\n{tb}"))
        except Exception:
            pass
        print(f"[GPU-{gpu_id}] FATAL init: {e}\n{tb}", flush=True)
        return

    try:
        out_q.put(('READY', gpu_id))
    except Exception as e:
        print(f"[GPU-{gpu_id}] failed to signal READY: {e}", flush=True)
        return

    while True:
        try:
            msg = in_q.get(timeout=60)
        except queue.Empty:
            # A blocking get() with a dead parent would park this child (and
            # its CUDA context) forever. If we've been reparented to init
            # (ppid==1, Linux container), the parent is gone — exit.
            try:
                if os.getppid() == 1:
                    print(f"[GPU-{gpu_id}] parent died (reparented to init); "
                          f"exiting", flush=True)
                    break
            except Exception:
                pass
            continue
        except (EOFError, KeyboardInterrupt):
            print(f"[GPU-{gpu_id}] in_q closed; exiting", flush=True)
            break
        if not isinstance(msg, tuple) or not msg:
            continue
        cmd = msg[0]
        if cmd == 'STOP':
            print(f"[GPU-{gpu_id}] STOP received; exiting", flush=True)
            break
        if cmd != 'PROCESS':
            print(f"[GPU-{gpu_id}] unknown cmd {cmd!r}; ignoring", flush=True)
            continue
        try:
            _, chunk_id, records, work_dir_str, partition_id = msg
        except ValueError as e:
            try:
                out_q.put(('ERROR', f"gpu={gpu_id} malformed PROCESS msg: {e}"))
            except Exception:
                pass
            continue
        work_dir = Path(work_dir_str)
        t_start = time.time()
        try:
            result = _child_extract_partition(
                extractor, records, partition_id, work_dir, chunk_id, gpu_id,
            )
            result['elapsed_sec'] = time.time() - t_start
            out_q.put(('DONE', result))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            try:
                out_q.put(('ERROR', f"gpu={gpu_id} chunk={chunk_id}: "
                                    f"{type(e).__name__}: {e}\n{tb}"))
            except Exception:
                pass
            print(f"[GPU-{gpu_id}] chunk {chunk_id} FAILED: {e}\n{tb}", flush=True)
            # Don't break — try next chunk. The parent decides whether to
            # mark the chunk failed and continue with the remaining workers.
        finally:
            # Aggressive cleanup between chunks: free CUDA cache, force GC.
            # Long-running pools accumulate fragmentation otherwise.
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

    # Graceful cleanup on STOP.
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"[GPU-{gpu_id}] child exiting", flush=True)


class GpuWorkerPool:
    """Long-lived multi-process GPU pool.

    Lifecycle:
      pool = GpuWorkerPool(n_gpus=4)        # spawn + wait for READY
      partials = pool.process_chunk_partitioned(chunk_id, records, work_dir)
      pool.shutdown()                       # send STOP, join, terminate stragglers

    Failure semantics:
      * If any worker dies during init → constructor raises after killing
        siblings. Caller should fall back to single-GPU path.
      * If any worker errors on a chunk → process_chunk_partitioned raises
        RuntimeError describing which GPU(s) failed. The parent's process_chunk
        catches and fail_task's that Redis chunk; the pool stays alive for
        future chunks (each worker is independent).
      * If a worker dies (process exit) mid-chunk → the parent detects
        non-alive process while waiting on out_q and raises. The dead
        worker is gone for the rest of the run; pool degrades gracefully
        (we just don't dispatch to that partition next time).
    """

    INIT_TIMEOUT_SEC = 600     # 10 min for all workers to load model + probe
    CHUNK_TIMEOUT_SEC = 7200   # 2 hours per chunk worst case

    def __init__(self, n_gpus: int):
        if n_gpus < 1:
            raise ValueError(f"GpuWorkerPool requires n_gpus>=1 (got {n_gpus})")
        self.n_gpus = n_gpus
        self._alive = [False] * n_gpus  # toggled True on READY
        import torch.multiprocessing as mp
        self._mp = mp
        # 'spawn' is mandatory: CUDA cannot survive os.fork().
        try:
            self._ctx = mp.get_context('spawn')
        except RuntimeError:
            # Already set somewhere upstream; that's fine.
            self._ctx = mp.get_context()
        self.in_qs = [self._ctx.Queue(maxsize=4) for _ in range(n_gpus)]
        self.out_q = self._ctx.Queue()
        self.procs = []
        print(f"[POOL] spawning {n_gpus} GPU worker process(es)...", flush=True)
        for i in range(n_gpus):
            p = self._ctx.Process(
                target=_gpu_worker_main,
                args=(i, self.in_qs[i], self.out_q),
                name=f"gpu-worker-{i}",
                daemon=False,  # CUDA child cannot be daemon (CUDA forbids it)
            )
            p.start()
            self.procs.append(p)

        # Wait for READY from every worker. Strict — if any dies before
        # signalling, abort the whole pool so the caller can fall back.
        ready = 0
        deadline = time.time() + self.INIT_TIMEOUT_SEC
        while ready < n_gpus and time.time() < deadline:
            # Detect early death of any worker.
            for idx, p in enumerate(self.procs):
                if not p.is_alive() and not self._alive[idx]:
                    self._kill_all()
                    raise RuntimeError(
                        f"GPU worker {idx} (pid {p.pid}) died during init "
                        f"with exit code {p.exitcode}"
                    )
            try:
                sig, payload = self.out_q.get(timeout=15)
            except queue.Empty:
                continue
            if sig == 'READY':
                gid = int(payload)
                if 0 <= gid < n_gpus and not self._alive[gid]:
                    self._alive[gid] = True
                    ready += 1
                    print(f"[POOL] GPU {gid} ready ({ready}/{n_gpus})", flush=True)
            elif sig == 'ERROR':
                self._kill_all()
                raise RuntimeError(f"GPU worker init error: {payload}")
            else:
                print(f"[POOL] ignored startup msg {sig!r}", flush=True)

        if ready < n_gpus:
            self._kill_all()
            raise RuntimeError(
                f"GpuWorkerPool init timeout: only {ready}/{n_gpus} workers "
                f"ready after {self.INIT_TIMEOUT_SEC}s"
            )
        print(f"[POOL] all {n_gpus} workers ready", flush=True)

    # ── Dispatch ──

    def process_chunk_partitioned(self, chunk_id: str, records: List[dict],
                                   work_dir: Path,
                                   timeout: float = None) -> List[dict]:
        """Partition records across alive workers and collect partial outputs.

        Returns the list of partial-result dicts (one per dispatched partition).
        Raises RuntimeError if any worker reports ERROR or dies before DONE.
        """
        timeout = timeout if timeout is not None else self.CHUNK_TIMEOUT_SEC

        alive_indices = [i for i, ok in enumerate(self._alive)
                         if ok and self.procs[i].is_alive()]
        if not alive_indices:
            raise RuntimeError("GpuWorkerPool has no alive workers")

        partitions = self._partition_records(records, len(alive_indices))
        # Send each partition; record the (gpu_index, partition_id) for collection.
        for slot, gpu_idx in enumerate(alive_indices):
            part = partitions[slot]
            self.in_qs[gpu_idx].put(
                ('PROCESS', chunk_id, part, str(work_dir), slot)
            )

        expected = len(alive_indices)
        partials: List[dict] = [None] * expected
        errors: List[str] = []
        deadline = time.time() + timeout
        done_count = 0

        while done_count < expected and time.time() < deadline:
            # Detect worker death (process exited while we were waiting).
            for gpu_idx in alive_indices:
                p = self.procs[gpu_idx]
                if not p.is_alive():
                    self._alive[gpu_idx] = False
                    msg = (f"GPU worker {gpu_idx} died mid-chunk "
                           f"(pid {p.pid}, exit {p.exitcode})")
                    if msg not in errors:
                        errors.append(msg)
            try:
                sig, payload = self.out_q.get(timeout=5)
            except queue.Empty:
                continue
            if sig == 'DONE':
                pid = payload.get('partition_id', -1)
                if 0 <= pid < expected and partials[pid] is None:
                    partials[pid] = payload
                    done_count += 1
            elif sig == 'ERROR':
                errors.append(str(payload))
                done_count += 1  # count the slot so we don't deadlock
            else:
                print(f"[POOL] ignored runtime msg {sig!r}", flush=True)

        if errors:
            raise RuntimeError(
                f"GpuWorkerPool error(s) on chunk {chunk_id}: " +
                " | ".join(errors[:5])
            )
        missing = [i for i, p in enumerate(partials) if p is None]
        if missing:
            raise RuntimeError(
                f"GpuWorkerPool timeout on chunk {chunk_id}: "
                f"partitions {missing} never reported DONE in {timeout:.0f}s"
            )
        return partials

    @staticmethod
    def _partition_records(records: List[dict], n: int) -> List[List[dict]]:
        """Split records into n partitions via stride (round-robin).

        Why stride and not contiguous slices: CSV rows often cluster nearby
        panos together. Contiguous slices would dump all of "downtown" on one
        GPU and force its IP through Google's per-area rate limit; striding
        spreads geographically clustered panos across GPUs so the per-IP
        request budget for any sub-area is shared.
        """
        if n <= 1:
            return [list(records)]
        out = [[] for _ in range(n)]
        for i, r in enumerate(records):
            out[i % n].append(r)
        return out

    # ── Shutdown ──

    def shutdown(self, drain_timeout: float = 60.0):
        """Send STOP to all workers and join. Terminates stragglers."""
        print(f"[POOL] shutting down...", flush=True)
        for i, q in enumerate(self.in_qs):
            if self.procs[i].is_alive():
                try:
                    q.put(('STOP',), timeout=5)
                except Exception:
                    pass
        deadline = time.time() + drain_timeout
        for p in self.procs:
            remaining = max(1.0, deadline - time.time())
            p.join(timeout=remaining)
            if p.is_alive():
                print(f"[POOL] worker {p.pid} did not exit; terminating", flush=True)
                try:
                    p.terminate()
                except Exception:
                    pass
        # Final reap of any zombies.
        for p in self.procs:
            try:
                p.join(timeout=5)
            except Exception:
                pass
        print(f"[POOL] shutdown complete", flush=True)

    def _kill_all(self):
        for p in self.procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in self.procs:
            try:
                p.join(timeout=10)
            except Exception:
                pass


def _stitch_partials(partials: List[dict], features_file: str,
                     metadata_file: str, failed_file: str,
                     feature_dim: int = 8448) -> Tuple[int, dict]:
    """Concatenate per-GPU partial outputs into the final NPY + metadata files.

    Re-indexes `feature_index` in each metadata line so it points into the
    final stitched NPY rather than the partial that produced it.

    Returns (total_count, aggregated_stats).
    """
    total_count = sum(int(p.get('feature_count', 0)) for p in partials)
    agg_stats = {'dl_ok': 0, 'dl_fail': 0, 'ext_ok': 0, 'views_produced': 0,
                 'ip_blocked_403': False}
    for p in partials:
        s = p.get('stats') or {}
        agg_stats['dl_ok'] += int(s.get('dl_ok', 0))
        agg_stats['dl_fail'] += int(s.get('dl_fail', 0))
        agg_stats['ext_ok'] += int(s.get('ext_ok', 0))
        agg_stats['views_produced'] += int(s.get('views_produced', 0))
        if s.get('ip_blocked_403'):
            agg_stats['ip_blocked_403'] = True

    if total_count == 0:
        # Touch the metadata + failed files to avoid downstream "missing file"
        # confusion. NPY left non-existent → caller treats as empty chunk.
        try:
            open(metadata_file, 'w').close()
            open(failed_file, 'w').close()
        except Exception:
            pass
        return 0, agg_stats

    final_mm = np.lib.format.open_memmap(
        features_file, mode='w+', dtype='float32',
        shape=(total_count, feature_dim),
    )

    cursor = 0
    try:
        with open(metadata_file, 'w', encoding='utf-8') as meta_out, \
             open(failed_file, 'w', encoding='utf-8') as fail_out:
            # Process partitions in partition_id order so the final NPY's row
            # ordering is deterministic across runs.
            ordered = sorted(partials, key=lambda x: x.get('partition_id', 0))
            for p in ordered:
                n = int(p.get('feature_count', 0))
                src_npy = p.get('partial_npy')
                src_meta = p.get('partial_meta')
                src_failed = p.get('partial_failed')

                if n > 0 and src_npy and os.path.exists(src_npy):
                    try:
                        src = np.load(src_npy, mmap_mode='r')
                        # Tolerate src being larger than n (over-allocated memmap)
                        if src.shape[0] < n:
                            print(f"[STITCH][WARN] {src_npy} has only "
                                  f"{src.shape[0]} rows, expected {n}", flush=True)
                            n = src.shape[0]
                        final_mm[cursor:cursor + n] = src[:n]
                        del src
                    except Exception as e:
                        print(f"[STITCH][ERROR] failed to copy {src_npy}: {e}",
                              flush=True)
                        raise

                    if src_meta and os.path.exists(src_meta):
                        with open(src_meta, 'r', encoding='utf-8') as mf:
                            for line in mf:
                                if not line.strip():
                                    continue
                                try:
                                    rec = json.loads(line)
                                except Exception:
                                    continue  # drop malformed lines
                                if 'feature_index' in rec:
                                    try:
                                        rec['feature_index'] = (
                                            cursor + int(rec['feature_index'])
                                        )
                                    except Exception:
                                        pass
                                meta_out.write(json.dumps(rec) + '\n')
                    cursor += n
                # Append failures verbatim (no re-indexing needed)
                if src_failed and os.path.exists(src_failed):
                    try:
                        with open(src_failed, 'r', encoding='utf-8') as ff:
                            shutil.copyfileobj(ff, fail_out)
                    except Exception as e:
                        print(f"[STITCH][WARN] failed to copy failed log: {e}",
                              flush=True)
    finally:
        try:
            final_mm.flush()
            del final_mm
        except Exception:
            pass
        gc.collect()

        # Cleanup partial files. Failures here are non-fatal — the chunk
        # is already stitched to the final destination.
        for p in partials:
            for k in ('partial_npy', 'partial_meta', 'partial_failed'):
                v = p.get(k)
                if v and os.path.exists(v):
                    try:
                        os.remove(v)
                    except OSError:
                        pass

    return cursor, agg_stats


# ═══════════════════════════════════════════════════════════════════════════════
# Prefetch Next Chunk (background CSV download during extraction)
# ═══════════════════════════════════════════════════════════════════════════════

def _do_prefetch(result_ref, r2, tq, work_dir, skip_prefixes=None, hb=None):
    """Thread target: claim next chunk + download/parse CSV. Stores result in result_ref[0].
    skip_prefixes: shared set of city CSV prefixes to skip (missing on R2).
    hb: optional HeartbeatThread — the claimed chunk is registered so it
    doesn't go stale while the current chunk is still extracting."""
    try:
        next_id = tq.claim_task(REGION, INSTANCE_ID)
        if next_id is None:
            return
    except Exception as e:
        print(f"[PREFETCH] Claim error: {e}")
        return

    if hb is not None:
        hb.register(next_id)

    # Everything after a successful claim must either hand the chunk to the
    # main loop (result_ref) or put it back — an exception here would leave
    # it dangling in active until reclaim_stale.
    try:
        # Batch mode: look up per-chunk metadata
        _bmeta = None
        try:
            _bmeta = tq.get_chunk_meta(REGION, next_id)
        except Exception:
            pass

        if _bmeta:
            city = _bmeta['city_name']
            csv_prefix = _bmeta['csv_prefix']
            file_cid = f"chunk_{_bmeta['chunk_num']:04d}"
        else:
            city = CITY_NAME
            csv_prefix = CSV_BUCKET_PREFIX
            file_cid = next_id

        # Skip if this city's CSVs are known missing
        if skip_prefixes and csv_prefix in skip_prefixes:
            print(f"[PREFETCH] Skipping {next_id} — city '{city}' CSVs missing on R2")
            try:
                tq.fail_task(REGION, next_id, INSTANCE_ID,
                             f"city_csv_missing:{csv_prefix}")
            except Exception:
                pass
            if hb is not None:
                hb.unregister(next_id)
            return

        csv_fn = f"{city}_{file_cid}.csv"
        csv_key = f"{csv_prefix}/{csv_fn}"
        local_csv = str(work_dir / csv_fn)

        print(f"[PREFETCH] Downloading CSV for next chunk {next_id}...")
        if not r2.download_file(csv_key, local_csv, max_retries=3):
            # Classify: real 404 (city CSVs genuinely absent) vs transient
            # R2/network failure. Blacklisting a whole city because R2 had
            # a bad minute permanently failed every one of its chunks.
            missing = r2.object_missing(csv_key)
            if missing is True:
                print(f"[PREFETCH] CSV {csv_key} missing on R2 (404)")
                if skip_prefixes is not None:
                    skip_prefixes.add(csv_prefix)
                    print(f"[PREFETCH] Blacklisting city prefix '{csv_prefix}' — "
                          f"all future chunks from {city} will be skipped")
                try:
                    tq.fail_task(REGION, next_id, INSTANCE_ID,
                                 f"city_csv_missing:{csv_prefix}")
                except Exception:
                    pass
            else:
                print(f"[PREFETCH] CSV download failed for {next_id} "
                      f"(transient) — unclaiming, NOT blacklisting")
                try:
                    tq.unclaim_task(REGION, next_id, INSTANCE_ID,
                                    reason="csv_download_transient", back=True)
                except Exception:
                    pass
            if hb is not None:
                hb.unregister(next_id)
            return

        records, metadata_map = load_csv(local_csv)
        result_ref[0] = (next_id, local_csv, records, metadata_map, _bmeta)
        print(f"[PREFETCH] Ready: chunk {next_id} ({len(records)} panos)")
    except Exception as e:
        print(f"[PREFETCH] Error after claim: {type(e).__name__}: {e} — "
              f"unclaiming {next_id}")
        try:
            tq.unclaim_task(REGION, next_id, INSTANCE_ID,
                            reason=f"prefetch_error: {str(e)[:100]}", back=True)
        except Exception as re:
            print(f"[PREFETCH] unclaim failed: {re}")
        if hb is not None:
            hb.unregister(next_id)


# ═══════════════════════════════════════════════════════════════════════════════
# Resume — Reconcile Redis with R2 (source of truth)
# ═══════════════════════════════════════════════════════════════════════════════

def reconcile_with_r2(r2, tq: TaskQueue):
    """
    Scan R2 for already-uploaded NPY+metadata files and mark those chunks done
    in Redis. Also reclaim stale active tasks and recover lost tasks.
    This makes the pipeline fully resumable after worker crashes.
    """
    # Batch mode: skip R2 scan (chunks have mixed paths), just do stale recovery
    try:
        if (tq.redis.hlen(f"job:{REGION}:cmap") or 0) > 0:
            print("[RESUME] Batch mode — skipping R2 reconciliation, using stale recovery only")
            stale = tq.reclaim_stale(REGION)
            if stale:
                print(f"[RESUME] Reclaimed {len(stale)} stale tasks: {stale}")
            lost = tq.recover_lost_tasks(REGION)
            if lost:
                print(f"[RESUME] Recovered {len(lost)} lost tasks: {lost}")
            progress = tq.get_progress(REGION)
            print(f"[RESUME] After recovery — todo: {progress['todo']}, "
                  f"active: {progress['active']}, done: {progress['done']}/{progress['total_chunks']}")
            return
    except Exception:
        pass

    city = CITY_NAME
    prefix = f"{FEATURES_BUCKET_PREFIX}/{city}_"

    print(f"[RESUME] Scanning R2 for existing outputs under '{prefix}'...")

    # List NPY and metadata files already in R2
    npy_keys = set(r2.list_objects(prefix, suffix='.npy'))
    meta_keys = set(r2.list_objects(
        f"{FEATURES_BUCKET_PREFIX}/Metadata_{city}_", suffix='.jsonl'
    ))

    # Extract chunk IDs from NPY filenames: "Features/KansasCity_1.11.npy" → "chunk_0001"
    import re as _re
    npy_pattern = _re.compile(rf'^{_re.escape(city)}_(\d+)\.\d+\.npy$')
    done_chunks = set()
    for key in npy_keys:
        filename = key.rsplit('/', 1)[-1]
        m = npy_pattern.match(filename)
        if not m:
            continue
        chunk_num = int(m.group(1))
        chunk_id = f"chunk_{chunk_num:04d}"
        # Verify metadata also exists
        expected_meta = f"{FEATURES_BUCKET_PREFIX}/Metadata_{city}_{chunk_num}.{TOTAL_CHUNKS}.jsonl"
        if expected_meta in meta_keys:
            done_chunks.add(chunk_id)

    if done_chunks:
        print(f"[RESUME] Found {len(done_chunks)} completed chunks on R2")
        reconciled = tq.reconcile_done(REGION, done_chunks)
        if reconciled > 0:
            print(f"[RESUME] Marked {reconciled} chunks as done in Redis (were stale/orphaned)")

    # Reclaim active tasks from dead workers (stale > 5 min)
    stale = tq.reclaim_stale(REGION)
    if stale:
        print(f"[RESUME] Reclaimed {len(stale)} stale tasks: {stale}")

    # Recover any tasks that fell through the cracks (LPOP succeeded but HSET failed)
    lost = tq.recover_lost_tasks(REGION)
    if lost:
        print(f"[RESUME] Recovered {len(lost)} lost tasks: {lost}")

    progress = tq.get_progress(REGION)
    print(f"[RESUME] After reconciliation — todo: {progress['todo']}, "
          f"active: {progress['active']}, done: {progress['done']}/{progress['total_chunks']}")

    return done_chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline — Pipelined Chunk Queue Loop
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_instance_id():
    """Detect Vast.ai instance ID at runtime."""
    global INSTANCE_ID
    if INSTANCE_ID:
        return
    # Try Vast.ai CLI detection
    if VAST_API_KEY:
        try:
            result = subprocess.run(
                ["vastai", "--api-key", VAST_API_KEY, "show", "instances", "--raw"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                instances = json.loads(result.stdout.strip())
                if instances and len(instances) == 1:
                    INSTANCE_ID = str(instances[0].get('id', ''))
        except Exception:
            pass
    # Final fallback: hostname
    if not INSTANCE_ID:
        import socket
        INSTANCE_ID = socket.gethostname()


def main():
    global CITY_NAME, CSV_BUCKET_PREFIX, FEATURES_BUCKET_PREFIX, TOTAL_CHUNKS

    work_dir = Path('/app/work')
    work_dir.mkdir(parents=True, exist_ok=True)

    _detect_instance_id()
    print(f"[INFO] Worker {INSTANCE_ID} starting")
    print(f"[INFO] City: {CITY_NAME}")
    print(f"[INFO] Region: {REGION}")
    print(f"[INFO] CSV prefix: {CSV_BUCKET_PREFIX}")
    print(f"[INFO] Features prefix: {FEATURES_BUCKET_PREFIX}")
    print(f"[INFO] Redis: {REDIS_URL[:40]}...")

    # ── Init R2 client ──
    r2 = R2Client()

    # ── Init Redis task queue ──
    if not REDIS_URL or not REDIS_TOKEN:
        print("[FATAL] REDIS_URL and REDIS_TOKEN must be set")
        sys.exit(1)
    if not REGION:
        print("[FATAL] REGION must be set (e.g. 'AU/Queensland/Brisbane')")
        sys.exit(1)

    tq = TaskQueue(REDIS_URL, REDIS_TOKEN)
    try:
        tq.report_status(REGION, INSTANCE_ID, "STARTING")
    except Exception:
        pass

    # Verify connection and job exists
    progress = tq.get_progress(REGION)
    print(f"[INFO] Job status: {progress}")
    if progress['total_chunks'] == 0:
        print("[FATAL] No job found in Redis for this region")
        sys.exit(1)

    # Set TOTAL_CHUNKS so output filenames include the total
    global TOTAL_CHUNKS
    TOTAL_CHUNKS = progress['total_chunks']
    print(f"[INFO] Total chunks: {TOTAL_CHUNKS} — output naming: {{city}}_N.{TOTAL_CHUNKS}.npy")

    # ── Resume: reconcile Redis state with R2 reality ──
    # Scans R2 for already-uploaded NPY files, marks them done in Redis,
    # reclaims stale tasks from dead workers, recovers lost tasks.
    reconcile_with_r2(r2, tq)

    # Check if job is already fully done after reconciliation
    if tq.is_complete(REGION):
        print("[INFO] Job already complete (all chunks uploaded to R2). Self-destructing.")
        upload_logs_to_r2()
        self_destruct()
        return

    # ── Init GPU extractor(s) ──
    # Single-GPU machines (or DISABLE_MULTI_GPU=1) → one GpuExtractor in this
    # process, legacy behaviour. Multi-GPU machines → spawn a GpuWorkerPool
    # with one child process per device. The extractor variable in this
    # process is unused in pool mode but we keep it None-checked so failures
    # to spawn the pool fall back to single-process.
    try:
        tq.report_status(REGION, INSTANCE_ID, "LOADING_MODEL")
    except Exception:
        pass

    try:
        n_gpus = torch.cuda.device_count()
    except Exception as e:
        print(f"[FATAL] torch.cuda.device_count() failed: {e}", flush=True)
        sys.exit(1)

    disable_multi = os.environ.get('DISABLE_MULTI_GPU', '').strip() in ('1', 'true', 'yes')
    use_pool = (n_gpus > 1) and not disable_multi
    extractor = None
    gpu_pool = None

    if use_pool:
        print(f"[INFO] Multi-GPU detected ({n_gpus} GPUs) — spawning worker pool...",
              flush=True)
        try:
            gpu_pool = GpuWorkerPool(n_gpus=n_gpus)
            print(f"[INFO] Pool active — extraction will run across {n_gpus} GPUs",
                  flush=True)
        except Exception as e:
            print(f"[WARN] GpuWorkerPool init failed: {type(e).__name__}: {e}",
                  flush=True)
            print(f"[WARN] Falling back to single-process extraction "
                  f"(DataParallel across {n_gpus} GPUs)", flush=True)
            gpu_pool = None
            try:
                extractor = GpuExtractor()
            except Exception as e2:
                print(f"[FATAL] Fallback extractor also failed: {e2}", flush=True)
                upload_logs_to_r2()
                sys.exit(1)
    else:
        if n_gpus > 1 and disable_multi:
            print(f"[INFO] {n_gpus} GPUs visible but DISABLE_MULTI_GPU=1 — "
                  f"using single-process path", flush=True)
        else:
            print(f"[INFO] Single GPU ({n_gpus} visible) — using single-process path",
                  flush=True)
        try:
            extractor = GpuExtractor()
        except Exception as e:
            print(f"[FATAL] GpuExtractor init failed: {e}", flush=True)
            upload_logs_to_r2()
            sys.exit(1)

    # ── Pipelined chunk queue loop ──
    # Overlaps: uploads run in a bounded background pool while extraction
    # claims and processes the next chunk. The pool caps at 5 pending — if
    # uploads can't keep up, submit() blocks the extractor (backpressure),
    # so local disk doesn't fill and RSS stays bounded. A chunk is only
    # marked complete in Redis after its upload succeeds; failed uploads
    # mark the chunk failed (and reconcile-on-startup will pick up any
    # R2 objects that landed despite the failure flag).
    chunks_done = 0
    chunks_failed = 0
    idle_cycles = 0
    idle_start_time = None
    MAX_IDLE_CYCLES = 6
    MAX_IDLE_SECONDS = 600  # 10 min hard idle timeout → self-destruct
    skip_city_prefixes = set()  # CSV prefixes with missing R2 files — skip entire city

    upload_mgr = AsyncUploadManager(
        r2,
        max_pending=int(os.environ.get('UPLOAD_MAX_PENDING', '5')),
        num_workers=int(os.environ.get('UPLOAD_WORKERS', '2')),
    )

    # Background heartbeat covering ALL claimed chunks (pool extraction,
    # upload phase, prefetched chunk) — not just the single-GPU loop.
    hb = HeartbeatThread(tq, REGION, INSTANCE_ID)
    hb.start()

    def _handle_completions(completions):
        """Translate finished upload jobs into Redis state. Returns
        (n_done_delta, n_failed_delta) so caller can keep counters."""
        d, f = 0, 0
        for redis_cid, err in completions:
            if err is None:
                if _redis_retry(tq.complete_task, REGION, redis_cid, INSTANCE_ID,
                                label=f"complete_task({redis_cid})"):
                    d += 1
                    print(f"[INFO] Completed chunk {redis_cid} "
                          f"(running total +{d} this batch)")
                else:
                    # Data is safely on R2 — reconcile-on-restart will tidy up.
                    print(f"[WARN] Could not mark {redis_cid} done in Redis "
                          "(NPY uploaded; reconcile will fix)")
                    d += 1
            else:
                msg = f"{type(err).__name__}: {str(err)[:200]}"
                print(f"[ERROR] Upload for {redis_cid} failed: {msg}")
                # Upload failure is an infra problem, not a content problem —
                # the extracted features were fine. Unclaim (no fcnt bump,
                # capped by the unclaim counter) so a possibly-healthier
                # worker retries; local files were already cleaned by the
                # upload worker.
                try:
                    tq.unclaim_task(REGION, redis_cid, INSTANCE_ID,
                                    reason=f"upload_failed: {msg}", back=True)
                except Exception as re:
                    print(f"[WARN] unclaim_task({redis_cid}) failed: {re}")
                f += 1
            hb.unregister(redis_cid)
        return d, f

    # Pipeline state
    prefetched = None       # (chunk_id, local_csv, records, metadata_map[, batch_meta])
    # Shared download→extract queue: ONE queue across all chunks so an
    # overlapped next-chunk downloader can feed it while the current chunk's
    # straggler tail drains (single-GPU path only; pool mode ignores it).
    shared_item_queue = queue.Queue(maxsize=HARDCODED_CONFIG['queue_size'])
    live_handoff = None     # _ChunkRun whose downloader is already running

    while True:
        # ── Drain any uploads that finished while we were extracting ──
        # Non-blocking — just translates completions into Redis state. The
        # extractor doesn't wait on uploads here; backpressure happens later
        # at upload_mgr.submit() if the queue is full.
        d, f = _handle_completions(upload_mgr.drain_completions())
        chunks_done += d
        chunks_failed += f

        # ── Get next chunk: live handoff, prefetch, or fresh claim ──
        _bmeta = None  # batch metadata for current chunk
        cur_handoff = None
        if live_handoff is not None:
            # The previous process_chunk already started this chunk's
            # downloader (cross-chunk overlap). Its files/sink exist; the
            # R2 skip check was done at overlap start.
            cur_handoff = live_handoff
            live_handoff = None
            chunk_id = cur_handoff.redis_cid
            _bmeta = cur_handoff.bmeta
            preloaded_data = None
        elif prefetched is not None:
            chunk_id = prefetched[0]
            pf_csv, pf_records, pf_meta = prefetched[1], prefetched[2], prefetched[3]
            _bmeta = prefetched[4] if len(prefetched) > 4 else None
            preloaded_data = (pf_csv, pf_records, pf_meta)
            prefetched = None
        else:
            try:
                chunk_id = tq.claim_task(REGION, INSTANCE_ID)
            except Exception as e:
                print(f"[WARN] Redis claim_task failed: {e} — retrying in 10s")
                time.sleep(10)
                continue

            if chunk_id is None:
                try:
                    if tq.is_complete(REGION):
                        print(f"[INFO] Job complete! Processed {chunks_done} chunks. Self-destructing.")
                        break
                except Exception as e:
                    print(f"[WARN] Redis is_complete check failed: {e}")

                idle_cycles += 1
                if idle_start_time is None:
                    idle_start_time = time.time()
                try:
                    tq.report_status(REGION, INSTANCE_ID, "IDLE",
                                     chunks_done=chunks_done)
                except Exception:
                    pass

                # Periodically reclaim stale tasks from dead/stuck workers
                if idle_cycles % MAX_IDLE_CYCLES == 0:
                    try:
                        reclaimed = tq.reclaim_stale(REGION)
                        if reclaimed:
                            print(f"[IDLE] Reclaimed {len(reclaimed)} stale tasks: {reclaimed}")
                    except Exception:
                        pass
                    try:
                        if tq.is_complete(REGION):
                            print(f"[INFO] Job complete after idle wait. Self-destructing.")
                            break
                    except Exception:
                        pass

                # Periodically recover lost tasks (claimed via LPOP but never
                # marked active — worker crashed between the two calls).
                # Startup-only recovery misses chunks lost mid-run; without
                # this the fleet idles forever on a job that isn't complete.
                if idle_cycles % 10 == 0:
                    try:
                        lost = tq.recover_lost_tasks(REGION)
                        if lost:
                            print(f"[IDLE] Recovered {len(lost)} lost tasks: {lost}")
                    except Exception:
                        pass

                # Hard idle timeout — self-destruct if idle too long
                idle_elapsed = time.time() - idle_start_time
                if idle_elapsed >= MAX_IDLE_SECONDS:
                    print(f"[INFO] Idle for {idle_elapsed:.0f}s with no work. Self-destructing.")
                    break
                time.sleep(5)  # Reduced from 30s for faster queue response
                continue

            # Keep the freshly claimed chunk alive in Redis through pool
            # extraction / upload phases (prefetched chunks are registered
            # inside _do_prefetch).
            hb.register(chunk_id)

            preloaded_data = None

            # Look up batch metadata for freshly claimed chunk
            try:
                _bmeta = tq.get_chunk_meta(REGION, chunk_id)
            except Exception:
                pass

        idle_cycles = 0
        idle_start_time = None

        # ── Batch mode: override globals with per-chunk paths ──
        _redis_cid = chunk_id  # preserve original for Redis ops
        if _bmeta:
            CITY_NAME = _bmeta['city_name']
            CSV_BUCKET_PREFIX = _bmeta['csv_prefix']
            FEATURES_BUCKET_PREFIX = _bmeta['features_prefix']
            TOTAL_CHUNKS = _bmeta['city_total']
            chunk_id = f"chunk_{_bmeta['chunk_num']:04d}"  # local file chunk ID
            print(f"[BATCH] {_redis_cid} → {CITY_NAME} {chunk_id} "
                  f"(csv={CSV_BUCKET_PREFIX}, feat={FEATURES_BUCKET_PREFIX})")

            # Skip entire city if its CSV prefix previously 404'd. NEVER for
            # a handed-off chunk: its CSV already downloaded fine and its
            # downloader is running — bailing here would leak the run.
            if CSV_BUCKET_PREFIX in skip_city_prefixes and cur_handoff is None:
                print(f"[SKIP-CITY] {CITY_NAME} — CSVs missing on R2, skipping {_redis_cid}")
                try:
                    tq.fail_task(REGION, _redis_cid, INSTANCE_ID,
                                 f"city_csv_missing:{CSV_BUCKET_PREFIX}")
                except Exception:
                    pass
                hb.unregister(_redis_cid)
                prefetched = None
                continue

        # ── Skip if output already exists on R2 (avoids re-extracting) ──
        # Requires BOTH the NPY and its metadata JSONL with size > 0,
        # mirroring the startup reconcile — an NPY whose metadata upload
        # failed (or a zero-byte torn object) must be re-extracted, not
        # marked done. Handed-off chunks were checked at overlap start.
        if cur_handoff is None:
            out_base = _output_base(CITY_NAME, chunk_id)
            npy_key = f"{FEATURES_BUCKET_PREFIX}/{out_base}.npy"
            meta_skip_key = f"{FEATURES_BUCKET_PREFIX}/Metadata_{out_base}.jsonl"
            try:
                if r2.object_size(npy_key) > 0 and r2.object_size(meta_skip_key) > 0:
                    print(f"[SKIP] Chunk {chunk_id} output (NPY + metadata) already "
                          f"on R2 — marking done")
                    try:
                        tq.complete_task(REGION, _redis_cid, INSTANCE_ID)
                    except Exception:
                        pass
                    hb.unregister(_redis_cid)
                    chunks_done += 1
                    prefetched = None
                    continue
            except Exception:
                pass  # If check fails, process normally

        print(f"\n{'='*60}")
        print(f"[INFO] "
              f"{'Overlapped' if cur_handoff is not None else ('Pre-fetched' if preloaded_data else 'Claimed')} "
              f"chunk: {chunk_id} (completed so far: {chunks_done})")
        print(f"{'='*60}")

        # ── Prefetch next chunk's CSV in background during extraction ──
        pf_result = [None]
        pf_thread = threading.Thread(
            target=_do_prefetch,
            args=(pf_result, r2, tq, work_dir, skip_city_prefixes, hb),
            daemon=True,
        )
        pf_thread.start()

        # ── Process current chunk ──
        handoff_slot = [None]
        try:
            result = process_chunk(r2, tq, extractor, chunk_id, work_dir,
                                   preloaded=preloaded_data,
                                   chunks_done_so_far=chunks_done,
                                   redis_chunk_id=_redis_cid,
                                   gpu_pool=gpu_pool,
                                   item_queue=shared_item_queue,
                                   overlap_pf=pf_result,
                                   handoff=cur_handoff,
                                   handoff_out=handoff_slot,
                                   hb=hb)
            # If the straggler-tail overlap kicked in, the next chunk's
            # downloader is already running — carry it into the next
            # iteration instead of consuming the (already-spent) prefetch.
            live_handoff = handoff_slot[0]

            # Wait for prefetch to finish before starting upload
            pf_thread.join()
            prefetched = pf_result[0]

            if result is None:
                # Empty chunk — mark complete immediately
                tq.complete_task(REGION, _redis_cid, INSTANCE_ID)
                hb.unregister(_redis_cid)
                chunks_done += 1
                print(f"[INFO] Completed empty chunk {chunk_id} ({chunks_done} total)")
            else:
                features_file, metadata_file, local_csv = result

                # Snapshot the per-chunk paths HERE on the main thread,
                # before batch-mode globals can be overwritten by the next
                # claim. Reading them inside the upload worker risked
                # uploading this chunk's bytes under the NEXT chunk's city
                # prefix (silent cross-city contamination).
                _ul_out_base = _output_base(CITY_NAME, chunk_id)
                _ul_npy_key = f"{FEATURES_BUCKET_PREFIX}/{_ul_out_base}.npy"
                _ul_meta_key = f"{FEATURES_BUCKET_PREFIX}/Metadata_{_ul_out_base}.jsonl"

                job = _UploadJob(
                    redis_chunk_id=_redis_cid, chunk_id=chunk_id,
                    features_file=features_file, metadata_file=metadata_file,
                    local_csv=local_csv, work_dir=work_dir,
                    npy_key=_ul_npy_key, meta_key=_ul_meta_key,
                    out_base=_ul_out_base,
                )

                # ── Backpressure ──
                # If the pool already has max_pending uploads in flight,
                # submit() blocks until a slot frees. The extractor pauses
                # here rather than letting local disk fill. We deliberately
                # drain completions WHILE waiting so Redis state stays
                # current and so multi-pending failures unblock the
                # extractor as soon as one finishes.
                if upload_mgr.pending_count() >= upload_mgr.max_pending:
                    print(f"[BACKPRESSURE] {upload_mgr.pending_count()} pending uploads — "
                          f"pausing extraction until a slot frees", flush=True)
                    try:
                        tq.report_status(REGION, INSTANCE_ID, "UPLOAD_BACKPRESSURE",
                                         chunk_id=_redis_cid,
                                         chunks_done=chunks_done)
                    except Exception:
                        pass
                    # Drain at least one completion before submitting.
                    while upload_mgr.pending_count() >= upload_mgr.max_pending:
                        completions = upload_mgr.wait_one_completion(timeout=30)
                        d, f = _handle_completions(completions)
                        chunks_done += d
                        chunks_failed += f
                upload_mgr.submit(job)

                try:
                    tq.report_status(REGION, INSTANCE_ID, "UPLOADING",
                                     chunk_id=_redis_cid, chunks_done=chunks_done)
                except Exception:
                    pass
                # Loop immediately → claim + extract the next chunk while
                # this one's upload runs in the background pool.

        except IPBlockedError as e:
            # Google has 403-blocked this instance's IP. Do NOT fail_task
            # (that would poison the fcnt for a chunk that's still perfectly
            # good). Unclaim the chunk + also any prefetched chunk, then
            # self-destruct so the fleet orchestrator can relaunch on a new IP.
            print(f"[IP-BLOCK] {e}", flush=True)
            try:
                tq.unclaim_task(REGION, _redis_cid, INSTANCE_ID,
                                reason="ip_blocked_403")
            except Exception as re:
                print(f"[WARN] Failed to unclaim {_redis_cid}: {re}")
            hb.unregister(_redis_cid)
            _cleanup_chunk_files(work_dir, chunk_id)

            # Also unclaim the prefetched chunk if one is pending — same IP,
            # same problem.
            pf_thread.join()
            pf_result_val = pf_result[0]
            if pf_result_val is not None:
                try:
                    pf_next_id = pf_result_val[0]
                    tq.unclaim_task(REGION, pf_next_id, INSTANCE_ID,
                                    reason="ip_blocked_403_prefetch")
                    hb.unregister(pf_next_id)
                    # Delete the prefetched CSV since we're bailing
                    try:
                        os.remove(pf_result_val[1])
                    except OSError:
                        pass
                except Exception as re:
                    print(f"[WARN] Failed to unclaim prefetched chunk: {re}")

            # Drain pending uploads before dying so we don't lose a chunk
            # whose extraction already succeeded. Failed uploads at this
            # point still fail_task in Redis so reconcile can recover.
            try:
                final_completions = upload_mgr.shutdown(drain=True, timeout=600)
                d, f = _handle_completions(final_completions)
                chunks_done += d
                chunks_failed += f
            except Exception as ue:
                print(f"[WARN] Upload manager shutdown during IP-block failed: {ue}",
                      flush=True)

            # Clean GPU pool teardown so child processes don't outlive the
            # parent (Vast can re-use the container; orphaned CUDA contexts
            # would leak VRAM on a relaunch attempt).
            if gpu_pool is not None:
                try:
                    gpu_pool.shutdown(drain_timeout=30.0)
                except Exception as gpe:
                    print(f"[WARN] gpu_pool.shutdown on IP-block failed: {gpe}",
                          flush=True)

            print("[IP-BLOCK] Self-destructing to release this IP.", flush=True)
            upload_logs_to_r2()
            self_destruct()
            return

        except Exception as e:
            chunks_failed += 1
            error_msg = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"[ERROR] Chunk {chunk_id} failed: {error_msg}")
            import traceback
            traceback.print_exc()

            # ── Classify the failure ──
            # Content failures (zero features extracted) → fail_task: the
            # fcnt 3-strike is right for a chunk that genuinely produces
            # nothing. Infra failures (CSV download, disk, stall, pool
            # death) → unclaim_task(back=True): no fcnt bump, ucnt-capped,
            # so a healthier worker retries without the chunk getting
            # permanently failed for this worker's misfortune.
            err_str = str(e)
            csv_dl_failed = "Failed to download" in err_str
            content_failure = "Zero features extracted" in err_str

            if csv_dl_failed and CSV_BUCKET_PREFIX:
                # Blacklist the city ONLY on a real 404/NoSuchKey — R2
                # having a bad minute must not permanently fail every
                # chunk of a city.
                csv_key = f"{CSV_BUCKET_PREFIX}/{CITY_NAME}_{chunk_id}.csv"
                missing = None
                try:
                    missing = r2.object_missing(csv_key)
                except Exception:
                    pass
                if missing is True:
                    skip_city_prefixes.add(CSV_BUCKET_PREFIX)
                    print(f"[SKIP-CITY] CSV {csv_key} missing on R2 (404) — "
                          f"blacklisting prefix '{CSV_BUCKET_PREFIX}', all "
                          f"future chunks from {CITY_NAME} will be skipped")
                    try:
                        tq.fail_task(REGION, _redis_cid, INSTANCE_ID, error_msg)
                    except Exception as re:
                        print(f"[WARN] Failed to return chunk to queue: {re}")
                else:
                    print(f"[WARN] CSV download failed for {chunk_id} "
                          f"(transient) — unclaiming, NOT blacklisting")
                    try:
                        tq.unclaim_task(
                            REGION, _redis_cid, INSTANCE_ID,
                            reason=f"csv_download_transient: {error_msg}",
                            back=True)
                    except Exception as re:
                        print(f"[WARN] Failed to unclaim chunk: {re}")
            elif content_failure:
                try:
                    tq.fail_task(REGION, _redis_cid, INSTANCE_ID, error_msg)
                except Exception as re:
                    print(f"[WARN] Failed to return chunk to queue: {re}")
            else:
                try:
                    tq.unclaim_task(REGION, _redis_cid, INSTANCE_ID,
                                    reason=f"infra: {error_msg}", back=True)
                except Exception as re:
                    print(f"[WARN] Failed to unclaim chunk: {re}")
            hb.unregister(_redis_cid)
            _cleanup_chunk_files(work_dir, chunk_id)

            # Still collect prefetch result
            pf_thread.join()
            prefetched = pf_result[0]

        # Force GC between chunks
        gc.collect()
        torch.cuda.empty_cache()

    # ── Wait for all in-flight uploads before exit ──
    # Block on the bounded upload pool. Anything that finishes here gets
    # marked complete/failed in Redis. We don't kill workers until every
    # job is drained — losing a finished extraction would force a re-scrape.
    print(f"[INFO] Waiting for {upload_mgr.pending_count()} in-flight upload(s) "
          f"to complete before exit...")
    final_completions = upload_mgr.shutdown(drain=True, timeout=900)
    d, f = _handle_completions(final_completions)
    chunks_done += d
    chunks_failed += f
    final_stats = upload_mgr.stats()
    print(f"[INFO] Upload manager final: lifetime completed={final_stats['completed']}, "
          f"failed={final_stats['failed']}")

    # Tear down the GPU pool (if any) before self-destruct so child
    # processes get a clean STOP and don't leak CUDA contexts.
    if gpu_pool is not None:
        try:
            gpu_pool.shutdown(drain_timeout=60.0)
        except Exception as e:
            print(f"[WARN] gpu_pool.shutdown failed: {e}", flush=True)

    # ── Done — self-destruct ──
    try:
        final_progress = tq.get_progress(REGION)
        perm_failed = final_progress.get('failed', 0)
    except Exception:
        perm_failed = 0
    print(f"\n[INFO] Final stats: {chunks_done} chunks completed, {chunks_failed} failed this session, {perm_failed} permanently failed")
    try:
        tq.report_status(REGION, INSTANCE_ID, "DONE", chunks_done=chunks_done)
    except Exception:
        pass
    upload_logs_to_r2()
    self_destruct()


if __name__ == '__main__':
    _log_fh = _start_log_capture()
    try:
        main()
    except Exception as e:
        error_type = type(e).__name__
        print(f"[CRITICAL] {error_type}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        upload_logs_to_r2()
        sys.exit(1)
