#!/bin/bash
set -e

# ── Brisbane AEST (UTC+10) timestamps ──
export TZ="Australia/Brisbane"

ts() { date "+%Y-%m-%d %H:%M:%S AEST"; }

echo "[$(ts)] === Hypervision VPS Pipeline ==="
echo "[$(ts)] Worker: ${WORKER_INDEX}/${NUM_WORKERS}"
echo "[$(ts)] City: ${CITY_NAME}"
echo "[$(ts)] Instance: ${INSTANCE_ID}"

# ── GPU diagnostics ──
echo "[$(ts)] ── GPU Diagnostics ──"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')
    GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'N/A')
    GPU_CUDA=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || echo 'N/A')
    echo "[$(ts)] GPU: ${GPU_NAME} | VRAM: ${GPU_MEM} | Driver: ${GPU_DRIVER} | Compute: ${GPU_CUDA}"
else
    echo "[$(ts)] [ERROR] nvidia-smi not found! GPU may not be available."
fi

echo "[$(ts)] ── System Info ──"
echo "[$(ts)] Disk: $(df -h / | tail -1 | awk '{print $4}') free | RAM: $(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo 'N/A') | CPU: $(nproc) cores"
echo "[$(ts)] Python: $(python --version 2>&1) | PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A') | CUDA: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"

# Instance ID detection is handled by pipeline.py via R2 lookup.
if [ -z "${INSTANCE_ID}" ]; then
    echo "[$(ts)] [INFO] INSTANCE_ID not set — pipeline.py will detect via R2"
fi

# Verify MegaLoc architecture is baked into the image
HUB_CACHE="/root/.cache/torch/hub/gmberton_MegaLoc_main"
if [ -d "$HUB_CACHE" ]; then
    echo "[$(ts)] [OK] MegaLoc architecture cached at $HUB_CACHE"
else
    echo "[$(ts)] [WARN] MegaLoc architecture NOT cached! Model download may be slower."
    echo "[$(ts)] [WARN] Rebuild Docker image to bake in architecture."
fi

# Fix for PyTorch 2.x compile (inductor) missing libcuda.so
echo "[$(ts)] [INFO] Checking for libcuda.so..."
LIBCUDA_PATH=$(ldconfig -p | grep libcuda.so.1 | head -n 1 | awk '{print $4}')
if [ -n "$LIBCUDA_PATH" ]; then
    DIR=$(dirname "$LIBCUDA_PATH")
    if [ ! -f "$DIR/libcuda.so" ]; then
        echo "[$(ts)] [INFO] Creating libcuda.so symlink at $DIR/libcuda.so -> $LIBCUDA_PATH"
        ln -s "$LIBCUDA_PATH" "$DIR/libcuda.so"
    else
        echo "[$(ts)] [INFO] libcuda.so already exists at $DIR/libcuda.so"
    fi
else
    echo "[$(ts)] [WARN] Could not find libcuda.so.1 in ldconfig cache."
    if [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ] && [ ! -f /usr/lib/x86_64-linux-gnu/libcuda.so ]; then
         echo "[$(ts)] [INFO] Fallback: Creating symlink in /usr/lib/x86_64-linux-gnu/"
         ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so || true
    fi
fi

# Run the pipeline
echo "[$(ts)] [INFO] Starting pipeline.py..."
python pipeline.py
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(ts)] [ERROR] Pipeline exited with code $EXIT_CODE"
    echo "[$(ts)] [INFO] Container kept alive for debugging. SSH in to investigate."
    tail -f /dev/null
fi
