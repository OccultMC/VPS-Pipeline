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

# Verify MegaLoc model is baked into the image
MODEL_FILE="/app/models/megaloc/model.safetensors"
HUB_CACHE="/root/.cache/torch/hub/gmberton_MegaLoc_main"
if [ -f "$MODEL_FILE" ]; then
    MODEL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo "[$(ts)] [OK] MegaLoc model baked in: $MODEL_FILE ($MODEL_SIZE)"
else
    echo "[$(ts)] [ERROR] MegaLoc model NOT found at $MODEL_FILE! Rebuild Docker image."
fi
if [ -d "$HUB_CACHE" ]; then
    echo "[$(ts)] [OK] MegaLoc architecture cached at $HUB_CACHE"
else
    echo "[$(ts)] [ERROR] MegaLoc architecture NOT cached! Rebuild Docker image."
fi

# ── Pre-flight: verify torch actually imports ──
echo "[$(ts)] [INFO] Pre-flight: testing PyTorch import..."
if ! python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'; print(f'OK: PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')" 2>/tmp/torch_check.log; then
    echo "[$(ts)] [FATAL] PyTorch/CUDA pre-flight FAILED. This machine is broken."
    cat /tmp/torch_check.log 2>/dev/null
    echo "[$(ts)] [INFO] Reporting FAILED status to R2 and self-destructing..."
    python -c "
import os, json, time
try:
    from r2_storage import R2Client
    r2 = R2Client(os.environ['R2_ACCOUNT_ID'], os.environ['R2_ACCESS_KEY_ID'],
                   os.environ['R2_SECRET_ACCESS_KEY'], os.environ['R2_BUCKET_NAME'])
    city = os.environ.get('CITY_NAME', 'Unknown')
    iid = os.environ.get('INSTANCE_ID', '')
    if iid:
        key = f'Status/{city}_{iid}.json'
    else:
        widx = os.environ.get('WORKER_INDEX', '0')
        nw = os.environ.get('NUM_WORKERS', '1')
        key = f'Status/{city}_worker{widx}.json'
    r2.upload_json(key, {'s': 'FAILED_PREFLIGHT', 'p': 0, 't': 0, 'eta': 0, 'spd': 0,
                         'iid': iid, 'ts': time.time()})
    print(f'Reported FAILED_PREFLIGHT to {key}')
except Exception as e:
    print(f'Could not report status: {e}')
try:
    import subprocess
    vast_key = os.environ.get('VAST_API_KEY', '')
    if vast_key:
        subprocess.run(['vastai', 'destroy', 'instance', os.environ.get('CONTAINER_ID', ''),
                        '--api-key', vast_key], timeout=30)
except Exception:
    pass
" 2>&1 || true
    echo "[$(ts)] [INFO] Container kept alive for debugging. SSH in to investigate."
    tail -f /dev/null
    exit 1
fi
echo "[$(ts)] [OK] Pre-flight passed"

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
    # Report FAILED to R2 so the monitor picks it up
    python -c "
import os, json, time
try:
    from r2_storage import R2Client
    r2 = R2Client(os.environ['R2_ACCOUNT_ID'], os.environ['R2_ACCESS_KEY_ID'],
                   os.environ['R2_SECRET_ACCESS_KEY'], os.environ['R2_BUCKET_NAME'])
    city = os.environ.get('CITY_NAME', 'Unknown')
    iid = os.environ.get('INSTANCE_ID', '')
    if iid:
        key = f'Status/{city}_{iid}.json'
    else:
        widx = os.environ.get('WORKER_INDEX', '0')
        nw = os.environ.get('NUM_WORKERS', '1')
        key = f'Status/{city}_worker{widx}.json'
    r2.upload_json(key, {'s': 'FAILED_CRASH', 'p': 0, 't': 0, 'eta': 0, 'spd': 0,
                         'iid': iid, 'ts': time.time()})
    print(f'Reported FAILED_CRASH to {key}')
except Exception as e:
    print(f'Could not report status: {e}')
" 2>&1 || true
    echo "[$(ts)] [INFO] Container kept alive for debugging. SSH in to investigate."
    tail -f /dev/null
fi
