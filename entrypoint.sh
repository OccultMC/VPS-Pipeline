#!/bin/bash
set -e

date -u
echo "=== Hypervision VPS Pipeline ==="
echo "Instance: ${INSTANCE_ID}"
echo "City: ${CITY_NAME}"
echo "Region: ${REGION}"
echo "Redis: ${REDIS_URL:0:40}..."

# ── GPU diagnostics ──
echo "── GPU Diagnostics ──"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')
    GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'N/A')
    GPU_CUDA=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || echo 'N/A')
    echo "GPU: ${GPU_NAME}"
    echo "VRAM: ${GPU_MEM}"
    echo "Driver: ${GPU_DRIVER}"
    echo "Compute Cap: ${GPU_CUDA}"
else
    echo "[ERROR] nvidia-smi not found! GPU may not be available."
fi

echo "── System Info ──"
echo "Disk: $(df -h / | tail -1 | awk '{print $4}') free"
echo "RAM: $(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo 'N/A') total"
echo "CPU: $(nproc) cores"
echo "Python: $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "CUDA (PyTorch): $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"
echo "Working Directory: $(pwd)"
echo "Files in /app:"
ls -la /app
echo "================================"

# Instance ID detection is handled by pipeline.py via R2 lookup.
if [ -z "${INSTANCE_ID}" ]; then
    echo "[INFO] INSTANCE_ID not set — pipeline.py will detect via R2"
fi

# Check MegaLoc model sources
MODEL_FILE="/app/models/megaloc/model.safetensors"
HUB_CACHE="/root/.cache/torch/hub/gmberton_MegaLoc_main"
echo "── MegaLoc Model ──"
echo "Primary: torch.hub get_trained_model (HuggingFace, same as inference server)"
if [ -f "$MODEL_FILE" ]; then
    MODEL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo "Fallback: baked model at $MODEL_FILE ($MODEL_SIZE)"
else
    echo "Fallback: no baked model (hub download required)"
fi
if [ -d "$HUB_CACHE" ]; then
    echo "[OK] MegaLoc architecture cached at $HUB_CACHE"
else
    echo "[INFO] MegaLoc architecture not cached — will download from GitHub"
fi

# Fix for PyTorch 2.x compile (inductor) missing libcuda.so
# Find where libcuda.so.1 is (mounted by nvidia-runtime) and symlink libcuda.so
echo "[INFO] Checking for libcuda.so..."
LIBCUDA_PATH=$(ldconfig -p | grep libcuda.so.1 | head -n 1 | awk '{print $4}')
if [ -n "$LIBCUDA_PATH" ]; then
    DIR=$(dirname "$LIBCUDA_PATH")
    if [ ! -f "$DIR/libcuda.so" ]; then
        echo "[INFO] Creating libcuda.so symlink at $DIR/libcuda.so -> $LIBCUDA_PATH"
        ln -s "$LIBCUDA_PATH" "$DIR/libcuda.so"
    else
        echo "[INFO] libcuda.so already exists at $DIR/libcuda.so"
    fi
else
    echo "[WARN] Could not find libcuda.so.1 in ldconfig cache. Compilation might fail."
    # Try common location fallback
    if [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ] && [ ! -f /usr/lib/x86_64-linux-gnu/libcuda.so ]; then
         echo "[INFO] Fallback: Creating symlink in /usr/lib/x86_64-linux-gnu/"
         ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so || true
    fi
fi

# Suppress Vast.ai SSH port-forwarding noise from polluting pipeline logs
# These spam "Warning: Permanently added" + "remote port forwarding failed" every 2s
export VAST_CONTAINERLABEL=${VAST_CONTAINERLABEL:-}
(
    # Kill the noisy ssh port-forward retry loop if it exists
    # Redirect any remaining sshd/ssh stderr to /dev/null
    for pid in $(pgrep -f "ssh.*vast.ai" 2>/dev/null); do
        kill "$pid" 2>/dev/null || true
    done
) &>/dev/null || true

# Run the pipeline — filter out SSH noise from stdout/stderr.
# --line-buffered so progress lines reach `vastai logs` as they happen
# instead of in 4KB block-buffered bursts.
run_pipeline() {
    python pipeline.py 2>&1 | grep --line-buffered -v -E "^(Warning: Permanently added|Error: remote port forwarding|kex_exchange_identification|Connection closed by|Connection from|banner exchange|ssh: Could not resolve)"
    return "${PIPESTATUS[0]}"
}

# Self-destruct this Vast.ai instance. Matches pipeline.py's self_destruct():
# vastai CLI, plural "destroy instances" form (singular prompts [y/N] and
# exits 0 without destroying on empty stdin), success confirmed by the
# "destroying instance N." stdout pattern. Falls back to the raw REST API
# (DELETE /api/v0/instances/{id}/) if the CLI is unavailable or unconfirmed.
self_destruct() {
    if [ -z "${INSTANCE_ID}" ] || [ -z "${VAST_API_KEY}" ]; then
        echo "[ERROR] Cannot self-destruct: INSTANCE_ID or VAST_API_KEY not set"
        return 1
    fi
    local attempt
    for attempt in 1 2 3 4 5; do
        echo "[INFO] Self-destruct attempt ${attempt} for instance ${INSTANCE_ID}..."
        if command -v vastai &>/dev/null; then
            local out
            out=$(echo "y" | vastai --api-key "${VAST_API_KEY}" destroy instances "${INSTANCE_ID}" 2>&1) || true
            echo "[INFO] Self-destruct response: ${out}"
            if echo "${out}" | grep -qi "destroying instance ${INSTANCE_ID}"; then
                echo "[INFO] Instance ${INSTANCE_ID} destroyed successfully."
                return 0
            fi
        fi
        local code
        code=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE \
            "https://console.vast.ai/api/v0/instances/${INSTANCE_ID}/" \
            -H "Authorization: Bearer ${VAST_API_KEY}") || true
        echo "[INFO] Vast API DELETE returned HTTP ${code}"
        if [ "${code}" = "200" ]; then
            return 0
        fi
        echo "[WARN] Self-destruct attempt ${attempt} did not confirm — retrying in 30s"
        sleep 30
    done
    return 1
}

MAX_ATTEMPTS=3
EXIT_CODE=0
for ATTEMPT in $(seq 1 ${MAX_ATTEMPTS}); do
    echo "[INFO] Starting pipeline.py at $(date -u) (attempt ${ATTEMPT}/${MAX_ATTEMPTS})..."
    set +e
    run_pipeline
    EXIT_CODE=$?
    set -e
    if [ "${EXIT_CODE}" -eq 0 ]; then
        break
    fi
    echo "[ERROR] Pipeline exited with code ${EXIT_CODE} at $(date -u)"
    if [ "${ATTEMPT}" -lt "${MAX_ATTEMPTS}" ]; then
        echo "[INFO] Retrying in 30s..."
        sleep 30
    fi
done

if [ "${EXIT_CODE}" -ne 0 ]; then
    if [ "${DEBUG_KEEP_ALIVE:-0}" = "1" ]; then
        echo "[INFO] DEBUG_KEEP_ALIVE=1 — container kept alive for debugging. SSH in to investigate."
        tail -f /dev/null
    fi
    echo "[ERROR] Pipeline failed ${MAX_ATTEMPTS} times — self-destructing so the instance stops billing."
    self_destruct || echo "[ERROR] Self-destruct failed — exiting anyway (do NOT idle on a paid GPU)"
    exit "${EXIT_CODE}"
fi
