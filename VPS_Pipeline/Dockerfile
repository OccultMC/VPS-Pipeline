FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 curl dos2unix build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install vastai CLI for self-destruct
RUN pip install --no-cache-dir vastai

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache MegaLoc architecture (hubconf.py + model code)
RUN python -c "import torch; torch.hub._get_cache_or_reload('gmberton/MegaLoc', force_reload=False, trust_repo=True, calling_fn='load')"

# Download and embed MegaLoc model weights into the image
# Uses BuildKit secret so the HF token never leaks into image layers
RUN --mount=type=secret,id=HF_TOKEN \
    python -c "\
token = open('/run/secrets/HF_TOKEN').read().strip(); \
from huggingface_hub import hf_hub_download; \
hf_hub_download('gmberton/MegaLoc', 'model.safetensors', local_dir='/app/models/megaloc', token=token)" && \
    ls -lh /app/models/megaloc/model.safetensors

# Application code
COPY pipeline.py r2_storage.py entrypoint.sh ./
COPY gsvpd/ ./gsvpd/

RUN dos2unix entrypoint.sh && chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
