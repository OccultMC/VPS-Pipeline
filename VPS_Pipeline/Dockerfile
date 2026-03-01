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

# MegaLoc model weights - downloaded from R2 by CI workflow, copied into image
COPY models/megaloc/model.safetensors /app/models/megaloc/model.safetensors

# Application code
COPY pipeline.py r2_storage.py entrypoint.sh ./
COPY gsvpd/ ./gsvpd/

RUN dos2unix entrypoint.sh && chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
