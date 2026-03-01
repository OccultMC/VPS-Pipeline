FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Symlink python -> python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Python deps (install torch first for caching)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Model weights (baked into image — downloaded in CI step)
COPY model.safetensors /app/model.safetensors

# Application code
COPY constants.py megaloc_model.py directional_views.py tile_fetcher.py \
     producer.py consumer.py pipeline_logger.py r2_client.py \
     self_destruct.py main.py ./

# Non-root user
RUN useradd -m worker
USER worker

ENTRYPOINT ["python", "main.py"]
