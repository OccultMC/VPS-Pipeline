"""Self-destruct: destroy this Vast.AI instance after work is done."""

import logging
import os

import requests

logger = logging.getLogger(__name__)

API_BASE = "https://console.vast.ai/api/v0"


def destroy_self():
    """Destroy this Vast.AI instance using VASTAI_API_KEY and VASTAI_INSTANCE_ID."""
    api_key = os.environ.get("VASTAI_API_KEY")
    instance_id = os.environ.get("VASTAI_INSTANCE_ID") or os.environ.get("CONTAINER_ID")

    if not api_key or not instance_id:
        logger.warning("Cannot self-destruct: missing VASTAI_API_KEY or VASTAI_INSTANCE_ID")
        return False

    url = f"{API_BASE}/instances/{instance_id}/"
    try:
        resp = requests.put(url, json={"delete": True}, params={"api_key": api_key}, timeout=30)
        if resp.status_code == 200:
            logger.info(f"Self-destruct initiated for instance {instance_id}")
            return True
        else:
            logger.error(f"Self-destruct failed ({resp.status_code}): {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Self-destruct error: {e}")
        return False
