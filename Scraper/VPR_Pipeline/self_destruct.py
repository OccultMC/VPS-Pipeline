"""Self-destruct: destroy this Vast.AI instance after work is done."""

import logging
import os

import aiohttp

logger = logging.getLogger(__name__)

VASTAI_BASE_URL = "https://console.vast.ai"


async def self_destruct():
    """Destroy the current Vast.AI instance using env vars."""
    instance_id = os.environ.get("VASTAI_INSTANCE_ID")
    api_key = os.environ.get("VASTAI_API_KEY")

    if not instance_id or not api_key:
        logger.warning("VASTAI_INSTANCE_ID or VASTAI_API_KEY not set — skipping self-destruct")
        return False

    url = f"{VASTAI_BASE_URL}/api/v0/instances/{instance_id}/"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.put(
                url,
                json={"delete": True},
                params={"api_key": api_key},
            ) as resp:
                if resp.status == 200:
                    logger.info(f"Self-destruct successful (instance {instance_id})")
                    return True
                text = await resp.text()
                logger.error(f"Self-destruct failed ({resp.status}): {text}")
                return False
    except Exception as e:
        logger.error(f"Self-destruct error: {e}")
        return False
