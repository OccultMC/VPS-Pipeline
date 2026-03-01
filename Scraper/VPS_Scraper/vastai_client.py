"""Vast.AI API client for searching offers, creating instances, and destroying them."""

import logging
from typing import List, Dict, Any, Optional

import aiohttp

logger = logging.getLogger(__name__)

VASTAI_BASE_URL = "https://console.vast.ai"


class VastAIClient:
    """Async Vast.AI API wrapper."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def search_gpu_offers(self, num_results: int = 20) -> List[Dict]:
        """Search for GPU offers matching pipeline requirements (Section 5.1)."""
        session = await self._get_session()

        payload = {
            "select_cols": [
                "id", "gpu_name", "num_gpus", "dph_total", "cuda_max_good",
                "inet_down", "inet_up", "cpu_ram", "disk_space",
                "reliability", "verified", "geolocation", "datacenter"
            ],
            "order": [["dph_total", "asc"]],
            "limit": num_results,
            "type": "on-demand",
            "allocated_storage": 100,
            "q": {
                "verified": {"eq": True},
                "gpu_name": {"in": [
                    "RTX_3060", "RTX_3060_Ti", "RTX_3070", "RTX_3070_Ti",
                    "RTX_3080", "RTX_3080_Ti", "RTX_3090", "RTX_3090_Ti",
                    "RTX_4060", "RTX_4060_Ti", "RTX_4070", "RTX_4070_Super",
                    "RTX_4070_Ti", "RTX_4070_Ti_Super", "RTX_4080", "RTX_4080_Super",
                    "RTX_4090", "RTX_5070", "RTX_5070_Ti", "RTX_5080", "RTX_5090"
                ]},
                "dph_total": {"lte": 0.15},
                "cuda_max_good": {"gte": 12.8},
                "inet_down": {"gte": 100},
                "cpu_ram": {"gte": 32768},
                "reliability": {"gte": 0.95},
                "datacenter": {"eq": True},
                "geolocation": {"in": ["US", "CA"]},
            },
        }

        async with session.post(
            f"{VASTAI_BASE_URL}/api/v0/bundles/",
            json=payload,
            params={"api_key": self.api_key},
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"Search GPU offers failed ({resp.status}): {text}")
                return []
            data = await resp.json()
            return data.get("offers", data) if isinstance(data, dict) else data

    async def search_cpu_offers(self, num_results: int = 10) -> List[Dict]:
        """Search for CPU offers matching builder requirements (Section 5.2)."""
        session = await self._get_session()

        payload = {
            "select_cols": [
                "id", "gpu_name", "num_gpus", "dph_total",
                "cpu_ram", "disk_space", "cpu_cores",
                "reliability", "verified", "geolocation"
            ],
            "order": [["dph_total", "asc"]],
            "limit": num_results,
            "type": "on-demand",
            "allocated_storage": 700,
            "q": {
                "verified": {"eq": True},
                "cpu_ram": {"gte": 131072},
                "cpu_cores": {"gte": 16},
                "reliability": {"gte": 0.95},
                "geolocation": {"in": ["US", "CA"]},
            },
        }

        async with session.post(
            f"{VASTAI_BASE_URL}/api/v0/bundles/",
            json=payload,
            params={"api_key": self.api_key},
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"Search CPU offers failed ({resp.status}): {text}")
                return []
            data = await resp.json()
            return data.get("offers", data) if isinstance(data, dict) else data

    async def create_instance(
        self,
        offer_id: int,
        docker_image: str,
        env_vars: Dict[str, str],
        disk_gb: int = 100,
        onstart: str = "",
    ) -> Optional[Dict]:
        """Create a Vast.AI instance from an offer.

        Args:
            offer_id: The offer ID to rent.
            docker_image: Docker image to run.
            env_vars: Environment variables to pass to the container.
            disk_gb: Disk space to allocate.
            onstart: Optional startup command.

        Returns:
            Instance creation response dict, or None on failure.
        """
        session = await self._get_session()

        # Format env vars as Vast.AI expects: dict or "-e KEY=VAL" string
        env = {k: v for k, v in env_vars.items()}

        payload = {
            "client_id": "me",
            "image": docker_image,
            "disk": disk_gb,
            "env": env,
            "runtype": "args",
        }

        if onstart:
            payload["onstart"] = onstart

        async with session.post(
            f"{VASTAI_BASE_URL}/api/v0/asks/{offer_id}/",
            json=payload,
            params={"api_key": self.api_key},
        ) as resp:
            if resp.status not in (200, 201):
                text = await resp.text()
                logger.error(f"Create instance failed ({resp.status}): {text}")
                return None
            data = await resp.json()
            logger.info(f"Created instance from offer {offer_id}: {data}")
            return data

    async def destroy_instance(self, instance_id: int) -> bool:
        """Destroy a Vast.AI instance."""
        session = await self._get_session()

        async with session.put(
            f"{VASTAI_BASE_URL}/api/v0/instances/{instance_id}/",
            json={"delete": True},
            params={"api_key": self.api_key},
        ) as resp:
            if resp.status == 200:
                logger.info(f"Destroyed instance {instance_id}")
                return True
            text = await resp.text()
            logger.error(f"Destroy instance {instance_id} failed ({resp.status}): {text}")
            return False

    async def list_instances(self) -> List[Dict]:
        """List all running instances."""
        session = await self._get_session()

        async with session.get(
            f"{VASTAI_BASE_URL}/api/v0/instances/",
            params={"api_key": self.api_key},
        ) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            return data.get("instances", data) if isinstance(data, dict) else data

    async def destroy_all_instances(self, instance_ids: List[int]) -> int:
        """Destroy multiple instances. Returns count of successfully destroyed."""
        destroyed = 0
        for iid in instance_ids:
            if await self.destroy_instance(iid):
                destroyed += 1
        return destroyed
