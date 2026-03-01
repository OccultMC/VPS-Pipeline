"""Pipeline logger — periodically uploads progress JSON to R2.

Writes a {CityName}_{N}_{Total}_Logs.json file every 10 seconds with:
  worker, total_workers, status, panos_done, panos_total, panos_remaining,
  eta_seconds, timestamp
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from r2_client import R2Client

logger = logging.getLogger(__name__)


class PipelineLogger:
    """Tracks pipeline progress and uploads periodic JSON logs to R2."""

    def __init__(
        self,
        r2: R2Client,
        logs_r2_path: str,
        city_name: str,
        worker_number: int,
        total_workers: int,
        total_panos: int,
        upload_interval: float = 10.0,
    ):
        self.r2 = r2
        self.logs_r2_path = logs_r2_path
        self.city_name = city_name
        self.worker = worker_number
        self.total_workers = total_workers
        self.total_panos = total_panos
        self.upload_interval = upload_interval

        self.panos_done = 0
        self.panos_failed = 0
        self.features_done = 0
        self.status = "starting"
        self.error = ""
        self.t_start = time.time()

        # Ensure trailing slash
        if self.logs_r2_path and not self.logs_r2_path.endswith("/"):
            self.logs_r2_path += "/"

        self._log_key = f"{self.logs_r2_path}{city_name}_{worker_number}_{total_workers}_Logs.json"
        self._task: Optional[asyncio.Task] = None

    def start(self):
        """Start the periodic upload loop."""
        self._task = asyncio.create_task(self._upload_loop())

    async def stop(self):
        """Stop the upload loop and do a final upload."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._upload_once()

    def update_scrape(self, panos_done: int, total: int, failed: int):
        self.panos_done = panos_done
        self.total_panos = total
        self.panos_failed = failed
        self.status = "downloading"

    def update_features(self, features_done: int):
        self.features_done = features_done
        self.status = "extracting"

    def set_done(self):
        self.status = "done"

    def set_error(self, msg: str):
        self.status = "error"
        self.error = msg

    def _build_log(self) -> dict:
        elapsed = time.time() - self.t_start
        rate = self.panos_done / max(elapsed, 0.01)
        remaining = self.total_panos - self.panos_done
        eta = remaining / rate if rate > 0 else 0

        return {
            "worker": self.worker,
            "total_workers": self.total_workers,
            "status": self.status,
            "panos_done": self.panos_done,
            "panos_total": self.total_panos,
            "panos_remaining": max(remaining, 0),
            "eta_seconds": round(max(eta, 0), 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def _upload_once(self):
        try:
            self.r2.upload_json(self._log_key, self._build_log())
        except Exception as e:
            logger.warning(f"Log upload failed: {e}")

    async def _upload_loop(self):
        try:
            while True:
                await self._upload_once()
                await asyncio.sleep(self.upload_interval)
        except asyncio.CancelledError:
            pass
