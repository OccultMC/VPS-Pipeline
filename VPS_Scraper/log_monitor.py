"""
R2 Status Monitor

Polls R2 for compact status JSON from pipeline workers.
Status keys use flat format: Status/{city}_{instance_id}.json
This supports multiple sessions scraping the same city concurrently.
"""
import time
import logging
from typing import Dict
from dataclasses import dataclass

from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


@dataclass
class WorkerProgress:
    """Progress state for a single worker."""
    worker_index: int = 0
    processed: int = 0
    total: int = 0
    eta_seconds: float = 0
    speed: float = 0.0
    status: str = "WAITING"
    instance_id: str = ""
    last_update: float = 0.0


class R2StatusMonitorThread(QThread):
    """
    Background thread that polls R2 for worker status JSON files.
    No instance health checking — just reads R2 and emits signals.

    Polls flat status keys: Status/{city_name}_{instance_id}.json
    """

    # (worker_index, processed, total, eta_seconds, speed, status)
    progress_update = pyqtSignal(int, int, int, float, float, str)
    # (worker_index, status) — terminal states
    worker_finished = pyqtSignal(int, str)
    # General log messages
    log_message = pyqtSignal(str)
    # Kept for API compatibility — never emitted
    instance_needs_replace = pyqtSignal(int, str, str)

    def __init__(
        self,
        r2_client,
        city_name: str,
        instance_worker_map: Dict[str, int],
        poll_interval: float = 10.0,
        vast_manager=None,  # kept for API compat, unused
    ):
        super().__init__()
        self.r2_client = r2_client
        self.city_name = city_name
        self.instance_worker_map = instance_worker_map or {}
        self.poll_interval = poll_interval
        self._running = True
        self._worker_states: Dict[int, WorkerProgress] = {}
        self._finished_instances: set = set()

        for iid, widx in self.instance_worker_map.items():
            self._worker_states[widx] = WorkerProgress(
                worker_index=widx, instance_id=iid
            )

    @property
    def num_workers(self):
        return len(self.instance_worker_map)

    def stop(self):
        self._running = False

    def run(self):
        self.log_message.emit("R2 status monitor started")

        while self._running:
            active_instances = [
                iid for iid in self.instance_worker_map
                if iid not in self._finished_instances
            ]

            if not active_instances:
                self.log_message.emit("All workers finished")
                self._cleanup_status_files()
                break

            for iid in active_instances:
                if not self._running:
                    break
                self._poll_instance(iid)

            for _ in range(int(self.poll_interval * 2)):
                if not self._running:
                    break
                time.sleep(0.5)

        self.log_message.emit("R2 status monitor stopped")

    def _poll_instance(self, instance_id: str):
        worker_idx = self.instance_worker_map.get(instance_id, 0)
        status_key = f"Status/{self.city_name}_{instance_id}.json"
        try:
            data = self.r2_client.download_json(status_key)
            if not data:
                return

            processed = data.get('p', data.get('processed', 0))
            total = data.get('t', data.get('total', 0))
            eta = data.get('eta', 0)
            speed = data.get('spd', data.get('speed', 0.0))
            status = data.get('s', data.get('status', 'UNKNOWN'))
            iid = data.get('iid', data.get('instance_id', instance_id))
            ts = data.get('ts', data.get('timestamp', 0))

            state = self._worker_states.get(worker_idx)
            if state:
                if ts <= state.last_update:
                    return
                state.processed = processed
                state.total = total
                state.eta_seconds = eta
                state.speed = speed
                state.status = status
                state.instance_id = iid
                state.last_update = ts

            self.progress_update.emit(
                worker_idx, processed, total, float(eta), float(speed), status
            )

            if status == "COMPLETED" or status.startswith("FAILED"):
                self._finished_instances.add(instance_id)
                self.worker_finished.emit(worker_idx, status)
                self.log_message.emit(f"Worker {worker_idx} (instance {instance_id}) finished: {status}")

        except Exception as e:
            logger.debug(f"Error polling status for instance {instance_id}: {e}")

    def _cleanup_status_files(self):
        """Delete status and lookup files from R2 after all workers finish."""
        for iid, widx in self.instance_worker_map.items():
            for key in [
                f"Status/{self.city_name}_{iid}.json",
                f"Status/_lookup_{self.city_name}_{widx}.{len(self.instance_worker_map)}.json",
            ]:
                try:
                    self.r2_client.delete_file(key)
                except Exception:
                    pass
        self.log_message.emit("Cleaned up status files from R2")

    def get_overall_progress(self) -> Dict:
        total_processed = sum(s.processed for s in self._worker_states.values())
        total_rows = sum(s.total for s in self._worker_states.values())
        active = len(self._worker_states) - len(self._finished_instances)
        return {
            "total_processed": total_processed,
            "total_rows": total_rows,
            "active_workers": active,
            "finished_workers": len(self._finished_instances),
            "total_workers": len(self._worker_states),
        }


# Backward compat alias
LogMonitorThread = R2StatusMonitorThread
