"""
R2 Status Monitor — Web-compatible version (no PyQt6 dependency).

Drop-in replacement for log_monitor.py using plain threading.Thread
and callback functions instead of QThread/pyqtSignal.
"""
import time
import logging
import threading
from typing import Dict, Callable, Optional
from dataclasses import dataclass

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


class R2StatusMonitorThread(threading.Thread):
    """
    Background thread that polls R2 for worker status JSON files.
    Uses callback functions instead of Qt signals.
    """

    def __init__(
        self,
        r2_client,
        status_prefix: str,
        num_workers: int,
        instance_worker_map: Dict[str, int] = None,
        poll_interval: float = 10.0,
        on_progress: Optional[Callable] = None,
        on_worker_finished: Optional[Callable] = None,
        on_log_message: Optional[Callable] = None,
    ):
        super().__init__(daemon=True)
        self.r2_client = r2_client
        self.status_prefix = status_prefix
        self.num_workers = num_workers
        self.instance_worker_map = instance_worker_map or {}
        self.poll_interval = poll_interval
        self._running = True
        self._worker_states: Dict[int, WorkerProgress] = {}
        self._finished_workers: set = set()

        # Callbacks
        self._on_progress = on_progress
        self._on_worker_finished = on_worker_finished
        self._on_log_message = on_log_message

        for widx in range(1, num_workers + 1):
            self._worker_states[widx] = WorkerProgress(worker_index=widx)

    def stop(self):
        self._running = False

    def run(self):
        self._emit_log("R2 status monitor started")

        while self._running:
            active_workers = [
                widx for widx in range(1, self.num_workers + 1)
                if widx not in self._finished_workers
            ]

            if not active_workers:
                self._emit_log("All workers finished")
                break

            for widx in active_workers:
                if not self._running:
                    break
                self._poll_worker(widx)

            for _ in range(int(self.poll_interval * 2)):
                if not self._running:
                    break
                time.sleep(0.5)

        self._emit_log("R2 status monitor stopped")

    def _poll_worker(self, worker_idx: int):
        status_key = f"Status/{self.status_prefix}/worker_{worker_idx}.json"
        try:
            data = self.r2_client.download_json(status_key)
            if not data:
                return

            processed = data.get('p', data.get('processed', 0))
            total = data.get('t', data.get('total', 0))
            eta = data.get('eta', 0)
            speed = data.get('spd', data.get('speed', 0.0))
            status = data.get('s', data.get('status', 'UNKNOWN'))
            instance_id = data.get('iid', data.get('instance_id', ''))
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
                state.instance_id = instance_id
                state.last_update = ts

            if self._on_progress:
                self._on_progress(worker_idx, processed, total, float(eta), float(speed), status)

            if status == "COMPLETED" or status.startswith("FAILED"):
                self._finished_workers.add(worker_idx)
                if self._on_worker_finished:
                    self._on_worker_finished(worker_idx, status)
                self._emit_log(f"Worker {worker_idx} finished: {status}")

        except Exception as e:
            logger.debug(f"Error polling status for worker {worker_idx}: {e}")

    def _emit_log(self, message: str):
        if self._on_log_message:
            self._on_log_message(message)

    def get_overall_progress(self) -> Dict:
        total_processed = sum(s.processed for s in self._worker_states.values())
        total_rows = sum(s.total for s in self._worker_states.values())
        active = len(self._worker_states) - len(self._finished_workers)
        return {
            "total_processed": total_processed,
            "total_rows": total_rows,
            "active_workers": active,
            "finished_workers": len(self._finished_workers),
            "total_workers": len(self._worker_states),
        }
