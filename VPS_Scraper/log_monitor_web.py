"""
Status Monitors — Web-compatible version (no PyQt6 dependency).

Two monitor classes:
  - R2StatusMonitorThread: Legacy R2-based polling for per-worker status
  - RedisQueueMonitorThread: Redis queue-based monitoring with stale task recovery

Also polls Vast.ai instance status for workers that haven't reported yet.
"""
import time
import logging
import threading
from typing import Dict, Callable, Optional, List
from dataclasses import dataclass, field

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
    has_r2_data: bool = False  # True once we get a real R2 status update


class R2StatusMonitorThread(threading.Thread):
    """
    Background thread that polls R2 for worker status JSON files
    and Vast.ai for instance status (loading/running/exited/error).

    Polls flat status keys: Status/{city_name}_{instance_id}.json
    The instance_worker_map maps instance_id -> worker_index.
    """

    def __init__(
        self,
        r2_client,
        city_name: str,
        instance_worker_map: Dict[str, int],
        poll_interval: float = 10.0,
        on_progress: Optional[Callable] = None,
        on_worker_finished: Optional[Callable] = None,
        on_log_message: Optional[Callable] = None,
        vast_manager=None,
    ):
        super().__init__(daemon=True)
        self.r2_client = r2_client
        self.city_name = city_name
        self.instance_worker_map = instance_worker_map or {}
        self.poll_interval = poll_interval
        self._running = True
        self._worker_states: Dict[int, WorkerProgress] = {}
        self._finished_instances: set = set()
        self.vast_manager = vast_manager

        # Callbacks
        self._on_progress = on_progress
        self._on_worker_finished = on_worker_finished
        self._on_log_message = on_log_message

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
        self._emit_log("R2 status monitor started")

        while self._running:
            active_instances = [
                iid for iid in self.instance_worker_map
                if iid not in self._finished_instances
            ]

            if not active_instances:
                self._emit_log("All workers finished")
                self._cleanup_status_files()
                break

            for iid in active_instances:
                if not self._running:
                    break
                self._poll_instance(iid)

            # Poll Vast.ai status for instances that haven't reported R2 data yet
            if self.vast_manager:
                for iid in active_instances:
                    if not self._running:
                        break
                    widx = self.instance_worker_map.get(iid, 0)
                    state = self._worker_states.get(widx)
                    if state and not state.has_r2_data:
                        self._poll_vast_status(iid, widx)

            for _ in range(int(self.poll_interval * 2)):
                if not self._running:
                    break
                time.sleep(0.5)

        self._emit_log("R2 status monitor stopped")

    def _poll_vast_status(self, instance_id: str, worker_idx: int):
        """Poll Vast.ai for instance status (loading, running, exited, error)."""
        try:
            info = self.vast_manager.get_instance_status(instance_id)
            actual_status = info.get('status', 'unknown')
            status_msg = info.get('status_msg', '')

            # Map Vast.ai status to a display status
            status_map = {
                'loading': 'LOADING',
                'pulling': 'PULLING',
                'creating': 'CREATING',
                'running': 'STARTING',
                'exited': 'EXITED',
                'error': 'ERROR',
            }
            display_status = status_map.get(actual_status, actual_status.upper() if actual_status else 'WAITING')

            # Emit progress with the Vast.ai status so UI shows something
            if self._on_progress:
                self._on_progress(worker_idx, 0, 0, 0, 0.0, display_status)

            # If the instance died before ever reporting to R2, mark finished
            if actual_status in ('exited', 'error'):
                detail = f" — {status_msg}" if status_msg else ""
                self._emit_log(f"Worker {worker_idx} (instance {instance_id}) {actual_status}{detail}")
                # Don't mark finished yet — give it a chance to come back
                # Only mark finished if R2 also never reports

        except Exception as e:
            logger.debug(f"Error polling Vast.ai status for instance {instance_id}: {e}")

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
                state.has_r2_data = True

            if self._on_progress:
                self._on_progress(worker_idx, processed, total, float(eta), float(speed), status)

            if status == "COMPLETED" or status.startswith("FAILED"):
                self._finished_instances.add(instance_id)
                if self._on_worker_finished:
                    self._on_worker_finished(worker_idx, status)
                self._emit_log(f"Worker {worker_idx} (instance {instance_id}) finished: {status}")

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
        self._emit_log("Cleaned up status files from R2")

    def _emit_log(self, message: str):
        if self._on_log_message:
            self._on_log_message(message)

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


class RedisQueueMonitorThread(threading.Thread):
    """
    Background thread that monitors the Redis task queue for job progress.

    Features:
    - Polls Redis every `poll_interval` seconds for todo/active/done counts
    - Reclaims stale tasks every `stale_interval` seconds (tasks stuck >5min)
    - Recovers lost tasks (LPOP succeeded but HSET failed edge case)
    - Polls Vast.ai for instance status (loading/running/exited/error)
    - Detects job completion and fires callback
    """

    def __init__(
        self,
        task_queue,
        region: str,
        instance_ids: List[str],
        poll_interval: float = 10.0,
        stale_interval: float = 60.0,
        on_progress: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
        on_log: Optional[Callable] = None,
        vast_manager=None,
    ):
        super().__init__(daemon=True)
        self.tq = task_queue
        self.region = region
        self.instance_ids = instance_ids or []
        self.poll_interval = poll_interval
        self.stale_interval = stale_interval
        self._running = True
        self._last_stale_check = 0.0
        self._last_lost_check = 0.0
        self.vast_manager = vast_manager

        # Track Vast.ai instance states
        self._instance_states: Dict[str, str] = {}

        # Callbacks
        self._on_progress = on_progress
        self._on_complete = on_complete
        self._on_log = on_log

    def stop(self):
        self._running = False

    def run(self):
        self._emit_log("Redis queue monitor started")

        while self._running:
            now = time.time()

            # ── 1. Get job progress from Redis ──
            try:
                progress = self.tq.get_progress(self.region)
                active_details = self.tq.get_active_details(self.region)

                if self._on_progress:
                    self._on_progress({
                        'total_chunks': progress['total_chunks'],
                        'total_panos': progress['total_panos'],
                        'todo': progress['todo'],
                        'active': progress['active'],
                        'done': progress['done'],
                        'active_details': {
                            k: {'worker_id': v['worker_id'],
                                'age_seconds': now - v['claimed_at']}
                            for k, v in active_details.items()
                        },
                    })
            except Exception as e:
                logger.debug(f"Error polling Redis progress: {e}")

            # ── 2. Reclaim stale tasks every stale_interval ──
            if now - self._last_stale_check >= self.stale_interval:
                try:
                    reclaimed = self.tq.reclaim_stale(self.region)
                    for chunk_id in reclaimed:
                        self._emit_log(f"Reclaimed stale task: {chunk_id}")
                except Exception as e:
                    logger.debug(f"Error reclaiming stale tasks: {e}")
                self._last_stale_check = now

            # ── 3. Recover lost tasks every 5 minutes ──
            if now - self._last_lost_check >= 300:
                try:
                    lost = self.tq.recover_lost_tasks(self.region)
                    for chunk_id in lost:
                        self._emit_log(f"Recovered lost task: {chunk_id}")
                except Exception as e:
                    logger.debug(f"Error recovering lost tasks: {e}")
                self._last_lost_check = now

            # ── 4. Poll Vast.ai instance status ──
            if self.vast_manager:
                for iid in self.instance_ids:
                    if not self._running:
                        break
                    try:
                        info = self.vast_manager.get_instance_status(iid)
                        actual_status = info.get('status', 'unknown')
                        prev_status = self._instance_states.get(iid)

                        if actual_status != prev_status:
                            self._instance_states[iid] = actual_status
                            status_msg = info.get('status_msg', '')
                            detail = f" — {status_msg}" if status_msg else ""
                            self._emit_log(f"Instance {iid}: {actual_status}{detail}")
                    except Exception as e:
                        logger.debug(f"Error polling Vast.ai for {iid}: {e}")

            # ── 5. Check if job is complete ──
            try:
                if self.tq.is_complete(self.region):
                    self._emit_log("All chunks completed!")
                    if self._on_complete:
                        self._on_complete()
                    break
            except Exception as e:
                logger.debug(f"Error checking job completion: {e}")

            # ── Sleep with early exit check ──
            for _ in range(int(self.poll_interval * 2)):
                if not self._running:
                    break
                time.sleep(0.5)

        self._emit_log("Redis queue monitor stopped")

    def _emit_log(self, message: str):
        if self._on_log:
            self._on_log(message)

    def get_overall_progress(self) -> Dict:
        try:
            return self.tq.get_progress(self.region)
        except Exception:
            return {
                'total_chunks': 0, 'total_panos': 0,
                'todo': 0, 'active': 0, 'done': 0,
            }
