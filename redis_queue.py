"""
Redis Work Queue Manager (Upstash)

Manages a shared task queue for distributed panorama processing.
Uses upstash_redis (HTTP-based, no TLS client needed).

Redis key schema:
    job:{region}:todo     → List   [chunk_0001, chunk_0002, ...]
    job:{region}:active   → Hash   {chunk_id: "worker_id|timestamp"}
    job:{region}:done     → Set    {chunk_0001, chunk_0003, ...}
    job:{region}:failed   → Set    {chunk_0002, ...}  (permanently failed after max retries)
    job:{region}:fcnt     → Hash   {chunk_id: fail_count}
    job:{region}:meta     → Hash   {total_chunks, total_panos, city_name, created_at}
    job:{region}:ws:{wid} → Hash   {status, chunk_id, chunks_done, processed, total, speed, eta, ts}
"""
import time
import logging
from typing import List, Dict, Optional

from upstash_redis import Redis

logger = logging.getLogger(__name__)

STALE_TIMEOUT = 300  # 5 minutes


class TaskQueue:
    """Manages a Redis-backed work queue for chunk-based processing."""

    def __init__(self, redis_url: str, redis_token: str):
        self.redis = Redis(url=redis_url, token=redis_token)
        # upstash_redis 1.6.0 builds httpx.Client(timeout=None) internally —
        # any Redis call can hang forever on a dead connection, freezing the
        # worker. Patch a 10s timeout onto the underlying httpx client
        # (attribute path verified against upstash_redis/http.py: SyncHttpClient
        # stores it at Redis._http._client).
        try:
            import httpx
            self.redis._http._client.timeout = httpx.Timeout(10.0)
            logger.info("Patched Upstash httpx client timeout to 10s")
        except (AttributeError, ImportError) as e:
            print(f"[WARN] Could not patch Upstash httpx timeout "
                  f"({type(e).__name__}: {e}) — Redis calls may hang "
                  f"indefinitely on dead connections", flush=True)

    # ── Key helpers ───────────────────────────────────────────────────────

    def _todo_key(self, region: str) -> str:
        return f"job:{region}:todo"

    def _active_key(self, region: str) -> str:
        return f"job:{region}:active"

    def _done_key(self, region: str) -> str:
        return f"job:{region}:done"

    def _meta_key(self, region: str) -> str:
        return f"job:{region}:meta"

    def _failed_key(self, region: str) -> str:
        return f"job:{region}:failed"

    def _fcnt_key(self, region: str) -> str:
        return f"job:{region}:fcnt"

    def _ucnt_key(self, region: str) -> str:
        return f"job:{region}:ucnt"

    def _worker_status_key(self, region: str, worker_id: str) -> str:
        return f"job:{region}:ws:{worker_id}"

    # ── Job lifecycle ─────────────────────────────────────────────────────

    def init_job(
        self,
        region: str,
        chunk_ids: List[str],
        total_panos: int,
        city_name: str,
    ):
        """Populate the todo list with all chunk IDs and set job metadata."""
        todo_key = self._todo_key(region)
        active_key = self._active_key(region)
        done_key = self._done_key(region)
        meta_key = self._meta_key(region)

        # Clean any previous job data
        self.redis.delete(todo_key, active_key, done_key, meta_key)

        # Push all chunks to the todo list
        if chunk_ids:
            self.redis.rpush(todo_key, *chunk_ids)

        # Set job metadata
        self.redis.hset(meta_key, values={
            "total_chunks": str(len(chunk_ids)),
            "total_panos": str(total_panos),
            "city_name": city_name,
            "created_at": str(time.time()),
        })

        logger.info(
            f"Initialized job {region}: {len(chunk_ids)} chunks, "
            f"{total_panos} panos"
        )

    # LPOP todo + HSET active in one atomic server-side script — a crash
    # between the two calls would otherwise lose the chunk entirely.
    _CLAIM_LUA = (
        "local cid = redis.call('LPOP', KEYS[1]) "
        "if cid then redis.call('HSET', KEYS[2], cid, ARGV[1]) end "
        "return cid"
    )

    def claim_task(self, region: str, worker_id: str) -> Optional[str]:
        """
        Atomically pop a chunk from the todo list and mark it active.

        Returns the chunk_id or None if no tasks are available.
        """
        todo_key = self._todo_key(region)
        active_key = self._active_key(region)

        try:
            chunk_id = self.redis.eval(
                self._CLAIM_LUA,
                keys=[todo_key, active_key],
                args=[f"{worker_id}|{time.time()}"],
            )
            if chunk_id is None:
                return None
            logger.info(f"Claimed chunk {chunk_id} for worker {worker_id}")
            return chunk_id
        except Exception as e:
            print(f"[WARN] claim_task EVAL failed ({type(e).__name__}: {e}) "
                  f"— falling back to non-atomic LPOP+HSET", flush=True)

        chunk_id = self.redis.lpop(todo_key)
        if chunk_id is None:
            return None

        # Mark as active with worker ID and timestamp
        self.redis.hset(active_key, chunk_id, f"{worker_id}|{time.time()}")
        logger.info(f"Claimed chunk {chunk_id} for worker {worker_id}")
        return chunk_id

    def complete_task(self, region: str, chunk_id: str, worker_id: str):
        """Move a task from active to done.

        SADD done BEFORE HDEL active: a crash between the two leaves the
        chunk in both sets, which reclaim/reconcile resolve as a no-op
        duplicate. The reverse order would lose the chunk entirely.
        """
        active_key = self._active_key(region)
        done_key = self._done_key(region)

        self.redis.sadd(done_key, chunk_id)
        self.redis.hdel(active_key, chunk_id)
        logger.info(f"Completed chunk {chunk_id} (worker {worker_id})")

    MAX_UNCLAIMS = 10

    def unclaim_task(self, region: str, chunk_id: str, worker_id: str,
                     reason: str = "", back: bool = False):
        """
        Return a task to todo WITHOUT incrementing the fail counter.

        Use when the failure is self-inflicted or infrastructural (IP
        403-block, upload failure, disk pressure) so a fresh worker isn't
        punished for the previous worker's misfortune.

        back=False (default) → LPUSH front: genuine-IP-block path, a new
        instance with a fresh IP should pick the chunk up first.
        back=True → RPUSH back: infra failures, let other chunks go first.

        A separate unclaim counter (ucnt) caps infinite unclaim ping-pong:
        after MAX_UNCLAIMS the chunk is routed to the failed set instead.
        """
        active_key = self._active_key(region)
        todo_key = self._todo_key(region)

        try:
            ucnt = self.redis.hincrby(self._ucnt_key(region), chunk_id, 1)
        except Exception as e:
            print(f"[WARN] unclaim counter HINCRBY failed for {chunk_id}: {e}",
                  flush=True)
            ucnt = 1

        if ucnt >= self.MAX_UNCLAIMS:
            print(f"[WARN] !!! chunk {chunk_id} unclaimed {ucnt} times "
                  f"(worker {worker_id}, reason: {reason}) — routing to "
                  f"FAILED set instead of requeueing !!!", flush=True)
            self.redis.sadd(self._failed_key(region), chunk_id)
            self.redis.hdel(active_key, chunk_id)
            logger.warning(
                f"Chunk {chunk_id} permanently failed after {ucnt} unclaims"
            )
            return

        self.redis.hdel(active_key, chunk_id)
        if back:
            self.redis.rpush(todo_key, chunk_id)
        else:
            # lpush so the chunk is picked up quickly by the next worker
            self.redis.lpush(todo_key, chunk_id)
        logger.warning(
            f"Unclaimed chunk {chunk_id} (worker {worker_id}) — {reason} — "
            f"returned to todo ({'back' if back else 'front'}) without "
            f"fail-count bump (unclaim {ucnt}/{self.MAX_UNCLAIMS})"
        )

    def fail_task(
        self, region: str, chunk_id: str, worker_id: str, error: str,
        max_retries: int = 3,
    ):
        """Increment fail count and re-queue, or permanently fail after max_retries."""
        active_key = self._active_key(region)
        fcnt_key = self._fcnt_key(region)

        self.redis.hdel(active_key, chunk_id)

        count = self.redis.hincrby(fcnt_key, chunk_id, 1)

        if count >= max_retries:
            # Permanently failed — do NOT re-queue
            self.redis.sadd(self._failed_key(region), chunk_id)
            logger.warning(
                f"Permanently failed chunk {chunk_id} (worker {worker_id}) "
                f"after {count} attempts: {error}"
            )
        else:
            self.redis.rpush(self._todo_key(region), chunk_id)
            logger.warning(
                f"Failed chunk {chunk_id} (worker {worker_id}): {error} — "
                f"returned to todo (attempt {count}/{max_retries})"
            )

    def heartbeat(self, region: str, worker_id: str, chunk_id: str):
        """Update the timestamp for an active task (proves worker is alive)."""
        active_key = self._active_key(region)
        self.redis.hset(active_key, chunk_id, f"{worker_id}|{time.time()}")

    # Only refresh the timestamp if the chunk is still active — an
    # unconditional HSET racing a concurrent complete/unclaim would
    # re-insert a chunk that was just removed from the active hash.
    _HB_LUA = (
        "if redis.call('HEXISTS', KEYS[1], ARGV[1]) == 1 then "
        "redis.call('HSET', KEYS[1], ARGV[1], ARGV[2]) return 1 "
        "else return 0 end"
    )

    def heartbeat_if_active(self, region: str, worker_id: str, chunk_id: str) -> bool:
        """Heartbeat only if the chunk is still in the active hash.

        Used by the background HeartbeatThread, which can race the main
        thread's complete_task/unclaim_task. Falls back to a plain
        heartbeat if EVAL is unavailable.
        """
        active_key = self._active_key(region)
        try:
            res = self.redis.eval(
                self._HB_LUA,
                keys=[active_key],
                args=[chunk_id, f"{worker_id}|{time.time()}"],
            )
            return bool(res)
        except Exception:
            self.redis.hset(active_key, chunk_id, f"{worker_id}|{time.time()}")
            return True

    # ── Stale task recovery ───────────────────────────────────────────────

    def reclaim_stale(
        self, region: str, timeout: int = STALE_TIMEOUT
    ) -> List[str]:
        """
        Find tasks stuck in active for longer than `timeout` seconds.

        Moves them back to todo so other workers can pick them up.
        """
        active_key = self._active_key(region)
        todo_key = self._todo_key(region)

        all_active = self.redis.hgetall(active_key)
        if not all_active:
            return []

        now = time.time()
        reclaimed = []

        for chunk_id, value in all_active.items():
            try:
                parts = value.rsplit("|", 1)
                claimed_at = float(parts[-1])
            except (ValueError, IndexError):
                claimed_at = 0.0

            if now - claimed_at > timeout:
                # Skip chunks already done — a complete_task that raced a
                # previous reclaim can leave a stale active entry behind.
                try:
                    if self.redis.sismember(self._done_key(region), chunk_id):
                        self.redis.hdel(active_key, chunk_id)
                        continue
                except Exception:
                    pass
                # Only requeue if WE removed the active entry (hdel == 1) —
                # another worker's concurrent reclaim already requeued it
                # otherwise, and a second rpush would duplicate the chunk.
                if self.redis.hdel(active_key, chunk_id) == 1:
                    self.redis.rpush(todo_key, chunk_id)
                    reclaimed.append(chunk_id)
                    worker = parts[0] if len(parts) > 1 else "unknown"
                    logger.warning(
                        f"Reclaimed stale chunk {chunk_id} "
                        f"(was held by {worker} for {now - claimed_at:.0f}s)"
                    )

        return reclaimed

    def recover_lost_tasks(self, region: str) -> List[str]:
        """
        Safety net: if any chunk IDs are not in todo, active, or done,
        re-add them to todo. Handles the edge case where LPOP succeeds
        but HSET fails (worker crashed between the two calls).
        """
        meta = self.redis.hgetall(self._meta_key(region))
        if not meta:
            return []

        total_chunks = int(meta.get("total_chunks", "0"))
        if total_chunks == 0:
            return []

        # Build the full set of expected chunk IDs
        all_expected = {f"chunk_{i:04d}" for i in range(1, total_chunks + 1)}

        # Get current state
        todo_list = self.redis.lrange(self._todo_key(region), 0, -1) or []
        active_dict = self.redis.hgetall(self._active_key(region)) or {}
        done_set = self.redis.smembers(self._done_key(region)) or set()
        failed_set = self.redis.smembers(self._failed_key(region)) or set()

        accounted = set(todo_list) | set(active_dict.keys()) | set(done_set) | set(failed_set)
        lost = all_expected - accounted

        if lost:
            todo_key = self._todo_key(region)
            self.redis.rpush(todo_key, *sorted(lost))
            logger.warning(f"Recovered {len(lost)} lost tasks: {sorted(lost)}")

        return sorted(lost)

    # ── Progress queries ──────────────────────────────────────────────────

    def get_progress(self, region: str) -> Dict:
        """Return current job progress counts."""
        todo_count = self.redis.llen(self._todo_key(region)) or 0
        active_count = self.redis.hlen(self._active_key(region)) or 0
        done_count = self.redis.scard(self._done_key(region)) or 0
        failed_count = self.redis.scard(self._failed_key(region)) or 0

        meta = self.redis.hgetall(self._meta_key(region)) or {}
        total_chunks = int(meta.get("total_chunks", "0"))
        total_panos = int(meta.get("total_panos", "0"))

        return {
            "total_chunks": total_chunks,
            "total_panos": total_panos,
            "todo": todo_count,
            "active": active_count,
            "done": done_count,
            "failed": failed_count,
        }

    def get_active_details(self, region: str) -> Dict[str, Dict]:
        """Return details of all active tasks (chunk_id → {worker_id, claimed_at})."""
        all_active = self.redis.hgetall(self._active_key(region)) or {}
        result = {}
        for chunk_id, value in all_active.items():
            parts = value.rsplit("|", 1)
            result[chunk_id] = {
                "worker_id": parts[0] if len(parts) > 1 else "unknown",
                "claimed_at": float(parts[-1]) if parts else 0.0,
            }
        return result

    def is_complete(self, region: str) -> bool:
        """True if all chunks are done or permanently failed (nothing in todo or active)."""
        todo = self.redis.llen(self._todo_key(region)) or 0
        active = self.redis.hlen(self._active_key(region)) or 0
        return todo == 0 and active == 0

    # ── Resume / Reconciliation ─────────────────────────────────────────

    def reconcile_done(self, region: str, done_chunk_ids: set) -> int:
        """
        Mark chunks as done based on external verification (e.g., R2 upload check).
        Removes from todo and active, adds to done. Returns count reconciled.
        """
        if not done_chunk_ids:
            return 0

        todo_key = self._todo_key(region)
        active_key = self._active_key(region)
        done_key = self._done_key(region)

        # Skip chunks already marked done in Redis
        # Note: Upstash smembers() returns a list, not a set
        already_done = set(self.redis.smembers(done_key) or [])
        need_reconcile = done_chunk_ids - already_done

        if not need_reconcile:
            return 0

        count = 0
        for chunk_id in sorted(need_reconcile):
            self.redis.lrem(todo_key, 0, chunk_id)
            self.redis.hdel(active_key, chunk_id)
            self.redis.sadd(done_key, chunk_id)
            count += 1

        logger.info(f"Reconciled {count} chunks as done for {region}")
        return count

    # ── Per-worker status ─────────────────────────────────────────────────

    def report_status(
        self,
        region: str,
        worker_id: str,
        status: str,
        chunk_id: str = "",
        chunks_done: int = 0,
        processed: int = 0,
        total: int = 0,
        speed: float = 0.0,
        eta: float = 0.0,
    ):
        """Write per-worker status hash for monitoring."""
        key = self._worker_status_key(region, worker_id)
        self.redis.hset(key, values={
            "s": status,
            "cid": chunk_id,
            "cd": str(chunks_done),
            "p": str(processed),
            "t": str(total),
            "spd": f"{speed:.1f}",
            "eta": f"{eta:.0f}",
            "ts": f"{time.time():.1f}",
        })

    def get_all_worker_statuses(self, region: str) -> Dict[str, Dict]:
        """Scan for all worker status hashes under this region.

        Returns {worker_id: {status, chunk_id, chunks_done, processed, total, speed, eta, ts}}.
        """
        prefix = f"job:{region}:ws:"
        result = {}
        # Scan for matching keys
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match=f"{prefix}*", count=100)
            for key in keys:
                worker_id = key[len(prefix):]
                data = self.redis.hgetall(key)
                if data:
                    result[worker_id] = {
                        "status": data.get("s", "UNKNOWN"),
                        "chunk_id": data.get("cid", ""),
                        "chunks_done": int(data.get("cd", "0")),
                        "processed": int(data.get("p", "0")),
                        "total": int(data.get("t", "0")),
                        "speed": float(data.get("spd", "0")),
                        "eta": float(data.get("eta", "0")),
                        "ts": float(data.get("ts", "0")),
                    }
            if cursor == 0:
                break
        return result

    # ── Batch chunk metadata ─────────────────────────────────────────────

    def _cmap_key(self, region: str) -> str:
        return f"job:{region}:cmap"

    def set_batch_meta(self, region: str, chunk_metas: dict):
        """Store per-chunk metadata for batch jobs.

        chunk_metas: {global_chunk_id: "csv_prefix|features_prefix|city_name|city_total|chunk_num"}
        """
        key = self._cmap_key(region)
        if chunk_metas:
            self.redis.hset(key, values=chunk_metas)
        logger.info(f"Stored batch metadata for {len(chunk_metas)} chunks in {region}")

    def get_chunk_meta(self, region: str, chunk_id: str) -> Optional[dict]:
        """Get batch metadata for a specific chunk. Returns dict or None."""
        val = self.redis.hget(self._cmap_key(region), chunk_id)
        if not val:
            return None
        parts = val.split('|')
        if len(parts) < 5:
            return None
        return {
            'csv_prefix': parts[0],
            'features_prefix': parts[1],
            'city_name': parts[2],
            'city_total': int(parts[3]),
            'chunk_num': int(parts[4]),
        }

    # ── Redo city ─────────────────────────────────────────────────────────

    def redo_city(self, region: str, city_name: Optional[str] = None) -> List[str]:
        """
        Reset done/failed chunks for a city back to todo so they can be reprocessed.

        For single-city jobs (no city_name given or no cmap): resets ALL done/failed chunks.
        For batch jobs: only resets chunks belonging to the specified city_name (matched via cmap).

        Returns list of chunk IDs moved back to todo.
        """
        done_key = self._done_key(region)
        failed_key = self._failed_key(region)
        todo_key = self._todo_key(region)
        fcnt_key = self._fcnt_key(region)

        done_set = set(self.redis.smembers(done_key) or [])
        failed_set = set(self.redis.smembers(failed_key) or [])
        candidates = done_set | failed_set

        if not candidates:
            logger.info(f"redo_city: nothing to redo for {region}")
            return []

        # Filter by city_name if this is a batch job
        if city_name:
            cmap = self.redis.hgetall(self._cmap_key(region)) or {}
            if cmap:
                # cmap values: "csv_prefix|features_prefix|city_name|city_total|chunk_num"
                city_chunks = set()
                for chunk_id, val in cmap.items():
                    parts = val.split('|')
                    if len(parts) >= 3 and parts[2] == city_name:
                        city_chunks.add(chunk_id)
                candidates = candidates & city_chunks
                if not candidates:
                    logger.info(f"redo_city: no done/failed chunks found for city '{city_name}' in {region}")
                    return []

        # Move from done/failed → todo, clear fail + unclaim counts
        reset = sorted(candidates)
        for chunk_id in reset:
            self.redis.srem(done_key, chunk_id)
            self.redis.srem(failed_key, chunk_id)
            self.redis.hdel(fcnt_key, chunk_id)
            self.redis.hdel(self._ucnt_key(region), chunk_id)

        self.redis.rpush(todo_key, *reset)

        logger.info(
            f"redo_city: reset {len(reset)} chunks back to todo for "
            f"{region}{f' (city={city_name})' if city_name else ''}"
        )
        return reset

    # ── Cleanup ───────────────────────────────────────────────────────────

    def cleanup(self, region: str):
        """Delete all Redis keys for a completed job."""
        # Clean worker status keys
        prefix = f"job:{region}:ws:"
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match=f"{prefix}*", count=100)
            if keys:
                self.redis.delete(*keys)
            if cursor == 0:
                break

        self.redis.delete(
            self._todo_key(region),
            self._active_key(region),
            self._done_key(region),
            self._failed_key(region),
            self._fcnt_key(region),
            self._ucnt_key(region),
            self._meta_key(region),
            self._cmap_key(region),
        )
        logger.info(f"Cleaned up job keys for {region}")
