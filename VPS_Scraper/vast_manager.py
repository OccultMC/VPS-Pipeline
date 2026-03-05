"""
Vast.ai Instance Manager

Manages GPU instance lifecycle via the ``vastai`` CLI:
search offers, create instances, monitor status, destroy.
"""
import json
import subprocess
import logging
import time
import os
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class ContainerNotFoundError(Exception):
    """Raised when vastai reports 'No such container' for an instance."""
    pass


def _run_vastai(*args, api_key: str = None) -> str:
    """Run a vastai CLI command and return stdout."""
    cmd = ["vastai"]
    if api_key:
        cmd += ["--api-key", api_key]
    cmd += list(args)

    logger.debug(f"Running: {' '.join(cmd)}")
    # Force UTF-8 in both subprocess and the vastai CLI itself (Python on Windows
    # defaults to charmap, which crashes on emojis/non-ASCII in logs).
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        cmd, capture_output=True, encoding='utf-8', errors='replace',
        timeout=60, env=env,
    )

    if result.returncode != 0:
        err = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"vastai command failed (exit {result.returncode}): {err}")

    return result.stdout.strip()


class VastManager:
    """Manages Vast.ai GPU instances."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("VAST_API_KEY", "")
        if not self.api_key:
            raise ValueError("VAST_API_KEY not set")

        self._instances: Dict[str, Dict] = {}  # instance_id -> info

    # ── Offers ────────────────────────────────────────────────────────────

    def search_offers(
        self,
        gpu_type: str = "",
        region: str = "",
        min_disk_gb: int = 100,
        max_price: float = None,
        num_gpus: int = 1,
        min_ram_gb: float = 21,
    ) -> List[Dict]:
        """
        Search for available GPU offers.

        Args:
            min_ram_gb: Minimum system RAM in GB (default 21).
            min_disk_gb: Minimum disk space in GB.

        Returns a list of offer dicts sorted by price (cheapest first).
        """
        # Field reference: vastai search offers --help
        #   gpu_ram     = per-GPU VRAM in GB
        #   cpu_ram     = system RAM in GB
        #   cuda_vers   = max CUDA version (float, e.g. 12.4)
        #   compute_cap = CUDA compute capability * 100 (e.g. 860 = sm_86 = RTX 30xx)
        #   disk_space  = disk in GB
        #   gpu_name    = GPU name with underscores (e.g. RTX_4090)
        query_parts = [
            f"num_gpus={num_gpus}",
            f"disk_space>={min_disk_gb}",
            "rentable=true",
            "verified=true",
            # Machine reliability — filters out flaky hosts that fail to start
            "reliability>0.98",
            # CUDA 12.4+ required for PyTorch 2.4
            "cuda_vers>=12.4",
            # Bare metal only — avoids GPU passthrough issues (CDI errors)
            "vms_enabled=False",
            # No display-attached GPUs (desktop machines sharing GPU with monitor)
            "gpu_display_active=False",
            # Fast download for image pulls (uncached images can be multi-GB)
            "inet_down>200",
            # Direct port access for SSH/HTTP
            "direct_port_count>2",
        ]

        if min_ram_gb > 0:
            query_parts.append(f"cpu_ram>={min_ram_gb}")

        if gpu_type and gpu_type.strip():
            # Replace spaces with underscores per vastai syntax
            gpu_name = gpu_type.strip().replace(' ', '_')
            query_parts.append(f"gpu_name={gpu_name}")

        if region and region.strip():
            # Support single country ("US") or multiple ("US CA GB")
            codes = region.strip().split()
            if len(codes) == 1:
                query_parts.append(f"geolocation={codes[0]}")
            else:
                # Vast.ai 'in' syntax for multiple values
                query_parts.append(f"geolocation in [{','.join(codes)}]")

        if max_price:
            query_parts.append(f"dph<={max_price}")

        query = " ".join(query_parts)
        print(f"DEBUG: Vast.ai Query: {query}")

        try:
            raw = _run_vastai(
                "search", "offers", query,
                "--order", "score-",
                "--raw",
                api_key=self.api_key,
            )
            offers = json.loads(raw)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse offers JSON: {raw[:200]}")
            return []
        except Exception as e:
            logger.error(f"Failed to search offers: {e}")
            return []

        # Filter to RTX 30/40/50 series only (exclude datacenter cards like A10, A5000, etc.)
        import re
        _RTX_CONSUMER = re.compile(r'RTX\s*[345]0[0-9]{2}', re.IGNORECASE)
        if not gpu_type:
            offers = [o for o in offers if _RTX_CONSUMER.search(o.get('gpu_name', ''))]
            logger.info(f"Filtered to RTX 30/40/50 series: {len(offers)} offers")

        # Parse into clean dicts
        results = []
        for o in offers:
            results.append({
                "id": o.get("id"),
                "gpu_name": o.get("gpu_name", "Unknown"),
                "num_gpus": o.get("num_gpus", 1),
                "gpu_ram": o.get("gpu_ram", 0),
                "cpu_cores": o.get("cpu_cores_effective", 0),
                "ram": o.get("cpu_ram", 0),
                "disk": o.get("disk_space", 0),
                "price_per_hr": o.get("dph_total", 0),
                "inet_down": o.get("inet_down", 0),
                "inet_up": o.get("inet_up", 0),
                "reliability": o.get("reliability2", 0),
                "location": o.get("geolocation", "Unknown"),
                "driver_version": o.get("driver_version", ""),
                "cuda_vers": o.get("cuda_max_good", 0),
                "direct_ports": o.get("direct_port_count", 0),
                "machine_id": o.get("machine_id", ""),
            })

        logger.info(f"Found {len(results)} offers matching query")
        return results

    # ── Instance Creation ─────────────────────────────────────────────────

    # Template hash for ghcr.io/occultmc/geoaxisimage:latest
    # Using a template tells Vast.ai to prefer machines that already have
    # the image cached, dramatically reducing instance startup time.
    # NOTE: If the Docker image is rebuilt, you must recreate the template
    # via `vastai create template --name "GeoAxis Pipeline" --image ghcr.io/occultmc/geoaxisimage:latest --ssh --direct`
    # and update this hash with the new hash_id from the response.
    GEOAXIS_TEMPLATE_HASH = "1bb14bdc7a0b8f71ea8ea9227ca4da07"

    def create_instance(
        self,
        offer_id: int,
        docker_image: str,
        env_vars: Dict[str, str],
        disk_gb: int = 100,
        onstart_cmd: str = None,
        template_hash: str = None,
    ) -> Optional[str]:
        """
        Create an instance from an offer.

        Args:
            offer_id: The offer ID to rent.
            docker_image: Docker image to run.
            env_vars: Environment variables to pass.
            disk_gb: Disk space to allocate.
            onstart_cmd: Optional startup command.
            template_hash: Optional Vast.ai template hash for image caching.

        Returns:
            Instance ID string, or None on failure.
        """
        # Build environment string: --env "-e KEY=VALUE -e KEY2=VALUE"
        env_flags = []
        for k, v in env_vars.items():
            env_flags.append(f"-e {k}={v}")
        env_string = " ".join(env_flags)

        args = [
            "create", "instance", str(offer_id),
            "--image", docker_image,
            "--disk", str(disk_gb),
            "--env", env_string,
            "--raw",
        ]

        if template_hash:
            args += ["--template_hash", template_hash]

        if onstart_cmd:
            args += ["--onstart-cmd", onstart_cmd]

        try:
            raw = _run_vastai(*args, api_key=self.api_key)
            result = json.loads(raw)

            instance_id = None
            if isinstance(result, dict):
                instance_id = str(result.get("new_contract", result.get("id", "")))
            elif isinstance(result, str) and result.strip().isdigit():
                instance_id = result.strip()

            if instance_id:
                self._instances[instance_id] = {
                    "offer_id": offer_id,
                    "status": "creating",
                    "docker_image": docker_image,
                }
                logger.info(f"Created instance {instance_id} from offer {offer_id}")
                return instance_id

            logger.error(f"Unexpected create response: {raw[:200]}")
            return None

        except Exception as e:
            err_str = str(e).lower()
            # If the template hash is stale/invalid, retry without it
            if template_hash and ("template" in err_str or "secret" in err_str):
                logger.warning(
                    f"Template hash '{template_hash}' appears invalid, "
                    f"retrying without template (error: {e})"
                )
                args = [a for a in args if a not in ("--template_hash", template_hash)]
                try:
                    raw = _run_vastai(*args, api_key=self.api_key)
                    result = json.loads(raw)
                    instance_id = None
                    if isinstance(result, dict):
                        instance_id = str(result.get("new_contract", result.get("id", "")))
                    elif isinstance(result, str) and result.strip().isdigit():
                        instance_id = result.strip()
                    if instance_id:
                        self._instances[instance_id] = {
                            "offer_id": offer_id,
                            "status": "creating",
                            "docker_image": docker_image,
                        }
                        logger.info(
                            f"Created instance {instance_id} from offer {offer_id} "
                            f"(without template)"
                        )
                        return instance_id
                except Exception as retry_err:
                    logger.error(f"Retry without template also failed: {retry_err}")
                    return None

            logger.error(f"Failed to create instance from offer {offer_id}: {e}")
            return None

    # ── Status ────────────────────────────────────────────────────────────

    def get_instance_status(self, instance_id: str) -> Dict:
        """Get the status of an instance."""
        try:
            raw = _run_vastai(
                "show", "instance", str(instance_id),
                "--raw",
                api_key=self.api_key,
            )
            info = json.loads(raw)
            if isinstance(info, list) and info:
                info = info[0]
            return {
                "id": instance_id,
                "status": info.get("actual_status", "unknown"),
                "status_msg": info.get("status_msg", ""),
                "gpu_name": info.get("gpu_name", ""),
                "machine_id": info.get("machine_id", ""),
            }
        except Exception as e:
            logger.error(f"Failed to get status for instance {instance_id}: {e}")
            return {"id": instance_id, "status": "error", "status_msg": str(e)}

    def get_all_instances_status(self) -> List[Dict]:
        """Get status of all tracked instances."""
        return [self.get_instance_status(iid) for iid in self._instances]

    # ── Logs ──────────────────────────────────────────────────────────────

    def get_instance_logs(self, instance_id: str, tail: int = 100) -> str:
        """Get recent logs from an instance.

        Raises ContainerNotFoundError if the container is dead/missing.
        Returns empty string for other transient failures.
        """
        try:
            raw = _run_vastai(
                "logs", str(instance_id),
                "--tail", str(tail),
                api_key=self.api_key,
            )
            # Check for container error in the log output itself
            if "No such container" in raw:
                raise ContainerNotFoundError(
                    f"Container for instance {instance_id} not found"
                )
            return raw
        except ContainerNotFoundError:
            raise  # Re-raise so callers can handle it
        except RuntimeError as e:
            err_str = str(e)
            if "No such container" in err_str:
                raise ContainerNotFoundError(
                    f"Container for instance {instance_id} not found"
                ) from e
            logger.debug(f"Failed to get logs for {instance_id}: {e}")
            return ""
        except Exception as e:
            err_str = str(e)
            if "No such container" in err_str:
                raise ContainerNotFoundError(
                    f"Container for instance {instance_id} not found"
                ) from e
            logger.debug(f"Failed to get logs for {instance_id}: {e}")
            return ""

    # ── Destroy ───────────────────────────────────────────────────────────

    def destroy_instance(self, instance_id: str) -> bool:
        """Destroy an instance."""
        try:
            _run_vastai(
                "destroy", "instance", str(instance_id),
                api_key=self.api_key,
            )
            self._instances.pop(instance_id, None)
            logger.info(f"Destroyed instance {instance_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to destroy instance {instance_id}: {e}")
            return False

    def reboot_instance(self, instance_id: str) -> bool:
        """Reboot an instance."""
        try:
            _run_vastai(
                "reboot", "instance", str(instance_id),
                api_key=self.api_key,
            )
            logger.info(f"Rebooted instance {instance_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to reboot instance {instance_id}: {e}")
            return False

    def destroy_all(self) -> int:
        """Destroy all tracked instances. Returns number destroyed."""
        destroyed = 0
        for iid in list(self._instances.keys()):
            if self.destroy_instance(iid):
                destroyed += 1
        return destroyed

    @property
    def tracked_instance_ids(self) -> List[str]:
        return list(self._instances.keys())
