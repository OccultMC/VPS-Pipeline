"""Per-tab session state management.

Each browser tab gets a unique session UUID. Sessions store all state
for an independent scrape/deploy/build workflow.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class JobStatus(str, Enum):
    IDLE = "idle"
    SCRAPING = "scraping"
    UPLOADING = "uploading"
    DEPLOYING = "deploying"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class ShapeInfo:
    """Info about a loaded shape."""
    id: str
    geojson: dict  # GeoJSON Feature
    selected: bool = False
    path: Optional[str] = None  # e.g., US/California/Sacramento
    centroid_lat: float = 0.0
    centroid_lon: float = 0.0


@dataclass
class WorkerInfo:
    """Info about a deployed Vast.AI worker."""
    instance_id: int
    worker_number: int
    total_workers: int
    csv_filename: str
    status: str = "pending"
    panos_done: int = 0
    panos_total: int = 0
    eta_seconds: int = 0


@dataclass
class SessionState:
    """State for one browser tab / session."""
    session_id: str
    shapes: Dict[str, ShapeInfo] = field(default_factory=dict)
    job_status: JobStatus = JobStatus.IDLE
    job_message: str = ""

    # Scraping state
    scrape_progress: Dict[str, Any] = field(default_factory=dict)
    cancel_event: Optional[asyncio.Event] = field(default=None, repr=False)
    total_panos_scraped: int = 0

    # Deployment state
    deployed_workers: List[WorkerInfo] = field(default_factory=list)
    builder_instance_id: Optional[int] = None
    active_path: str = ""

    # Settings overrides
    worker_count_override: Optional[int] = None
    pipeline_image_override: str = ""
    builder_image_override: str = ""


class SessionManager:
    """Manages per-tab sessions."""

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}

    def create_session(self) -> SessionState:
        """Create a new session and return it."""
        session_id = str(uuid.uuid4())
        state = SessionState(session_id=session_id)
        self._sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get a session by ID, or None if not found."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str):
        """Remove a session."""
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    def get_all_instance_ids(self, session_id: str) -> List[int]:
        """Get all Vast.AI instance IDs for a session."""
        state = self.get_session(session_id)
        if not state:
            return []
        ids = [w.instance_id for w in state.deployed_workers]
        if state.builder_instance_id:
            ids.append(state.builder_instance_id)
        return ids
