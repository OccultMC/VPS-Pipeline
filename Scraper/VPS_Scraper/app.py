"""FastAPI application — routes, WebSocket, and deployment orchestration."""

import asyncio
import io
import json
import logging
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from session_manager import SessionManager, SessionState, JobStatus, ShapeInfo, WorkerInfo
from geocoder import reverse_geocode, get_city_name_from_path
from scraper_engine import scrape_polygon, split_csv_chunks, generate_csv_bytes
from vastai_client import VastAIClient
from r2_client import R2Client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

app = FastAPI(title="VPR Scraper WebUI")

# Global state
sessions = SessionManager()
ws_connections: Dict[str, List[WebSocket]] = {}  # session_id -> list of websockets

# Lazy-init clients (created on first use so missing env vars don't crash on import)
_r2: Optional[R2Client] = None
_vast: Optional[VastAIClient] = None


def get_r2() -> R2Client:
    global _r2
    if _r2 is None:
        _r2 = R2Client(
            endpoint_url=settings.r2_endpoint_url,
            access_key_id=settings.r2_access_key_id,
            secret_access_key=settings.r2_secret_access_key,
            bucket_name=settings.r2_bucket_name,
        )
    return _r2


def get_vast() -> VastAIClient:
    global _vast
    if _vast is None:
        _vast = VastAIClient(api_key=settings.vastai_api_key)
    return _vast


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = static_dir / "index.html"
    return FileResponse(str(index_path))


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
@app.post("/api/session")
async def create_session():
    state = sessions.create_session()
    return {"session_id": state.session_id}


# ---------------------------------------------------------------------------
# Shapefile import
# ---------------------------------------------------------------------------
@app.post("/api/import-shapefile")
async def import_shapefile(session_id: str, file: UploadFile = File(...)):
    state = sessions.get_session(session_id)
    if not state:
        raise HTTPException(404, "Session not found")

    content = await file.read()
    filename = file.filename or ""

    features = []
    if filename.endswith(".geojson") or filename.endswith(".json"):
        geojson = json.loads(content)
        if geojson.get("type") == "FeatureCollection":
            features = geojson.get("features", [])
        elif geojson.get("type") == "Feature":
            features = [geojson]
        elif geojson.get("type") in ("Polygon", "MultiPolygon"):
            features = [{"type": "Feature", "geometry": geojson, "properties": {}}]
    elif filename.endswith(".shp") or filename.endswith(".zip"):
        # Write to temp file and read with geopandas
        try:
            import geopandas as gpd
            suffix = ".zip" if filename.endswith(".zip") else ".shp"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                gdf = gpd.read_file(tmp_path)
                geojson = json.loads(gdf.to_json())
                features = geojson.get("features", [])
            finally:
                os.unlink(tmp_path)
        except ImportError:
            raise HTTPException(400, "geopandas not installed — use .geojson files instead")
    else:
        raise HTTPException(400, f"Unsupported file type: {filename}")

    # Store features as ShapeInfo
    state.shapes.clear()
    for i, feat in enumerate(features):
        geom = feat.get("geometry", {})
        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue

        shape_id = str(i)
        # Calculate centroid from first polygon ring
        coords = _get_polygon_coords(geom)
        if not coords:
            continue
        centroid_lat = sum(c[1] for c in coords) / len(coords)
        centroid_lon = sum(c[0] for c in coords) / len(coords)

        props = feat.get("properties", {})
        name = props.get("NAME", props.get("name", props.get("NAME_2", f"Shape_{i}")))

        state.shapes[shape_id] = ShapeInfo(
            id=shape_id,
            geojson=feat,
            centroid_lat=centroid_lat,
            centroid_lon=centroid_lon,
        )

    return {
        "count": len(state.shapes),
        "shapes": [
            {
                "id": s.id,
                "centroid": [s.centroid_lat, s.centroid_lon],
                "selected": s.selected,
                "path": s.path,
            }
            for s in state.shapes.values()
        ],
    }


def _get_polygon_coords(geom: dict) -> list:
    """Extract exterior ring coordinates from a Polygon or first polygon of MultiPolygon."""
    if geom["type"] == "Polygon":
        return geom["coordinates"][0]
    elif geom["type"] == "MultiPolygon":
        return geom["coordinates"][0][0]
    return []


# ---------------------------------------------------------------------------
# Shape selection + geocoding
# ---------------------------------------------------------------------------
@app.post("/api/select-shape")
async def select_shape(session_id: str, shape_id: str):
    state = sessions.get_session(session_id)
    if not state:
        raise HTTPException(404, "Session not found")
    shape = state.shapes.get(shape_id)
    if not shape:
        raise HTTPException(404, "Shape not found")

    shape.selected = not shape.selected

    # Auto-geocode on first selection if no path set
    if shape.selected and not shape.path:
        path = await reverse_geocode(shape.centroid_lat, shape.centroid_lon)
        if path:
            shape.path = path

    return {
        "id": shape.id,
        "selected": shape.selected,
        "path": shape.path,
    }


@app.post("/api/geocode")
async def geocode(session_id: str, shape_id: str):
    state = sessions.get_session(session_id)
    if not state:
        raise HTTPException(404, "Session not found")
    shape = state.shapes.get(shape_id)
    if not shape:
        raise HTTPException(404, "Shape not found")

    path = await reverse_geocode(shape.centroid_lat, shape.centroid_lon)
    if path:
        shape.path = path
    return {"path": shape.path}


@app.post("/api/set-path")
async def set_path(session_id: str, shape_id: str, path: str):
    """Manually set the R2 path for a shape."""
    state = sessions.get_session(session_id)
    if not state:
        raise HTTPException(404, "Session not found")
    shape = state.shapes.get(shape_id)
    if not shape:
        raise HTTPException(404, "Shape not found")
    shape.path = path.strip("/")
    return {"path": shape.path}


# ---------------------------------------------------------------------------
# Deploy Pipeline (GPU workers)
# ---------------------------------------------------------------------------
@app.post("/api/deploy-pipeline")
async def deploy_pipeline(session_id: str, worker_count: Optional[int] = None):
    state = sessions.get_session(session_id)
    if not state:
        raise HTTPException(404, "Session not found")

    # Find selected shapes
    selected = [s for s in state.shapes.values() if s.selected]
    if not selected:
        raise HTTPException(400, "No shapes selected")

    # Use first selected shape
    shape = selected[0]
    if not shape.path:
        raise HTTPException(400, "Shape has no path — geocode first")

    state.job_status = JobStatus.SCRAPING
    state.job_message = "Starting panorama scrape..."
    await _broadcast(session_id, {"type": "status", "status": "scraping", "message": state.job_message})

    # Run scrape + deploy in background
    asyncio.create_task(_scrape_and_deploy(state, shape, worker_count))
    return {"status": "started", "message": "Scraping started"}


async def _scrape_and_deploy(state: SessionState, shape: ShapeInfo, worker_count: Optional[int]):
    """Background task: scrape polygon → upload CSVs → deploy GPU workers."""
    session_id = state.session_id
    try:
        # Set up cancel event
        state.cancel_event = asyncio.Event()

        # Extract polygon coordinates (GeoJSON is [lon, lat], we need [lat, lon])
        coords_raw = _get_polygon_coords(shape.geojson.get("geometry", {}))
        polygon_coords = [(c[1], c[0]) for c in coords_raw]  # to (lat, lon)

        # Progress callback
        def on_progress(tiles_done, total_tiles, panos_found):
            state.scrape_progress = {
                "tiles_done": tiles_done,
                "total_tiles": total_tiles,
                "panos_found": panos_found,
            }
            asyncio.create_task(_broadcast(session_id, {
                "type": "scrape_progress",
                "tiles_done": tiles_done,
                "total_tiles": total_tiles,
                "panos_found": panos_found,
            }))

        # Scrape
        panos = await scrape_polygon(polygon_coords, on_progress, state.cancel_event)

        if state.cancel_event.is_set():
            state.job_status = JobStatus.CANCELLED
            state.job_message = "Scrape cancelled"
            await _broadcast(session_id, {"type": "status", "status": "cancelled"})
            return

        if not panos:
            state.job_status = JobStatus.ERROR
            state.job_message = "No panoramas found in selected area"
            await _broadcast(session_id, {"type": "status", "status": "error", "message": state.job_message})
            return

        state.total_panos_scraped = len(panos)
        state.job_message = f"Found {len(panos)} panoramas. Uploading CSVs..."
        state.job_status = JobStatus.UPLOADING
        await _broadcast(session_id, {"type": "status", "status": "uploading", "panos": len(panos)})

        # Split into chunks and upload to R2
        city_name = get_city_name_from_path(shape.path)
        wc = worker_count or state.worker_count_override
        chunks = split_csv_chunks(panos, city_name, worker_override=wc)

        r2 = get_r2()
        for filename, csv_data in chunks:
            r2_key = f"CSV/{shape.path}/{filename}"
            r2.upload_bytes(r2_key, csv_data, content_type="text/csv")

        state.job_message = f"Uploaded {len(chunks)} CSV chunks. Deploying workers..."
        state.job_status = JobStatus.DEPLOYING
        await _broadcast(session_id, {"type": "status", "status": "deploying", "chunks": len(chunks)})

        # Search for GPU offers
        vast = get_vast()
        offers = await vast.search_gpu_offers(num_results=len(chunks) * 3)
        if len(offers) < len(chunks):
            state.job_status = JobStatus.ERROR
            state.job_message = f"Only {len(offers)} GPU offers found, need {len(chunks)}"
            await _broadcast(session_id, {"type": "status", "status": "error", "message": state.job_message})
            return

        # Deploy one worker per chunk
        docker_image = state.pipeline_image_override or settings.pipeline_docker_image
        state.deployed_workers = []
        state.active_path = shape.path

        for i, (filename, _) in enumerate(chunks):
            offer = offers[i]
            offer_id = offer.get("id") or offer.get("ask_contract_id")

            env_vars = {
                "R2_ENDPOINT_URL": settings.r2_endpoint_url,
                "R2_ACCESS_KEY_ID": settings.r2_access_key_id,
                "R2_SECRET_ACCESS_KEY": settings.r2_secret_access_key,
                "R2_BUCKET_NAME": settings.r2_bucket_name,
                "CSV_R2_PATH": f"CSV/{shape.path}/{filename}",
                "OUTPUT_R2_PATH": f"Features/{shape.path}/",
                "LOGS_R2_PATH": f"Logs/{shape.path}/",
                "WORKER_NUMBER": str(i + 1),
                "TOTAL_WORKERS": str(len(chunks)),
                "CITY_NAME": city_name,
                "VASTAI_API_KEY": settings.vastai_api_key,
            }

            result = await vast.create_instance(
                offer_id=offer_id,
                docker_image=docker_image,
                env_vars=env_vars,
                disk_gb=100,
            )

            if result:
                instance_id = result.get("new_contract")
                worker = WorkerInfo(
                    instance_id=instance_id,
                    worker_number=i + 1,
                    total_workers=len(chunks),
                    csv_filename=filename,
                    status="deploying",
                    panos_total=len(panos) // len(chunks),
                )
                state.deployed_workers.append(worker)

                # Set VASTAI_INSTANCE_ID after creation
                # (worker reads it from env on startup, but we know it now)

        state.job_status = JobStatus.RUNNING
        state.job_message = f"Deployed {len(state.deployed_workers)}/{len(chunks)} workers"
        await _broadcast(session_id, {
            "type": "status",
            "status": "running",
            "workers": len(state.deployed_workers),
            "total_chunks": len(chunks),
        })

    except Exception as e:
        logger.exception("Deploy pipeline failed")
        state.job_status = JobStatus.ERROR
        state.job_message = str(e)
        await _broadcast(session_id, {"type": "status", "status": "error", "message": str(e)})


# ---------------------------------------------------------------------------
# Deploy Builder (CPU instance)
# ---------------------------------------------------------------------------
@app.post("/api/deploy-builder")
async def deploy_builder(
    session_id: str,
    index_type: str = "PQ",
    m: int = 256,
    nbits: int = 8,
    training_samples: int = 1000000,
    nlist: int = 0,
    nprobe: int = 100,
):
    state = sessions.get_session(session_id)
    if not state:
        raise HTTPException(404, "Session not found")

    if not state.active_path:
        # Find path from selected shape
        selected = [s for s in state.shapes.values() if s.selected]
        if not selected or not selected[0].path:
            raise HTTPException(400, "No active path — deploy pipeline first or select a shape")
        state.active_path = selected[0].path

    vast = get_vast()
    offers = await vast.search_cpu_offers(num_results=5)
    if not offers:
        raise HTTPException(500, "No CPU offers available")

    offer = offers[0]
    offer_id = offer.get("id") or offer.get("ask_contract_id")

    docker_image = state.builder_image_override or settings.builder_docker_image
    env_vars = {
        "R2_ENDPOINT_URL": settings.r2_endpoint_url,
        "R2_ACCESS_KEY_ID": settings.r2_access_key_id,
        "R2_SECRET_ACCESS_KEY": settings.r2_secret_access_key,
        "R2_BUCKET_NAME": settings.r2_bucket_name,
        "FEATURES_R2_PATH": f"Features/{state.active_path}/",
        "OUTPUT_R2_PATH": f"Index/{state.active_path}/",
        "INDEX_TYPE": index_type,
        "M": str(m),
        "NBITS": str(nbits),
        "TRAINING_SAMPLES": str(training_samples),
        "NLIST": str(nlist),
        "NPROBE": str(nprobe),
        "VASTAI_API_KEY": settings.vastai_api_key,
    }

    result = await vast.create_instance(
        offer_id=offer_id,
        docker_image=docker_image,
        env_vars=env_vars,
        disk_gb=700,
    )

    if not result:
        raise HTTPException(500, "Failed to create builder instance")

    state.builder_instance_id = result.get("new_contract")
    return {
        "instance_id": state.builder_instance_id,
        "offer": {"gpu": offer.get("gpu_name"), "cost": offer.get("dph_total")},
    }


# ---------------------------------------------------------------------------
# Logs polling
# ---------------------------------------------------------------------------
@app.get("/api/logs/{session_id}")
async def get_logs(session_id: str):
    state = sessions.get_session(session_id)
    if not state:
        raise HTTPException(404, "Session not found")

    if not state.active_path:
        return {"workers": [], "path": ""}

    r2 = get_r2()
    prefix = f"Logs/{state.active_path}/"
    keys = r2.list_objects(prefix)

    worker_logs = []
    for key in keys:
        if key.endswith("_Logs.json"):
            log_data = r2.download_json(key)
            if log_data:
                worker_logs.append(log_data)

    # Sort by worker number
    worker_logs.sort(key=lambda x: x.get("worker", 0))

    # Check if all done
    all_done = (
        len(worker_logs) > 0
        and all(w.get("status") == "done" for w in worker_logs)
    )
    if all_done and state.job_status == JobStatus.RUNNING:
        state.job_status = JobStatus.DONE
        state.job_message = "All workers completed"
        await _broadcast(session_id, {"type": "status", "status": "done"})

    return {
        "workers": worker_logs,
        "path": state.active_path,
        "job_status": state.job_status.value,
        "all_done": all_done,
    }


# ---------------------------------------------------------------------------
# Cancel / Destroy
# ---------------------------------------------------------------------------
@app.post("/api/cancel-scrape")
async def cancel_scrape(session_id: str):
    state = sessions.get_session(session_id)
    if not state:
        raise HTTPException(404, "Session not found")
    if state.cancel_event:
        state.cancel_event.set()
    state.job_status = JobStatus.CANCELLED
    return {"status": "cancelled"}


@app.post("/api/destroy-all")
async def destroy_all(session_id: str):
    state = sessions.get_session(session_id)
    if not state:
        raise HTTPException(404, "Session not found")

    instance_ids = sessions.get_all_instance_ids(session_id)
    if not instance_ids:
        return {"destroyed": 0}

    vast = get_vast()
    destroyed = await vast.destroy_all_instances(instance_ids)

    state.deployed_workers.clear()
    state.builder_instance_id = None
    state.job_status = JobStatus.IDLE
    state.job_message = f"Destroyed {destroyed} instances"

    await _broadcast(session_id, {"type": "status", "status": "idle", "message": state.job_message})
    return {"destroyed": destroyed}


# ---------------------------------------------------------------------------
# Session info
# ---------------------------------------------------------------------------
@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    state = sessions.get_session(session_id)
    if not state:
        raise HTTPException(404, "Session not found")

    return {
        "session_id": state.session_id,
        "job_status": state.job_status.value,
        "job_message": state.job_message,
        "shapes": {
            sid: {
                "id": s.id,
                "selected": s.selected,
                "path": s.path,
                "centroid": [s.centroid_lat, s.centroid_lon],
            }
            for sid, s in state.shapes.items()
        },
        "workers": [
            {
                "instance_id": w.instance_id,
                "worker_number": w.worker_number,
                "csv_filename": w.csv_filename,
                "status": w.status,
                "panos_done": w.panos_done,
                "panos_total": w.panos_total,
            }
            for w in state.deployed_workers
        ],
        "builder_instance_id": state.builder_instance_id,
        "active_path": state.active_path,
        "total_panos_scraped": state.total_panos_scraped,
    }


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    # Register connection
    if session_id not in ws_connections:
        ws_connections[session_id] = []
    ws_connections[session_id].append(websocket)

    # Ensure session exists
    state = sessions.get_session(session_id)
    if not state:
        state = sessions.create_session()
        # Override the auto-generated ID with the one from the URL
        sessions._sessions.pop(state.session_id, None)
        state.session_id = session_id
        sessions._sessions[session_id] = state

    try:
        # Send current state
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "job_status": state.job_status.value,
        })

        # Keep alive and handle messages
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    finally:
        if session_id in ws_connections:
            ws_connections[session_id] = [
                ws for ws in ws_connections[session_id] if ws != websocket
            ]


async def _broadcast(session_id: str, message: dict):
    """Broadcast a message to all WebSocket connections for a session."""
    connections = ws_connections.get(session_id, [])
    dead = []
    for ws in connections:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connections.remove(ws)


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    missing = settings.validate()
    if missing:
        logger.warning(f"Missing env vars (app may fail on use): {', '.join(missing)}")
    logger.info("VPR Scraper WebUI started")


@app.on_event("shutdown")
async def shutdown():
    global _vast
    if _vast:
        await _vast.close()
    logger.info("VPR Scraper WebUI stopped")


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
