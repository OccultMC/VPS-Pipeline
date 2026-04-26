"""
LookAround Scraper — local debug Web GUI.

Forked from VPS_Scraper/web_app.py with all deployment infrastructure removed.

Usage:
    cd LookAround_Scraper
    python web_app.py          # http://localhost:5000
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict

from flask import (
    Flask, jsonify, render_template, request, send_from_directory,
)
from flask_socketio import SocketIO, emit, join_room

from apple_scraper import scrape_polygon, stitch_faces
from cylindrical_project import trim_wrap_overlap, pad_to_2to1
from equirect_reproject import reproject_to_equirect
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT = Path(__file__).resolve().parent
DOWNLOADS_DIR = ROOT / "downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET", "dev-secret-key")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

sessions: Dict[str, Dict[str, Any]] = {}


def _get_session(sid: str) -> Dict[str, Any]:
    if not sid:
        sid = str(uuid.uuid4())
    if sid not in sessions:
        sessions[sid] = {"shapes_data": None}
    return sessions[sid]


# ───────────────────────────────────────── Routes ─────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/load-shapes", methods=["POST"])
def load_shapes():
    """Load shapes from GeoJSON, Shapefile (.shp/.zip), KML, or GeoPackage."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    all_files = request.files.getlist("file")
    sid = request.form.get("session_id", "")

    main_file = None
    for uf in all_files:
        ext_ = (uf.filename or "").rsplit(".", 1)[-1].lower()
        if ext_ in ("shp", "geojson", "json", "zip", "kml", "gpkg"):
            main_file = uf
            break
    if main_file is None:
        main_file = all_files[0]

    fname = main_file.filename or ""
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""

    try:
        if ext in ("geojson", "json"):
            content = main_file.read().decode("utf-8")
            geojson = json.loads(content)
        elif ext in ("shp", "dbf", "zip", "kml", "gpkg"):
            import geopandas as gpd
            os.environ["SHAPE_RESTORE_SHX"] = "YES"
            tmpdir = tempfile.mkdtemp()
            try:
                for uf in all_files:
                    uf.save(os.path.join(tmpdir, uf.filename or "upload"))
                if ext == "zip":
                    main_path = f"zip://{os.path.join(tmpdir, fname)}"
                else:
                    main_path = os.path.join(tmpdir, fname)
                gdf = gpd.read_file(main_path)
                if gdf.crs is None:
                    gdf = gdf.set_crs(epsg=4326)
                elif gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(epsg=4326)
                geojson = json.loads(gdf.to_json())
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            return jsonify({"error": f"Unsupported file type: .{ext}"}), 400

        sess = _get_session(sid)
        sess["shapes_data"] = geojson
        return jsonify({"geojson": geojson, "filename": fname})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/downloads/<pano_id>/<path:filename>")
def serve_download(pano_id: str, filename: str):
    pano_dir = DOWNLOADS_DIR / pano_id
    return send_from_directory(str(pano_dir), filename)


# ───────────────────────────────────────── SocketIO ───────────────────────────

@socketio.on("connect")
def on_connect():
    sid = request.args.get("session_id") or str(uuid.uuid4())
    join_room(sid)
    emit("connected", {"session_id": sid})


@socketio.on("scrape_polygon")
def on_scrape_polygon(data):
    sid = data.get("session_id") or request.sid
    coords = data.get("coords") or []
    zoom = int(data.get("zoom", 0))

    if not coords or len(coords) < 3:
        emit("scrape_error", {"message": "polygon needs at least 3 vertices"}, room=sid)
        return

    def _run():
        def progress_cb(event, payload):
            socketio.emit(event, payload, room=sid)
        try:
            result = scrape_polygon(
                coords, zoom=zoom, out_root=str(DOWNLOADS_DIR),
                progress_cb=progress_cb,
            )
            face_urls = [
                f"/downloads/{result.pano_id}/{os.path.basename(p)}"
                for p in result.face_paths
            ]
            csv_url = f"/downloads/{result.pano_id}/meta.csv"
            socketio.emit("scrape_done", {
                "pano_id": result.pano_id,
                "lat": result.lat,
                "lon": result.lon,
                "capture_date": result.capture_date,
                "heading": result.heading,
                "face_urls": face_urls,
                "csv_url": csv_url,
            }, room=sid)
        except Exception as e:
            logger.exception("scrape failed")
            socketio.emit("scrape_error", {"message": str(e)}, room=sid)

    threading.Thread(target=_run, daemon=True).start()


VIEWER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>360 Viewer - __PANO_ID__</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.css">
<style>
  html, body { margin:0; height:100%; background:#000; color:#eee; font-family:system-ui,sans-serif; }
  #panorama { height:100vh; }
  #info { position:absolute; top:10px; left:10px; background:rgba(0,0,0,0.6); padding:8px 12px; border-radius:4px; font-size:12px; z-index:10; }
</style>
</head>
<body>
<div id="info">pano __PANO_ID__ — drag to pan, scroll to zoom</div>
<div id="panorama"></div>
<script src="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.js"></script>
<script>
pannellum.viewer('panorama', {
    type: 'equirectangular',
    panorama: '/downloads/__PANO_ID__/equirect.jpg',
    autoLoad: true,
    showControls: true,
    showZoomCtrl: true,
    showFullscreenCtrl: true,
    minPitch: -90,
    maxPitch: 90,
    hfov: 100,
    minHfov: 30,
    maxHfov: 120,
    backgroundColor: [0, 0, 0]
});
</script>
</body>
</html>
"""


@app.route("/viewer/<pano_id>")
def viewer(pano_id: str):
    pano_dir = DOWNLOADS_DIR / pano_id
    if not (pano_dir / "equirect.jpg").exists():
        return f"equirect.jpg not found for {pano_id}. Stitch + project first.", 404
    return VIEWER_HTML.replace("__PANO_ID__", pano_id)


@socketio.on("make_equirect")
def on_make_equirect(data):
    """Build equirect.jpg from the existing stitched.jpg + face widths."""
    sid = data.get("session_id") or request.sid
    pano_id = data.get("pano_id") or ""
    if not pano_id:
        emit("equirect_error", {"message": "missing pano_id"}, room=sid)
        return

    pano_dir = DOWNLOADS_DIR / pano_id
    if not pano_dir.is_dir():
        emit("equirect_error", {"message": f"pano dir not found: {pano_id}"}, room=sid)
        return

    def _run():
        try:
            meta_json = pano_dir / "metadata.json"
            if not meta_json.exists():
                raise RuntimeError("metadata.json missing — re-scrape this pano")

            out_path = reproject_to_equirect(str(pano_dir), out_name="equirect.jpg")
            equi = Image.open(out_path)

            socketio.emit("equirect_done", {
                "pano_id": pano_id,
                "equirect_url": f"/downloads/{pano_id}/equirect.jpg",
                "viewer_url": f"/viewer/{pano_id}",
                "width": equi.width,
                "height": equi.height,
            }, room=sid)
        except Exception as e:
            logger.exception("equirect failed")
            socketio.emit("equirect_error", {"message": str(e)}, room=sid)

    threading.Thread(target=_run, daemon=True).start()


@socketio.on("stitch_pano")
def on_stitch_pano(data):
    sid = data.get("session_id") or request.sid
    pano_id = data.get("pano_id") or ""
    overlap_pct = float(data.get("overlap_pct", 3.0))

    if not pano_id:
        emit("stitch_error", {"message": "missing pano_id"}, room=sid)
        return

    pano_dir = DOWNLOADS_DIR / pano_id
    if not pano_dir.is_dir():
        emit("stitch_error", {"message": f"pano dir not found: {pano_id}"}, room=sid)
        return

    def _run():
        try:
            out_path = stitch_faces(str(pano_dir), overlap_pct=overlap_pct)
            stitched_url = f"/downloads/{pano_id}/{os.path.basename(out_path)}"
            socketio.emit("stitch_done", {
                "pano_id": pano_id,
                "stitched_url": stitched_url,
            }, room=sid)
        except Exception as e:
            logger.exception("stitch failed")
            socketio.emit("stitch_error", {"message": str(e)}, room=sid)

    threading.Thread(target=_run, daemon=True).start()


# ───────────────────────────────────────── Main ───────────────────────────────

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)
