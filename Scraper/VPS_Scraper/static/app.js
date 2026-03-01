/* VPR Scraper — Single-page app with deck.gl map */

// ─── State ───────────────────────────────────────────────────────────
let sessionId = null;
let ws = null;
let shapes = {};           // id -> { geojson, selected, path, centroid }
let selectedShapeId = null; // currently focused shape
let deckgl = null;
let logPollTimer = null;

// ─── API helpers ─────────────────────────────────────────────────────
async function api(method, path, body) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) opts.body = JSON.stringify(body);
    const resp = await fetch(path, opts);
    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`${resp.status}: ${text}`);
    }
    return resp.json();
}

async function apiForm(path, formData) {
    const resp = await fetch(path, { method: 'POST', body: formData });
    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`${resp.status}: ${text}`);
    }
    return resp.json();
}

// ─── Session + WebSocket ─────────────────────────────────────────────
async function initSession() {
    const data = await api('POST', '/api/session');
    sessionId = data.session_id;
    connectWS();
}

function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws/${sessionId}`);

    ws.onopen = () => {
        document.getElementById('ws-status').classList.add('connected');
        document.getElementById('ws-status').classList.remove('error');
    };

    ws.onclose = () => {
        document.getElementById('ws-status').classList.remove('connected');
        document.getElementById('ws-status').classList.add('error');
        // Reconnect after 3s
        setTimeout(connectWS, 3000);
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleWSMessage(msg);
    };
}

function handleWSMessage(msg) {
    switch (msg.type) {
        case 'connected':
            setStatus(msg.job_status || 'idle');
            break;

        case 'scrape_progress':
            updateScrapeProgress(msg);
            break;

        case 'status':
            setStatus(msg.status, msg.message);
            if (msg.status === 'running' || msg.status === 'done') {
                startLogPolling();
            }
            if (msg.status === 'idle' || msg.status === 'error' || msg.status === 'cancelled') {
                stopLogPolling();
            }
            break;

        case 'pong':
            break;
    }
}

// ─── Map (deck.gl + MapLibre) ────────────────────────────────────────
function initMap() {
    deckgl = new deck.DeckGL({
        container: 'map-canvas',
        mapStyle: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
        initialViewState: {
            longitude: -98.5795,
            latitude: 39.8283,
            zoom: 4,
            pitch: 0,
            bearing: 0,
        },
        controller: true,
        layers: [],
        getTooltip: ({ object }) => {
            if (!object) return null;
            const props = object.properties || {};
            return { text: props.NAME || props.name || props.NAME_2 || '' };
        },
    });
}

function updateMapLayers() {
    const features = Object.values(shapes).map(s => {
        const feat = JSON.parse(JSON.stringify(s.geojson));
        feat.properties = feat.properties || {};
        feat.properties._selected = s.selected;
        feat.properties._id = s.id;
        return feat;
    });

    const geojsonLayer = new deck.GeoJsonLayer({
        id: 'shapes',
        data: { type: 'FeatureCollection', features },
        filled: true,
        stroked: true,
        getFillColor: (f) => f.properties._selected ? [76, 175, 80, 80] : [74, 144, 217, 60],
        getLineColor: (f) => f.properties._selected ? [76, 175, 80, 200] : [74, 144, 217, 160],
        getLineWidth: 2,
        lineWidthUnits: 'pixels',
        pickable: true,
        onClick: (info) => {
            if (info.object) {
                const id = info.object.properties._id;
                toggleShape(id);
            }
        },
        updateTriggers: {
            getFillColor: Object.values(shapes).map(s => s.selected),
            getLineColor: Object.values(shapes).map(s => s.selected),
        },
    });

    deckgl.setProps({ layers: [geojsonLayer] });
}

// ─── File import ─────────────────────────────────────────────────────
function initFileImport() {
    const drop = document.getElementById('file-drop');
    const input = document.getElementById('file-input');

    drop.addEventListener('click', () => input.click());

    drop.addEventListener('dragover', (e) => {
        e.preventDefault();
        drop.classList.add('dragover');
    });

    drop.addEventListener('dragleave', () => {
        drop.classList.remove('dragover');
    });

    drop.addEventListener('drop', (e) => {
        e.preventDefault();
        drop.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    input.addEventListener('change', () => {
        if (input.files.length) handleFile(input.files[0]);
        input.value = '';
    });
}

async function handleFile(file) {
    try {
        setStatus('loading', `Importing ${file.name}...`);
        const fd = new FormData();
        fd.append('file', file);
        const data = await apiForm(`/api/import-shapefile?session_id=${sessionId}`, fd);

        shapes = {};
        for (const s of data.shapes) {
            shapes[s.id] = {
                id: s.id,
                geojson: null, // will get from full response
                selected: s.selected,
                path: s.path,
                centroid: s.centroid,
            };
        }

        // We need the full GeoJSON — re-read from file locally
        const text = await file.text();
        let geojson;
        try {
            geojson = JSON.parse(text);
        } catch {
            // For .shp files, we rely on server-parsed shapes
            // Just use empty geometry — the server has it
        }

        if (geojson) {
            let features = [];
            if (geojson.type === 'FeatureCollection') features = geojson.features || [];
            else if (geojson.type === 'Feature') features = [geojson];
            else if (geojson.type === 'Polygon' || geojson.type === 'MultiPolygon') {
                features = [{ type: 'Feature', geometry: geojson, properties: {} }];
            }

            let idx = 0;
            for (const feat of features) {
                const type = feat.geometry?.type;
                if (type !== 'Polygon' && type !== 'MultiPolygon') continue;
                const id = String(idx);
                if (shapes[id]) {
                    shapes[id].geojson = feat;
                }
                idx++;
            }
        }

        renderShapeList();
        updateMapLayers();
        document.getElementById('shapes-section').style.display = '';
        document.getElementById('shape-count').textContent = data.count;

        // Fly to shapes
        if (data.shapes.length > 0) {
            const s = data.shapes[0];
            deckgl.setProps({
                initialViewState: {
                    longitude: s.centroid[1],
                    latitude: s.centroid[0],
                    zoom: 10,
                    transitionDuration: 1000,
                },
            });
        }

        setStatus('idle', `Imported ${data.count} shapes`);
    } catch (err) {
        setStatus('error', err.message);
    }
}

// ─── Shape list ──────────────────────────────────────────────────────
function renderShapeList() {
    const list = document.getElementById('shape-list');
    list.innerHTML = '';

    for (const s of Object.values(shapes)) {
        const props = s.geojson?.properties || {};
        const name = props.NAME || props.name || props.NAME_2 || `Shape ${s.id}`;

        const item = document.createElement('div');
        item.className = 'shape-item' + (s.selected ? ' selected' : '');
        item.innerHTML = `
            <div class="shape-color ${s.selected ? 'selected' : 'unselected'}"></div>
            <span class="shape-name">${name}</span>
            ${s.path ? `<span class="shape-path">${s.path}</span>` : ''}
        `;
        item.addEventListener('click', () => {
            selectedShapeId = s.id;
            toggleShape(s.id);
        });
        list.appendChild(item);
    }

    updateDeployButton();
}

async function toggleShape(id) {
    try {
        const data = await api('POST', `/api/select-shape?session_id=${sessionId}&shape_id=${id}`);
        shapes[id].selected = data.selected;
        if (data.path) shapes[id].path = data.path;
        selectedShapeId = id;

        renderShapeList();
        updateMapLayers();
        showPathField();
    } catch (err) {
        setStatus('error', err.message);
    }
}

function showPathField() {
    const field = document.getElementById('path-field');
    const input = document.getElementById('path-input');

    if (selectedShapeId && shapes[selectedShapeId]) {
        field.style.display = '';
        input.value = shapes[selectedShapeId].path || '';
        document.getElementById('status-path').textContent = shapes[selectedShapeId].path || '';
    } else {
        field.style.display = 'none';
    }
}

function updateDeployButton() {
    const hasSelected = Object.values(shapes).some(s => s.selected);
    document.getElementById('deploy-btn').disabled = !hasSelected;
    document.getElementById('build-btn').disabled = !hasSelected;
}

// ─── Path save ───────────────────────────────────────────────────────
document.getElementById('path-save-btn').addEventListener('click', async () => {
    if (!selectedShapeId) return;
    const path = document.getElementById('path-input').value.trim();
    if (!path) return;
    try {
        await api('POST', `/api/set-path?session_id=${sessionId}&shape_id=${selectedShapeId}&path=${encodeURIComponent(path)}`);
        shapes[selectedShapeId].path = path;
        renderShapeList();
        document.getElementById('status-path').textContent = path;
    } catch (err) {
        setStatus('error', err.message);
    }
});

// ─── Deploy Pipeline ─────────────────────────────────────────────────
document.getElementById('deploy-btn').addEventListener('click', async () => {
    try {
        const workerCount = document.getElementById('worker-count').value;
        const image = document.getElementById('pipeline-image').value.trim();

        let url = `/api/deploy-pipeline?session_id=${sessionId}`;
        if (workerCount) url += `&worker_count=${workerCount}`;

        document.getElementById('deploy-btn').disabled = true;
        document.getElementById('cancel-btn').style.display = '';
        document.getElementById('scrape-progress').classList.add('visible');

        await api('POST', url);
    } catch (err) {
        setStatus('error', err.message);
        document.getElementById('deploy-btn').disabled = false;
    }
});

document.getElementById('cancel-btn').addEventListener('click', async () => {
    try {
        await api('POST', `/api/cancel-scrape?session_id=${sessionId}`);
        document.getElementById('cancel-btn').style.display = 'none';
        document.getElementById('scrape-progress').classList.remove('visible');
        document.getElementById('deploy-btn').disabled = false;
    } catch (err) {
        setStatus('error', err.message);
    }
});

document.getElementById('destroy-btn').addEventListener('click', async () => {
    if (!confirm('Destroy ALL deployed instances for this session?')) return;
    try {
        const data = await api('POST', `/api/destroy-all?session_id=${sessionId}`);
        setStatus('idle', `Destroyed ${data.destroyed} instances`);
        document.getElementById('destroy-btn').style.display = 'none';
        stopLogPolling();
    } catch (err) {
        setStatus('error', err.message);
    }
});

// ─── Deploy Builder ──────────────────────────────────────────────────
document.getElementById('build-btn').addEventListener('click', async () => {
    try {
        const params = new URLSearchParams({
            session_id: sessionId,
            index_type: document.getElementById('index-type').value,
            m: document.getElementById('pq-m').value,
            nbits: document.getElementById('pq-nbits').value,
            training_samples: document.getElementById('training-samples').value,
            nlist: document.getElementById('nlist').value,
            nprobe: document.getElementById('nprobe').value,
        });

        document.getElementById('build-btn').disabled = true;
        const data = await api('POST', `/api/deploy-builder?${params}`);
        setStatus('running', `Builder deployed (instance ${data.instance_id})`);
        document.getElementById('destroy-btn').style.display = '';
    } catch (err) {
        setStatus('error', err.message);
        document.getElementById('build-btn').disabled = false;
    }
});

// ─── Scrape progress ─────────────────────────────────────────────────
function updateScrapeProgress(msg) {
    const pct = msg.total_tiles > 0
        ? Math.round((msg.tiles_done / msg.total_tiles) * 100)
        : 0;
    document.getElementById('scrape-progress-bar').style.width = pct + '%';
    document.getElementById('scrape-stats').textContent =
        `Tiles: ${msg.tiles_done.toLocaleString()} / ${msg.total_tiles.toLocaleString()} | ` +
        `Panos found: ${msg.panos_found.toLocaleString()}`;
}

// ─── Log polling ─────────────────────────────────────────────────────
function startLogPolling() {
    if (logPollTimer) return;
    pollLogs();
    logPollTimer = setInterval(pollLogs, 10000);
    // Switch to logs tab
    switchTab('logs');
}

function stopLogPolling() {
    if (logPollTimer) {
        clearInterval(logPollTimer);
        logPollTimer = null;
    }
}

async function pollLogs() {
    try {
        const data = await api('GET', `/api/logs/${sessionId}`);
        renderWorkerLogs(data.workers);

        if (data.all_done) {
            stopLogPolling();
            setStatus('done', 'All workers completed');
            document.getElementById('build-btn').disabled = false;
        }
    } catch (err) {
        // Ignore polling errors
    }
}

function renderWorkerLogs(workers) {
    const container = document.getElementById('worker-logs');

    if (!workers || workers.length === 0) {
        container.innerHTML = '<div style="color:var(--text-secondary);font-size:13px;text-align:center;padding:20px">No worker logs yet</div>';
        return;
    }

    container.innerHTML = workers.map(w => {
        const pct = w.panos_total > 0
            ? Math.round((w.panos_done / w.panos_total) * 100)
            : 0;
        const eta = w.eta_seconds > 0 ? formatETA(w.eta_seconds) : '--';
        const status = w.status || 'pending';

        return `
            <div class="worker-card">
                <div class="worker-header">
                    <span class="worker-name">Worker ${w.worker}/${w.total_workers}</span>
                    <span class="worker-status ${status}">${status}</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width:${pct}%"></div>
                </div>
                <div class="worker-stats">
                    <span>${w.panos_done?.toLocaleString() || 0} / ${w.panos_total?.toLocaleString() || '?'} panos</span>
                    <span>ETA: ${eta}</span>
                </div>
            </div>
        `;
    }).join('');
}

function formatETA(seconds) {
    if (seconds <= 0) return '--';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    if (h > 0) return `${h}h ${m}m`;
    return `${m}m`;
}

// ─── Status ──────────────────────────────────────────────────────────
function setStatus(status, message) {
    const el = document.getElementById('status-text');
    const labels = {
        idle: 'Ready',
        loading: 'Loading...',
        scraping: 'Scraping...',
        uploading: 'Uploading...',
        deploying: 'Deploying...',
        running: 'Workers running',
        done: 'Complete',
        error: 'Error',
        cancelled: 'Cancelled',
    };
    el.textContent = message || labels[status] || status;

    // Update UI elements based on status
    if (status === 'running') {
        document.getElementById('cancel-btn').style.display = 'none';
        document.getElementById('scrape-progress').classList.remove('visible');
        document.getElementById('destroy-btn').style.display = '';
    }
    if (status === 'done' || status === 'idle') {
        document.getElementById('deploy-btn').disabled = false;
        document.getElementById('cancel-btn').style.display = 'none';
        document.getElementById('scrape-progress').classList.remove('visible');
    }
    if (status === 'error' || status === 'cancelled') {
        document.getElementById('deploy-btn').disabled = false;
        document.getElementById('cancel-btn').style.display = 'none';
        document.getElementById('scrape-progress').classList.remove('visible');
    }
}

// ─── Tabs ────────────────────────────────────────────────────────────
function initTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });
}

function switchTab(name) {
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `tab-${name}`));
}

// ─── Settings toggles ───────────────────────────────────────────────
function initSettings() {
    document.getElementById('deploy-settings-toggle').addEventListener('click', () => {
        document.getElementById('deploy-settings').classList.toggle('open');
    });
    document.getElementById('build-settings-toggle').addEventListener('click', () => {
        document.getElementById('build-settings').classList.toggle('open');
    });
}

// ─── Keep-alive ping ─────────────────────────────────────────────────
setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
    }
}, 30000);

// ─── Init ────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    initMap();
    initFileImport();
    initTabs();
    initSettings();
    await initSession();
});
