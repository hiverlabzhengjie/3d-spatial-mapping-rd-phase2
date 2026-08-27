# ruff: noqa: E501
"""Small local operator console for the XR02 WP4 live demonstrator."""

from __future__ import annotations

import json
import threading
import webbrowser
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol


class OperatorController(Protocol):
    def start(self) -> dict[str, object]: ...

    def stop(self) -> dict[str, object]: ...

    def open_rerun(self) -> dict[str, object]: ...

    def reset_trails(self) -> dict[str, object]: ...

    def export_evidence_snapshot(self) -> dict[str, object]: ...

    def status(self) -> dict[str, object]: ...


class XR02OperatorServer:
    """Loopback-only HTTP shell; it never receives or returns RTSP endpoints."""

    def __init__(self, controller: OperatorController, port: int = 8094) -> None:
        if port != 0 and not 1024 <= port <= 65535:
            raise ValueError("operator port must be within 1024..65535")
        handler = _handler_factory(controller)
        self._server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        address = self._server.server_address
        host, port = address[0], address[1]
        if isinstance(host, bytes):
            host = host.decode("ascii")
        return f"http://{host}:{port}/"

    def start(self, *, open_browser: bool = True) -> None:
        if self._thread is not None:
            raise RuntimeError("operator server is already running")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="xr02-wp4-operator",
            daemon=True,
        )
        self._thread.start()
        if open_browser:
            webbrowser.open(self.url)

    def serve_forever(self, *, open_browser: bool = True) -> None:
        if open_browser:
            webbrowser.open(self.url)
        try:
            self._server.serve_forever()
        finally:
            self._server.server_close()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def _handler_factory(controller: OperatorController) -> type[BaseHTTPRequestHandler]:
    routes: dict[str, Callable[[], dict[str, object]]] = {
        "/api/start": controller.start,
        "/api/stop": controller.stop,
        "/api/open-rerun": controller.open_rerun,
        "/api/reset-trails": controller.reset_trails,
        "/api/export": controller.export_evidence_snapshot,
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "XR02Operator/1"

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._send(HTTPStatus.OK, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if self.path == "/api/status":
                self._json(HTTPStatus.OK, controller.status())
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            action = routes.get(self.path)
            if action is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                result = action()
            except Exception as error:
                self._json(
                    HTTPStatus.CONFLICT,
                    {"error": type(error).__name__, "detail": str(error)},
                )
                return
            self._json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _json(self, status: HTTPStatus, value: object) -> None:
            payload = json.dumps(value, sort_keys=True).encode("utf-8")
            if b"rtsp://" in payload.lower() or b"rtsps://" in payload.lower():
                payload = b'{"error":"credential_safety_violation"}'
                status = HTTPStatus.INTERNAL_SERVER_ERROR
            self._send(status, payload, "application/json")

        def _send(self, status: HTTPStatus, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

    return Handler


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>XR02 Live Global Tracking</title><style>
:root{color-scheme:dark;--bg:#0b1020;--card:#151c30;--line:#293553;--ink:#e8edfa;
--muted:#99a8c7;--ok:#35d07f;--warn:#ffca5f;--bad:#ff6b74;--accent:#7ba6ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#182443,var(--bg) 45%);
font:15px/1.45 system-ui;color:var(--ink)}main{max-width:1180px;margin:auto;padding:28px}
h1{font-size:28px;margin:0}.sub{color:var(--muted);margin:5px 0 22px}.bar{display:flex;gap:10px;
flex-wrap:wrap;margin-bottom:20px}button{border:1px solid var(--line);border-radius:10px;padding:10px 15px;
background:#1b2742;color:var(--ink);font-weight:650;cursor:pointer}button.primary{background:#315eaf}
button:hover{filter:brightness(1.15)}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.card{background:rgba(21,28,48,.94);border:1px solid var(--line);border-radius:14px;padding:15px;
box-shadow:0 12px 35px #0003}.wide{grid-column:span 2}.label{font-size:12px;text-transform:uppercase;
letter-spacing:.1em;color:var(--muted)}.value{font-size:21px;font-weight:750;margin-top:4px}
.pill{display:inline-block;border-radius:99px;padding:3px 9px;background:#27334e;color:var(--muted)}
.current{color:var(--ok)}.stale,.missing,.starting,.reconnecting{color:var(--warn)}.failed{color:var(--bad)}
table{width:100%;border-collapse:collapse;margin-top:9px}td,th{text-align:left;padding:7px;border-bottom:1px solid #26314b}
pre{white-space:pre-wrap;color:var(--muted);max-height:180px;overflow:auto}#message{min-height:24px;color:var(--muted)}
.source{margin-top:18px;color:var(--muted)}.source a{color:var(--accent)}
@media(max-width:820px){.grid{grid-template-columns:1fr 1fr}.wide{grid-column:span 2}}
</style></head><body><main><h1>XR02 · Live Global Tracking</h1>
<p class="sub">Four-camera office demonstrator · anonymous scene-global IDs · native Rerun 3D</p>
<div class="bar"><button class="primary" data-action="start">Start live service</button>
<button data-action="open-rerun">Open Rerun 3D</button><button data-action="reset-trails">Reset trails</button>
<button data-action="export">Evidence snapshot</button><button data-action="stop">Stop &amp; finalize</button></div>
<div id="message"></div><section class="grid"><article class="card"><div class="label">Service</div>
<div class="value" id="service">—</div></article><article class="card"><div class="label">Global tracks</div>
<div class="value" id="tracks">0</div></article><article class="card"><div class="label">Local tracking ticks</div>
<div class="value" id="ticks">0</div></article><article class="card"><div class="label">Explicit inference drops</div>
<div class="value" id="drops">0</div><small id="pending-detail"></small></article><article class="card"><div class="label">Cadence</div>
<div class="value" id="cadence">—</div><small id="cadence-detail"></small></article><article class="card"><div class="label">Publication</div>
<div class="value" id="publication">—</div><small id="publication-detail"></small></article><article class="card"><div class="label">Media ingress</div>
<div class="value" id="ingress">—</div><small id="ingress-detail"></small></article><article class="card"><div class="label">Trial replay capture</div>
<div class="value" id="recording">—</div><small id="recording-detail"></small></article><article class="card wide"><div class="label">Camera health</div>
<table><thead><tr><th>Camera</th><th>State</th><th>Backend</th><th>Generation / epoch</th><th>Frames</th><th>Frame / heartbeat age</th></tr></thead>
<tbody id="cameras"></tbody></table></article><article class="card wide"><div class="label">Current scene</div>
<pre id="scene">—</pre></article></section>
<p class="source">Source: <a href="https://github.com/hiverlabzhengjie/3d-spatial-mapping-rd-phase2">AGPL-3.0 repository</a></p>
</main><script>
const $=id=>document.getElementById(id);let busy=false;
async function status(){try{const r=await fetch('/api/status',{cache:'no-store'});render(await r.json())}
catch(e){$('message').textContent='Console connection unavailable'}}
function render(s){const v=s.service||{};$('service').textContent=v.state||'stopped';
$('service').className='value '+(v.state==='running'?'current':'');$('tracks').textContent=(v.global_tracks||[]).length;
const w=v.worker||{};$('ticks').textContent=w.completed_ticks||0;$('drops').textContent=w.busy_dropped_ticks||0;
$('pending-detail').textContent=`pending: ${w.pending?'yes':'no'} · consumed ${w.pending_consumed_ticks||0} · replaced ${w.pending_replaced_ticks||0} · stale ${w.pending_stale_dropped_ticks||0} · epoch-invalidated ${w.pending_invalidated_ticks||0}`;
const c=v.cadence||{};$('cadence').textContent=c.local_tracking_hz?`${c.local_tracking_hz} Hz local`:'—';
$('cadence-detail').textContent=c.global_association_hz?`${c.global_association_hz} Hz global · ${c.appearance_available_hz||c.local_tracking_hz} Hz fresh appearance · ${c.appearance_persistence_hz||c.effective_appearance_hz} Hz durable gallery · ${c.publication_hz} Hz view`:'';
const p=v.publication_worker||{};$('publication').textContent=`${p.completed_items||0} views`;
$('publication-detail').textContent=`${p.busy_dropped_items||0} intermediate view drops · tracking unaffected`;
const g=s.media_ingress||{};$('ingress').textContent=g.state||'—';$('ingress').className='value '+(g.state==='running'?'current':'');
$('ingress-detail').textContent=g.version?`${g.version} · generation ${g.generation||0} · ${g.restarts||0} restarts`:'explicit diagnostic profile';
const r=s.trial_recording||{};$('recording').textContent=r.active?'recording':(r.configured?'ready':'off');
$('recording').className='value '+(r.active?'current':'');$('recording-detail').textContent=r.configured?`${r.segments_started||0} segments · ${r.segments_currently_writing||0} writing · automatic reconnect`: 'enable at console launch';
$('scene').textContent=(v.scene_context_sha256||'—')+(v.scene_update_available?'\\nNew scene available — restart required':'');
$('cameras').innerHTML=(v.camera_health||[]).map(c=>`<tr><td>${c.camera_id}</td><td class="${c.state}">${c.state}<br><small>${c.capture_process_state||'—'}</small></td>
<td>${c.capture_backend||'—'}</td><td>${c.generation} / ${c.tracker_epoch||0}</td><td>${c.delivered_frames||0} delivered<br><small>${c.decoded_frames} decoded · ${c.supervisor_restarts||0} watchdog</small></td><td>${c.frame_age_ms==null?'—':c.frame_age_ms.toFixed(0)+' ms'} / ${c.process_heartbeat_age_ms==null?'—':c.process_heartbeat_age_ms.toFixed(0)+' ms'}<br><small>${c.restart_reason||''}</small></td></tr>`).join('')}
document.querySelectorAll('button').forEach(b=>b.onclick=async()=>{if(busy)return;busy=true;
$('message').textContent='Working…';try{const r=await fetch('/api/'+b.dataset.action,{method:'POST'});const v=await r.json();
if(!r.ok)throw Error(v.detail||v.error);$('message').textContent='Done';render(v.service?v:await (await fetch('/api/status')).json())}
catch(e){$('message').textContent=e.message}finally{busy=false}});status();setInterval(status,1000);
</script></body></html>"""
