# ruff: noqa: E501
"""Small local operator console for the XR02 live/recording lifecycle."""

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

    def start_live(
        self,
        *,
        resumed_from_session_id: str | None = None,
        scene_update_id: str | None = None,
    ) -> dict[str, object]: ...

    def start_recording(self) -> dict[str, object]: ...

    def stop(self, *, reason: str = "operator") -> dict[str, object]: ...

    def open_rerun(self) -> dict[str, object]: ...

    def reset_trails(self) -> dict[str, object]: ...

    def export_evidence_snapshot(self) -> dict[str, object]: ...

    def view_recording(self, session_id: str) -> dict[str, object]: ...

    def save_recording(self, session_id: str, label: str) -> dict[str, object]: ...

    def delete_recording(self, session_id: str, confirmation: str) -> dict[str, object]: ...

    def status(self) -> dict[str, object]: ...


class XR02OperatorServer:
    """Loopback-only HTTP shell; it never receives or returns RTSP endpoints."""

    def __init__(
        self,
        controller: OperatorController,
        port: int = 8094,
        *,
        api_token: str | None = None,
        serve_page: bool = True,
    ) -> None:
        if port != 0 and not 1024 <= port <= 65535:
            raise ValueError("operator port must be within 1024..65535")
        if not serve_page and not api_token:
            raise ValueError("API-only XR02 worker requires an authentication token")
        self._server = ThreadingHTTPServer(
            ("127.0.0.1", port),
            _handler_factory(controller, api_token=api_token, serve_page=serve_page),
        )
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


def _handler_factory(
    controller: OperatorController,
    *,
    api_token: str | None = None,
    serve_page: bool = True,
) -> type[BaseHTTPRequestHandler]:
    routes: dict[str, Callable[[dict[str, object]], dict[str, object]]] = {
        "/api/start": lambda _body: controller.start(),
        "/api/start-live": lambda body: _start_live(controller, body),
        "/api/start-recording": lambda _body: controller.start_recording(),
        "/api/stop": lambda body: _stop(controller, body),
        "/api/open-rerun": lambda _body: controller.open_rerun(),
        "/api/reset-trails": lambda _body: controller.reset_trails(),
        "/api/export": lambda _body: controller.export_evidence_snapshot(),
        "/api/view-recording": lambda body: controller.view_recording(
            _required_string(body, "session_id")
        ),
        "/api/save-recording": lambda body: controller.save_recording(
            _required_string(body, "session_id"), _required_string(body, "label")
        ),
        "/api/delete-recording": lambda body: controller.delete_recording(
            _required_string(body, "session_id"), _required_string(body, "confirmation")
        ),
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "XR02Operator/2"

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                if serve_page:
                    self._send(HTTPStatus.OK, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "api_only"})
                return
            if self.path == "/api/status":
                if not self._authorized():
                    return
                self._json(HTTPStatus.OK, controller.status())
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                return
            action = routes.get(self.path)
            if action is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                result = action(self._json_body())
            except Exception as error:
                self._json(
                    HTTPStatus.CONFLICT,
                    {"error": type(error).__name__, "detail": str(error)},
                )
                return
            self._json(HTTPStatus.OK, result)

        def _authorized(self) -> bool:
            if api_token is None or self.headers.get("X-XR02-Worker-Token") == api_token:
                return True
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return False

        def _json_body(self) -> dict[str, object]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ValueError("invalid Content-Length") from error
            if length < 0 or length > 4096:
                raise ValueError("request body exceeds the 4096-byte operator limit")
            if length == 0:
                return {}
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("operator request body must be a JSON object")
            return value

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


def _required_string(body: dict[str, object], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value


def _optional_string(body: dict[str, object], key: str) -> str | None:
    value = body.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be null or a non-blank string")
    return value.strip()


def _start_live(controller: OperatorController, body: dict[str, object]) -> dict[str, object]:
    resumed = _optional_string(body, "resumed_from_session_id")
    update_id = _optional_string(body, "scene_update_id")
    if resumed is None and update_id is None:
        return controller.start_live()
    return controller.start_live(
        resumed_from_session_id=resumed,
        scene_update_id=update_id,
    )


def _stop(controller: OperatorController, body: dict[str, object]) -> dict[str, object]:
    reason = _optional_string(body, "reason")
    if reason is None:
        return controller.stop()
    return controller.stop(reason=reason)


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>XR02 Live Service</title><style>
:root{color-scheme:dark;--bg:#07111d;--card:#101f2d;--line:#263b4d;--ink:#eef7fb;--muted:#98acb9;
--ok:#1bd7b0;--warn:#ffc766;--bad:#ff7180;--accent:#39c6e5}*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at top,#102b3c,var(--bg) 48%);font:15px/1.5 system-ui;color:var(--ink)}
main{max-width:1080px;margin:auto;padding:38px 24px 70px}h1{font-size:34px;margin:0}.sub{color:var(--muted);margin:4px 0 25px}
.panel{background:#101f2dee;border:1px solid var(--line);border-radius:16px;padding:22px;margin:14px 0;box-shadow:0 16px 45px #0004}
.state{display:flex;justify-content:space-between;gap:16px;align-items:center}.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}
.title{font-size:23px;font-weight:760}.muted,small{color:var(--muted)}.actions{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:18px}
button{border:1px solid var(--line);border-radius:12px;padding:12px 16px;background:#173146;color:var(--ink);font-weight:700;cursor:pointer}
button.hero{padding:22px;text-align:left;font-size:18px;background:#15475a}button.record{background:#26385d}button.danger{background:#522a35;border-color:#81404d}
button:disabled{opacity:.42;cursor:not-allowed}button:hover:not(:disabled){filter:brightness(1.14)}.hidden{display:none!important}.full{width:100%}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}.metric{background:#0b1723;border:1px solid #203548;border-radius:11px;padding:12px}
.metric b{font-size:21px;display:block}.pill{border-radius:999px;padding:5px 11px;background:#193449;color:var(--muted)}.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.pending-actions{display:flex;gap:9px;flex-wrap:wrap;margin-top:16px}input{flex:1;min-width:220px;border:1px solid var(--line);border-radius:10px;background:#081622;color:var(--ink);padding:11px}
.list{display:grid;gap:9px;margin-top:12px}.row{display:flex;align-items:center;justify-content:space-between;gap:12px;background:#0b1723;border:1px solid #203548;border-radius:11px;padding:12px}
details summary{cursor:pointer;font-weight:700}table{width:100%;border-collapse:collapse;margin-top:10px}td,th{text-align:left;padding:7px;border-bottom:1px solid #263b4d}
#message{min-height:22px;color:var(--muted)}pre{white-space:pre-wrap;color:var(--muted);max-height:160px;overflow:auto}
@media(max-width:720px){.actions,.grid{grid-template-columns:1fr}.state,.row{align-items:flex-start;flex-direction:column}}
</style></head><body><main><div class="eyebrow">XR02 operator console</div><h1>Live people tracking</h1>
<p class="sub">Two operating modes · anonymous global tracks · facility XY · native Rerun 3D</p><div id="message"></div>
<section class="panel state"><div><div class="eyebrow">Service state</div><div class="title" id="state">Loading…</div><div class="muted" id="state-detail"></div></div><span class="pill" id="mode">Ready</span></section>
<section class="panel" id="start-panel"><div class="title">What would you like to do?</div><div class="actions">
<button class="hero" data-action="start-live">Start Live Service<br><small>Live view + compact 1 Hz count/track/XY history. No video or replay archive.</small></button>
<button class="hero record" id="start-recording" data-action="start-recording">Start Replayable Recording<br><small>Full Rerun, decision journals and camera footage. Save or delete after stopping.</small></button>
</div></section>
<section class="panel hidden" id="active-panel"><div class="title" id="active-title">Live is running</div><p class="muted" id="viewer-note"></p>
<div class="grid"><div class="metric"><span class="muted">People observed now</span><b id="people">0</b></div><div class="metric"><span class="muted">Tracking ticks</span><b id="ticks">0</b></div><div class="metric"><span class="muted">Camera health</span><b id="health">—</b></div></div>
<div class="pending-actions"><button class="full danger" data-action="stop" id="stop-button">Stop Live Service</button></div></section>
<section class="panel hidden" id="pending-panel"><div class="eyebrow">Recording finalized</div><div class="title">Choose what happens to this recording</div>
<p class="muted" id="pending-detail"></p><div class="pending-actions"><button data-action="view-pending">View recording</button>
<input id="recording-label" maxlength="80" placeholder="Recording name"><button data-action="save-pending">Name &amp; save</button>
<button class="danger" data-action="delete-pending">Permanently delete</button></div></section>
<section class="panel"><div class="title">Saved recordings</div><div class="list" id="saved"><div class="muted">No saved recordings yet.</div></div></section>
<section class="panel"><div class="title">Recent Live history</div><p class="muted">Small, durable count/track/XY telemetry only.</p><div class="list" id="live-history"><div class="muted">No completed Live runs yet.</div></div></section>
<details class="panel"><summary>Diagnostics &amp; engineering controls</summary><div class="pending-actions"><button data-action="open-rerun">Open Rerun 3D</button><button data-action="reset-trails">Reset trails</button><button data-action="export">Evidence snapshot</button></div>
<div class="grid"><div class="metric"><span class="muted">Inference drops</span><b id="drops">0</b></div><div class="metric"><span class="muted">Ingress</span><b id="ingress">—</b></div><div class="metric"><span class="muted">Storage free</span><b id="storage">—</b></div></div>
<table><thead><tr><th>Camera</th><th>State</th><th>Frames</th><th>Age</th></tr></thead><tbody id="cameras"></tbody></table><pre id="scene">—</pre></details>
</main><script>
const $=id=>document.getElementById(id);let busy=false,autoOpening=false,current={};
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const bytes=n=>n==null?'—':n<1048576?`${(n/1024).toFixed(1)} KB`:n<1073741824?`${(n/1048576).toFixed(1)} MB`:`${(n/1073741824).toFixed(1)} GB`;
async function api(action,body={}){if(busy)return;busy=true;$('message').textContent='Working…';try{const r=await fetch('/api/'+action,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const v=await r.json();if(!r.ok)throw Error(v.detail||v.error);$('message').textContent='';render(v.service?v:await getStatus())}catch(e){$('message').textContent=e.message}finally{busy=false}}
async function getStatus(){const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error('Console status unavailable');return r.json()}
async function poll(){try{render(await getStatus())}catch(e){$('message').textContent='Console connection unavailable'}}
function render(s){current=s;const active=!!s.active,p=s.pending_run||null,mode=s.active_mode;$('start-panel').classList.toggle('hidden',active||!!p);$('active-panel').classList.toggle('hidden',!active);$('pending-panel').classList.toggle('hidden',!p);
$('state').textContent=active?(mode==='recording'?'Replayable recording in progress':'Live service running'):p?(p.state==='recovery_required'?'Run recovery required':'Recording ready for review'):'Ready';
const storageGuard=(s.storage||{}).guard_message;$('state-detail').textContent=storageGuard||(active?(mode==='recording'?'Full evidence is being staged.':'Compact telemetry is being retained; replay evidence is not.'):p?(p.error_detail||'View, save, or delete this exact staged run before starting again.'):'Choose one operating mode below.');
$('mode').textContent=active?(mode==='recording'?'Recording':'Live'):p?'Decision required':'Ready';$('active-title').textContent=mode==='recording'?'Replayable recording is running':'Live service is running';$('stop-button').textContent=mode==='recording'?'Stop Recording':'Stop Live Service';
const v=s.service||{},w=v.worker||{},cams=v.camera_health||[],currentCount=cams.filter(c=>c.state==='current').length;$('people').textContent=(v.global_tracks||[]).filter(t=>t.state!=='ended').length;$('ticks').textContent=w.completed_ticks||0;$('health').textContent=`${currentCount}/${cams.length||4} current`;
const auto=s.viewer_auto_open||{};$('viewer-note').textContent=auto.opened?'Rerun 3D is open.':auto.all_cameras_current?'Opening Rerun now…':`Rerun opens when all cameras are current, or degraded after ${Math.max(0,30-(auto.startup_elapsed_seconds||0)).toFixed(0)} s.`;
if(active&&auto.eligible&&!auto.opened&&!autoOpening){autoOpening=true;api('open-rerun').finally(()=>autoOpening=false)}
const rb=$('start-recording');rb.disabled=!s.recording_available;rb.title=s.recording_available?'':'Restart the console with its recording profile to enable full capture.';
if(p){$('pending-detail').textContent=`${p.session_id} · ${bytes(p.byte_count)} · ${p.started_at_utc}`;const recovery=p.state==='recovery_required';$('recording-label').disabled=recovery;document.querySelector('[data-action="save-pending"]').disabled=recovery;document.querySelector('[data-action="view-pending"]').disabled=recovery||p.mode!=='recording'}
$('saved').innerHTML=(s.saved_recordings||[]).length?(s.saved_recordings||[]).map(r=>`<div class="row"><div><b>${esc(r.label)}</b><br><small>${esc(r.started_at_utc)} · ${bytes(r.byte_count)}</small></div><button data-view="${esc(r.session_id)}">View</button></div>`).join(''):'<div class="muted">No saved recordings yet.</div>';
$('live-history').innerHTML=(s.recent_live_runs||[]).length?(s.recent_live_runs||[]).map(r=>`<div class="row"><div><b>${esc(r.started_at_utc)}</b><br><small>${bytes(r.byte_count)} compact telemetry</small></div><span class="pill">Completed</span></div>`).join(''):'<div class="muted">No completed Live runs yet.</div>';
$('drops').textContent=w.busy_dropped_ticks||0;const g=s.media_ingress||{};$('ingress').textContent=g.state||'—';$('storage').textContent=bytes((s.storage||{}).free_bytes);$('scene').textContent=(v.scene_context_sha256||'—')+(v.scene_update_available?'\\nNew scene available — restart required':'');
$('cameras').innerHTML=cams.map(c=>`<tr><td>${esc(c.camera_id)}</td><td class="${esc(c.state)}">${esc(c.state)}</td><td>${c.delivered_frames||0}</td><td>${c.frame_age_ms==null?'—':c.frame_age_ms.toFixed(0)+' ms'}</td></tr>`).join('')}
document.addEventListener('click',e=>{const b=e.target.closest('button');if(!b||b.disabled)return;if(b.dataset.view){api('view-recording',{session_id:b.dataset.view});return}const a=b.dataset.action;if(!a)return;const p=current.pending_run||{};if(a==='view-pending')api('view-recording',{session_id:p.session_id});else if(a==='save-pending')api('save-recording',{session_id:p.session_id,label:$('recording-label').value});else if(a==='delete-pending'){const phrase=`DELETE ${p.session_id}`;if(confirm(`Permanently delete this exact run?\\n\\n${p.session_id}\\n\\nThis cannot be recovered.`))api('delete-recording',{session_id:p.session_id,confirmation:phrase})}else api(a)});
poll();setInterval(poll,1000);
</script></body></html>"""
