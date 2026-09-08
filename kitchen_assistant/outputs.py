"""Console and lightweight local web outputs."""

import json
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread
from typing import Protocol

from .state import Frame, StrategyResult
from .workflow import WorkflowView


class Output(Protocol):
    def start(self) -> None: ...

    def publish(self, frame: Frame | None, view: WorkflowView, result: StrategyResult | None = None) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class ConsoleOutput:
    debug: bool = False
    _last: tuple | None = field(default=None, init=False)

    def start(self) -> None:
        return None

    def publish(self, frame: Frame | None, view: WorkflowView, result: StrategyResult | None = None) -> None:
        signature = (
            view.step_id,
            view.phase,
            view.missing_objects,
            view.missing_ingredients,
            view.assessment_status,
        )
        if signature == self._last and not self.debug:
            return
        self._last = signature
        print(f"\n[{view.phase}] {view.step_name or 'Recipe complete'}")
        if view.missing_objects:
            print("Objects needed: " + ", ".join(view.missing_objects))
        if view.missing_ingredients:
            print("Ingredients needed: " + ", ".join(view.missing_ingredients))
        if view.description:
            print("Instruction: " + view.description)
        if view.next_step_name:
            next_items = (*view.next_required_objects, *view.next_required_ingredients)
            print("Next: " + view.next_step_name + (" — " + ", ".join(next_items) if next_items else ""))
        if view.assessment_status:
            print(f"Assessment: {view.assessment_status} ({view.assessment_confidence:.2f})")
        if self.debug and result:
            print("Detected: " + ", ".join(sorted(result.present_entities)))

    def close(self) -> None:
        return None


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kitchen assistant</title><style>
body{margin:0;background:#111815;color:#eff7f1;font:16px/1.5 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:24px}
.grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(280px,1fr);gap:22px}img{width:100%;border-radius:12px;background:#202824}
.card{background:#1b2420;border:1px solid #34443b;border-radius:12px;padding:20px}h1,h2{margin-top:0}.muted{color:#a9bbb0}
ul{padding-left:20px}@media(max-width:760px){.grid{grid-template-columns:1fr}}
</style></head><body><main><h1 id="title">Kitchen assistant</h1><div class="grid"><img id="feed" src="/frame.jpg">
<section class="card"><div class="muted" id="phase"></div><h2 id="step"></h2><p id="description"></p>
<h3>Needed now</h3><ul id="needed"></ul><h3>Next</h3><p id="next"></p><div class="muted" id="assessment"></div></section></div></main>
<script>function text(id,value){document.getElementById(id).textContent=value||''}
async function update(){try{const r=await fetch('/state',{cache:'no-store'});const s=await r.json();text('title',s.recipe_title);
text('phase',s.phase);text('step',s.step_name||'Recipe complete');text('description',s.description);
const needed=[...s.missing_objects,...s.missing_ingredients];document.getElementById('needed').innerHTML=needed.map(x=>'<li>'+x+'</li>').join('')||'<li>Everything is present</li>';
const next=[...s.next_required_objects,...s.next_required_ingredients];text('next',s.next_step_name?(s.next_step_name+(next.length?' — '+next.join(', '):'')):'—');
text('assessment',s.assessment_status?('Assessment: '+s.assessment_status+' ('+s.assessment_confidence.toFixed(2)+')'):'');
document.getElementById('feed').src='/frame.jpg?t='+Date.now()}catch(e){text('phase','Waiting for application state')}}setInterval(update,700);update();</script>
</body></html>"""


@dataclass(slots=True)
class WebOutput:
    host: str = "127.0.0.1"
    port: int = 8080
    _lock: Lock = field(default_factory=Lock, init=False)
    _state: bytes = field(default=b"{}", init=False)
    _frame: bytes | None = field(default=None, init=False)
    _server: ThreadingHTTPServer | None = field(default=None, init=False)
    _thread: Thread | None = field(default=None, init=False)

    def start(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith("/state"):
                    with owner._lock:
                        body, content_type = owner._state, "application/json"
                elif self.path.startswith("/frame.jpg"):
                    with owner._lock:
                        body = owner._frame
                    if body is None:
                        self.send_error(404, "No frame available")
                        return
                    content_type = "image/jpeg"
                elif self.path == "/" or self.path.startswith("/?"):
                    body, content_type = _PAGE.encode(), "text/html; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return None

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = Thread(target=self._server.serve_forever, name="web-output", daemon=True)
        self._thread.start()
        print(f"Web UI: http://{self.host}:{self._server.server_port}")

    def publish(self, frame: Frame | None, view: WorkflowView, result: StrategyResult | None = None) -> None:
        state = view.to_dict()
        if result:
            state["detections"] = [asdict(detection) for detection in result.detections]
            state["assessment"] = asdict(result.assessment)
        body = json.dumps(state, default=lambda value: value.isoformat()).encode()
        with self._lock:
            self._state = body
            if frame is not None:
                self._frame = frame.image

    def close(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=2)


@dataclass(slots=True)
class CompositeOutput:
    outputs: tuple[Output, ...]

    def start(self) -> None:
        for output in self.outputs:
            output.start()

    def publish(self, frame: Frame | None, view: WorkflowView, result: StrategyResult | None = None) -> None:
        for output in self.outputs:
            output.publish(frame, view, result)

    def close(self) -> None:
        for output in reversed(self.outputs):
            output.close()
