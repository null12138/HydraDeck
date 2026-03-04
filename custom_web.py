from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from app import _api_quick_check, _run_agentic_pipeline
from hydradeck.clients.grok_client import GrokClient


class RunRequest(BaseModel):
    topic: str
    model: str = "grok-3-mini"
    base_url: str = "https://api.example.com"
    api_key: str = ""
    request_budget: float = 30.0
    use_mock: bool = False
    language: str = "en"
    model_scope: str = ""
    model_structure: str = ""
    model_planner: str = ""
    model_section: str = ""
    model_paper: str = ""
    model_slides: str = ""


JOBS: dict[str, dict[str, Any]] = {}
LOCK = threading.Lock()
STATE_PATH = Path("/tmp/hydradeck_state.json")
HISTORY_LIMIT = 40

app = FastAPI(title="HydraDeck")


def _load_state() -> None:
    if not STATE_PATH.exists():
        return
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    jobs = data.get("jobs")
    if isinstance(jobs, dict):
        with LOCK:
            JOBS.update({str(k): v for k, v in jobs.items() if isinstance(v, dict)})


def _save_state() -> None:
    with LOCK:
        payload = {"jobs": JOBS}
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _prune_history() -> None:
    with LOCK:
        items = sorted(
            JOBS.items(),
            key=lambda kv: float(kv[1].get("updated_at", 0.0)),
            reverse=True,
        )
        keep = dict(items[:HISTORY_LIMIT])
        JOBS.clear()
        JOBS.update(keep)


_load_state()


def _new_job(req: RunRequest) -> dict[str, Any]:
    now = time.time()
    return {
        "id": str(uuid.uuid4()),
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "progress": 0,
        "status_text": "Queued",
        "progress_log": "",
        "scope": "",
        "sections": "",
        "paper": "",
        "slides": "",
        "pdf_paths": "",
        "paper_pdf": "",
        "slides_pdf": "",
        "error": "",
        "events": [],
        "params": req.model_dump(),
    }


def _update_job(job_id: str, updates: dict[str, Any]) -> None:
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = time.time()
    _prune_history()
    _save_state()


def _append_event(job_id: str, event: dict[str, Any]) -> None:
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        events = job.get("events")
        if isinstance(events, list):
            events.append(event)
    _save_state()


def _run_job(job_id: str, req: RunRequest) -> None:
    _update_job(job_id, {"status": "running", "status_text": "Running"})

    def on_stage(payload: dict[str, Any]) -> None:
        _update_job(
            job_id,
            {
                "status": "running",
                "status_text": str(payload.get("status", "Running")),
                "progress": int(str(payload.get("progress", "0"))),
                "progress_log": str(payload.get("progress_log", "")),
                "scope": str(payload.get("scope", "")),
                "sections": str(payload.get("sections", "")),
                "paper": str(payload.get("paper", "")),
                "slides": str(payload.get("slides", "")),
                "pdf_paths": str(payload.get("pdf_paths", "")),
                "paper_pdf": str(payload.get("paper_pdf", "")),
                "slides_pdf": str(payload.get("slides_pdf", "")),
            },
        )
        _append_event(
            job_id,
            {
                "ts": time.time(),
                "stage": str(payload.get("stage", "")),
                "detail": str(payload.get("detail", "")),
                "progress": int(str(payload.get("progress", "0"))),
            },
        )

    try:
        (
            status,
            progress_log,
            scope,
            sections,
            paper,
            slides,
            pdf_paths,
            paper_pdf,
            slides_pdf,
        ) = _run_agentic_pipeline(
            topic=req.topic,
            model=req.model,
            base_url=req.base_url,
            api_key=req.api_key,
            request_budget=req.request_budget,
            use_mock=req.use_mock,
            progress=None,
            stage_callback=on_stage,
            language=req.language,
            stage_models={
                "scope": req.model_scope,
                "structure": req.model_structure,
                "planner": req.model_planner,
                "section": req.model_section,
                "paper": req.model_paper,
                "slides": req.model_slides,
            },
        )
        _update_job(
            job_id,
            {
                "status": "done",
                "status_text": status,
                "progress": 100,
                "progress_log": progress_log,
                "scope": scope,
                "sections": sections,
                "paper": paper,
                "slides": slides,
                "pdf_paths": pdf_paths,
                "paper_pdf": paper_pdf,
                "slides_pdf": slides_pdf,
            },
        )
    except Exception as exc:
        _update_job(
            job_id,
            {
                "status": "error",
                "status_text": "Failed",
                "error": str(exc),
            },
        )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>HydraDeck</title>
  <style>
    :root{--bg:#f5ecd8;--paper:#fff9ec;--ink:#2a1f12;--muted:#7a5f3e;--accent:#8b3a3a;--ok:#2f6f3e}
    body{font-family:"IBM Plex Mono","Courier New",monospace;max-width:1220px;margin:18px auto;padding:0 12px;background:var(--bg);color:var(--ink)}
    .panel{border:2px solid var(--ink);background:var(--paper);box-shadow:2px 2px 0 #0002;padding:10px;margin:10px 0}
    .row{display:flex;gap:10px;margin:8px 0;flex-wrap:wrap}
    input,select,textarea{padding:8px;width:100%;border:1px solid #4b3924;background:#fffdf7;color:var(--ink)}
    button{padding:9px 13px;border:2px solid var(--ink);background:#ead2b0;color:var(--ink);cursor:pointer}
    button:hover{background:#f0ddc3}
    .bar{height:16px;background:#d8c3a5;border:1px solid #4b3924;overflow:hidden}
    .fill{height:100%;width:0%;background:linear-gradient(90deg,#8b3a3a,#d46a6a);transition:width .25s}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    pre{background:#1b130c;color:#f7e8d0;padding:10px;white-space:pre-wrap;max-height:260px;overflow:auto;border:1px solid #3a2a1b}
    .title{font-size:28px;font-weight:700;letter-spacing:1px}
    .sub{color:var(--muted)}
    .tiny{font-size:12px;color:var(--muted)}
    details{border:1px dashed #7a5f3e;padding:8px;background:#fff9ef}
    summary{cursor:pointer;font-weight:700}
  </style>
</head>
<body>
  <div class=\"panel\"><div class=\"title\">HydraDeck</div></div>
  <div class=\"panel\">
    <div class=\"row\" style=\"gap:6px\">
      <button onclick=\"showTab('tab-run')\">Run</button>
      <button onclick=\"showTab('tab-artifacts')\">Artifacts</button>
      <button onclick=\"showTab('tab-console')\">Console</button>
    </div>
  </div>

  <div id=\"tab-run\" class=\"panel tab\">
    <div class=\"row\"><input id=\"topic\" value=\"RynnBrain technical research report\" /></div>
    <div class=\"row\">
      <select id=\"model\"></select>
      <input id=\"base_url\" value=\"https://api.example.com\" />
    </div>
    <div class=\"row\">
      <label>language
        <select id=\"language\">
          <option value=\"en\" selected>English</option>
          <option value=\"zh\">中文</option>
        </select>
      </label>
      <input id=\"api_key\" placeholder=\"api key\" />
      <input id=\"request_budget\" value=\"30\" />
      <label><input id=\"use_mock\" type=\"checkbox\" /> use mock</label>
    </div>
    <div class=\"row\">
      <button onclick=\"quickCheck()\">Quick API Check</button>
      <button onclick=\"startRun()\">Run HydraDeck</button>
      <button onclick=\"resumeLastRun()\">Resume Last Run</button>
    </div>

    <details>
      <summary>Advanced model routing</summary>
      <div class=\"tiny\">Per-agent model overrides (optional)</div>
      <div class=\"row\"><select id=\"model_scope\"></select><select id=\"model_structure\"></select></div>
      <div class=\"row\"><select id=\"model_planner\"></select><select id=\"model_section\"></select></div>
      <div class=\"row\"><select id=\"model_paper\"></select><select id=\"model_slides\"></select></div>
    </details>
  </div>
  <div id=\"status\">Idle</div>
  <div class=\"bar\"><div id=\"fill\" class=\"fill\"></div></div>
  <div id=\"pct\">0%</div>
  <div id=\"tab-artifacts\" class=\"panel tab\" style=\"display:none\">
    <div class=\"row\">
    <a id=\"paperLink\" target=\"_blank\"></a>
    <a id=\"slidesLink\" target=\"_blank\"></a>
    </div>
    <div class=\"grid\">
    <div><h4>Scope</h4><pre id=\"scope\"></pre></div>
    <div><h4>Sections</h4><pre id=\"sections\"></pre></div>
    <div><h4>paper.tex</h4><pre id=\"paper\"></pre></div>
    <div><h4>slides.tex</h4><pre id=\"slides\"></pre></div>
    </div>
  </div>

  <div id=\"tab-console\" class=\"panel tab\" style=\"display:none\">
    <div class=\"grid\">
      <div><h4>Progress</h4><pre id=\"progress\"></pre></div>
      <div><h4>Events</h4><pre id=\"events\"></pre></div>
    </div>
  </div>

<script>
let jobId = null;
let timer = null;
let inflight = false;
let refreshFailCount = 0;

function showTab(id){
  for(const el of document.querySelectorAll('.tab')) el.style.display='none';
  document.getElementById(id).style.display='block';
}

function addModelOptions(selectId, models){
  const s=document.getElementById(selectId);
  s.innerHTML='';
  const blank=document.createElement('option');
  blank.value='';
  blank.textContent = selectId==='model' ? '(default model)' : '(inherit default)';
  s.appendChild(blank);
  for(const m of models){
    const o=document.createElement('option');
    o.value=m; o.textContent=m; s.appendChild(o);
  }
}

async function loadModels(){
  try{
    const ctl = new AbortController();
    const t = setTimeout(()=>ctl.abort(), 15000);
    const r=await fetch('/api/models?base_url='+encodeURIComponent(document.getElementById('base_url').value)+'&api_key='+encodeURIComponent(document.getElementById('api_key').value), {signal: ctl.signal});
    clearTimeout(t);
    const j=await r.json();
    const models=Array.isArray(j.models)?j.models:[];
    for(const id of ['model','model_scope','model_structure','model_planner','model_section','model_paper','model_slides']) addModelOptions(id, models);
    if(models.includes('grok-3-mini')) document.getElementById('model').value='grok-3-mini';
  }catch(e){
    document.getElementById('status').innerText='model list failed: '+e;
  }
}

function payload(){
  return {
    topic: document.getElementById('topic').value,
    model: document.getElementById('model').value,
    base_url: document.getElementById('base_url').value,
    api_key: document.getElementById('api_key').value,
    request_budget: Number(document.getElementById('request_budget').value || 30),
    use_mock: document.getElementById('use_mock').checked,
    language: document.getElementById('language').value,
    model_scope: document.getElementById('model_scope').value,
    model_structure: document.getElementById('model_structure').value,
    model_planner: document.getElementById('model_planner').value,
    model_section: document.getElementById('model_section').value,
    model_paper: document.getElementById('model_paper').value,
    model_slides: document.getElementById('model_slides').value,
  };
}

async function quickCheck(){
  const ctl = new AbortController();
  const t = setTimeout(()=>ctl.abort(), 20000);
  const r = await fetch('/api/quick-check',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload()),signal: ctl.signal});
  clearTimeout(t);
  const j = await r.json();
  document.getElementById('status').innerText = j.result || j.error;
  showTab('tab-console');
}

async function startRun(){
  if(inflight) return;
  inflight = true;
  const ctl = new AbortController();
  const t = setTimeout(()=>ctl.abort(), 20000);
  const r = await fetch('/api/jobs',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload()),signal: ctl.signal});
  clearTimeout(t);
  const j = await r.json();
  jobId = j.id;
  localStorage.setItem('hydradeck_last_job_id', jobId);
  if (timer) clearInterval(timer);
  timer = setInterval(refresh, 1000);
  refresh();
  showTab('tab-console');
}

async function refresh(){
  if(!inflight) return;
  if(!jobId) return;
  try {
    const ctl = new AbortController();
    const t = setTimeout(()=>ctl.abort(), 12000);
    const r = await fetch('/api/jobs/'+jobId, {signal: ctl.signal});
    clearTimeout(t);
    if(!r.ok) {
      refreshFailCount += 1;
      if (refreshFailCount >= 5) {
        inflight = false;
        if (timer) { clearInterval(timer); timer = null; }
        document.getElementById('status').innerText = 'Polling paused (network/server issue). Use Resume Last Run.';
      }
      return;
    }
    const j = await r.json();
    refreshFailCount = 0;
  document.getElementById('status').innerText = j.status_text || j.status;
  const p = Math.max(0, Math.min(100, Number(j.progress || 0)));
  document.getElementById('fill').style.width = p + '%';
  document.getElementById('pct').innerText = p + '%';
  document.getElementById('progress').innerText = j.progress_log || '';
  document.getElementById('scope').innerText = j.scope || '';
  document.getElementById('sections').innerText = j.sections || '';
  document.getElementById('paper').innerText = j.paper || '';
  document.getElementById('slides').innerText = j.slides || '';
  document.getElementById('events').innerText = JSON.stringify(j.events || [], null, 2);

  const p1 = document.getElementById('paperLink');
  const p2 = document.getElementById('slidesLink');
  if (j.paper_pdf){ p1.href = '/api/jobs/'+jobId+'/artifact/paper'; p1.innerText='Download paper.pdf'; }
  if (j.slides_pdf){ p2.href = '/api/jobs/'+jobId+'/artifact/slides'; p2.innerText='Download slides.pdf'; }

  if (j.status === 'done' || j.status === 'error') {
    clearInterval(timer);
    timer = null;
    inflight = false;
    localStorage.removeItem('hydradeck_last_job_id');
  }
  } catch (e) {
    refreshFailCount += 1;
    if (refreshFailCount >= 5) {
      inflight = false;
      if (timer) { clearInterval(timer); timer = null; }
      document.getElementById('status').innerText = 'Polling paused due to repeated timeout. Use Resume Last Run.';
    }
  }
}

function resumeLastRun(){
  const saved = localStorage.getItem('hydradeck_last_job_id');
  if(!saved){
    document.getElementById('status').innerText = 'No resumable job.';
    return;
  }
  jobId = saved;
  inflight = true;
  refreshFailCount = 0;
  if (timer) clearInterval(timer);
  timer = setInterval(refresh, 1000);
  refresh();
  showTab('tab-console');
}

document.getElementById('base_url').addEventListener('change', loadModels);
document.getElementById('api_key').addEventListener('change', loadModels);
loadModels();
showTab('tab-run');
if(localStorage.getItem('hydradeck_last_job_id')){
  document.getElementById('status').innerText = 'Last run available. Click Resume Last Run to continue.';
}
</script>
</body>
</html>
"""


@app.post("/api/quick-check")
def api_quick_check(req: RunRequest) -> dict[str, str]:
    result = _api_quick_check(req.base_url, req.api_key, req.model, req.request_budget)
    return {"result": result}


@app.post("/api/jobs")
def create_job(req: RunRequest) -> dict[str, str]:
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="topic is required")
    job = _new_job(req)
    with LOCK:
        JOBS[job["id"]] = job
    _prune_history()
    _save_state()
    t = threading.Thread(target=_run_job, args=(job["id"], req), daemon=True)
    t.start()
    return {"id": job["id"]}


@app.get("/api/history")
def get_history() -> dict[str, Any]:
    with LOCK:
        items = sorted(
            JOBS.values(),
            key=lambda j: float(j.get("updated_at", 0.0)),
            reverse=True,
        )
        rows = [
            {
                "id": j.get("id"),
                "status": j.get("status"),
                "progress": j.get("progress"),
                "topic": (j.get("params") or {}).get("topic", ""),
                "updated_at": j.get("updated_at"),
            }
            for j in items[:HISTORY_LIMIT]
        ]
    return {"items": rows}


@app.get("/api/models")
def get_models(base_url: str, api_key: str = "") -> dict[str, Any]:
    try:
        cli = GrokClient(base_url=base_url, api_key=api_key, model="grok-3-mini", timeout_s=20.0, max_retries=1)
        models = cli.list_models(timeout_s=20.0)
        return {"models": models}
    except Exception as exc:
        return {"models": [], "error": str(exc)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return dict(job)


@app.get("/api/jobs/{job_id}/artifact/{kind}")
def get_artifact(job_id: str, kind: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if kind == "paper":
            path = str(job.get("paper_pdf", ""))
            filename = "paper.pdf"
        elif kind == "slides":
            path = str(job.get("slides_pdf", ""))
            filename = "slides.pdf"
        else:
            raise HTTPException(status_code=400, detail="kind must be paper|slides")

    p = Path(path)
    if not path or not p.exists():
        raise HTTPException(status_code=404, detail="artifact not ready")
    return FileResponse(str(p), media_type="application/pdf", filename=filename)


if __name__ == "__main__":
    import uvicorn

    _load_state()
    port = int(os.getenv("PORT", "7861"))
    uvicorn.run(app, host="0.0.0.0", port=port)
