"""Flask API for Cosmos Transfer 2.5 inference.

Lives inside the cosmos-transfer2.5 container. Wraps the
`examples/inference.py -i spec.json -o output/` CLI in a small HTTP server.

Endpoints
─────────
GET  /healthz                      → liveness + GPU info
POST /v1/transfer                  → submit a job, returns {"job_id"}
GET  /v1/jobs/{job_id}             → {"status", "log_tail"}
GET  /v1/jobs/{job_id}/result      → MP4 stream (200 if completed, 404 otherwise)
DELETE /v1/jobs/{job_id}           → drop on-disk artifacts

Submission format
─────────────────
multipart/form-data:
  spec    (required): JSON string matching the T2.5 InferenceArguments schema
                       MINUS path fields (we'll fill those from the uploaded
                       files below).
  video   (required): the input MP4
  edge    (optional): pre-computed edge control MP4
  depth   (optional): pre-computed depth control MP4
  seg     (optional): pre-computed segmentation control MP4
  vis     (optional): pre-computed blur control MP4

The server expands {edge,depth,seg,vis} into the spec's controlnet branches
when that branch is requested in `spec` but no explicit control_path was
given. If a branch has `control_path: null` in `spec` AND no upload was
provided, T2.5 will compute the control on-the-fly from the input video.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from queue import Queue

from flask import Flask, jsonify, request, send_file

# ── Config ──────────────────────────────────────────────────────────────────
JOB_ROOT = Path(os.environ.get("COSMOS_JOB_ROOT", "/tmp/cosmos_jobs"))
COSMOS_REPO = Path(os.environ.get("COSMOS_REPO", "/workspace"))
INFERENCE_PY = COSMOS_REPO / "examples" / "inference.py"
PYTHON_BIN = os.environ.get("COSMOS_PYTHON", "python3")
JOB_ROOT.mkdir(parents=True, exist_ok=True)

CONTROL_KEYS = ("edge", "depth", "seg", "vis")

# ── In-memory state ─────────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_queue: Queue = Queue()


def _set_status(job_id: str, **fields):
    with _jobs_lock:
        _jobs[job_id].update(fields)


def _get_status(job_id: str) -> dict | None:
    with _jobs_lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


# ── Worker ──────────────────────────────────────────────────────────────────
def _worker() -> None:
    while True:
        job_id: str = _queue.get()
        try:
            _run_job(job_id)
        except Exception as e:
            _set_status(job_id, status="failed", error=str(e))
        finally:
            _queue.task_done()


def _run_job(job_id: str) -> None:
    job_dir = JOB_ROOT / job_id
    spec_path = job_dir / "spec.json"
    output_dir = job_dir / "output"
    log_path = job_dir / "run.log"

    _set_status(job_id, status="running", started_at=time.time())

    cmd = [PYTHON_BIN, str(INFERENCE_PY), "-i", str(spec_path), "-o", str(output_dir)]
    with log_path.open("w") as logf:
        logf.write(f"$ {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.run(
            cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(COSMOS_REPO)
        )

    if proc.returncode != 0:
        _set_status(job_id, status="failed", returncode=proc.returncode)
        return

    spec = json.loads(spec_path.read_text())
    name = spec.get("name", "result")
    candidate = output_dir / f"{name}.mp4"
    if not candidate.exists():
        # Cosmos may emit different filename — pick newest mp4 in output_dir
        mp4s = sorted(output_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if not mp4s:
            _set_status(job_id, status="failed", error="no MP4 produced")
            return
        candidate = mp4s[-1]

    _set_status(
        job_id,
        status="completed",
        result_path=str(candidate),
        finished_at=time.time(),
    )


# ── App ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.get("/healthz")
def healthz():
    info = {"status": "ok", "queue_depth": _queue.qsize()}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
             "--format=csv,noheader"], text=True, stderr=subprocess.STDOUT, timeout=5,
        )
        info["gpu"] = out.strip().splitlines()
    except Exception as e:
        info["gpu"] = f"unavailable: {e}"
    return jsonify(info)


@app.post("/v1/transfer")
def submit():
    if "video" not in request.files:
        return jsonify(error="missing required file 'video'"), 400
    if "spec" not in request.form:
        return jsonify(error="missing required form field 'spec'"), 400

    try:
        spec = json.loads(request.form["spec"])
    except json.JSONDecodeError as e:
        return jsonify(error=f"invalid spec JSON: {e}"), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save input video
    video_path = job_dir / "input.mp4"
    request.files["video"].save(video_path)
    spec["video_path"] = str(video_path)

    # Save any uploaded control videos and patch them into the spec
    for ck in CONTROL_KEYS:
        upload = request.files.get(ck)
        if upload is None:
            continue
        cp = job_dir / f"{ck}.mp4"
        upload.save(cp)
        # If user provided a branch but no control_path, fill it; if branch
        # missing entirely, create with default weight 1.0
        branch = spec.get(ck) or {}
        if branch.get("control_path") is None:
            branch["control_path"] = str(cp)
        spec[ck] = branch

    # If prompt was passed inline AND prompt_path missing, write a small file.
    # (Either is valid — we normalize on prompt for clarity.)
    if "prompt" in spec and "prompt_path" not in spec:
        pass  # prompt-as-string is fine

    # Ensure required `name` is present
    if "name" not in spec:
        spec["name"] = f"job_{job_id}"

    spec_path = job_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "submitted_at": time.time(),
            "spec": spec,
        }
    _queue.put(job_id)

    return jsonify(job_id=job_id, status="queued"), 202


@app.get("/v1/jobs/<job_id>")
def status(job_id: str):
    s = _get_status(job_id)
    if s is None:
        return jsonify(error="unknown job_id"), 404
    # Tail recent log
    log_path = JOB_ROOT / job_id / "run.log"
    if log_path.exists():
        log_bytes = log_path.read_bytes()[-4000:]
        s["log_tail"] = log_bytes.decode(errors="replace")
    return jsonify(s)


@app.get("/v1/jobs/<job_id>/result")
def result(job_id: str):
    s = _get_status(job_id)
    if s is None:
        return jsonify(error="unknown job_id"), 404
    if s.get("status") != "completed":
        return jsonify(error=f"job not complete (status={s.get('status')})"), 404
    rp = s.get("result_path")
    if not rp or not os.path.exists(rp):
        return jsonify(error="result file missing"), 410
    return send_file(rp, mimetype="video/mp4", as_attachment=True,
                     download_name=f"{job_id}.mp4")


@app.delete("/v1/jobs/<job_id>")
def cleanup(job_id: str):
    job_dir = JOB_ROOT / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    with _jobs_lock:
        _jobs.pop(job_id, None)
    return jsonify(deleted=True)


if __name__ == "__main__":
    threading.Thread(target=_worker, daemon=True).start()
    port = int(os.environ.get("COSMOS_API_PORT", "5000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
