"""Notebook-side client for the **NVIDIA Cosmos Transfer 2.5 NIM**.

Talks to nvcr.io/nim/nvidia/cosmos-transfer2.5-2b's `/v1/infer` endpoint
directly. NIM keeps models loaded persistently and handles concurrent
requests internally — no submit-then-poll, no per-job warm-up.

Drop-in replacement for the previous Flask-wrapper client. Same `transfer()`
signature so notebook cells don't change.

Typical use::

    from cosmos_t25_client import transfer

    out = transfer(
        api_url="http://10.79.252.45:8000",
        video_path="_isaaclab_out/shaded_seg.mp4",
        output_path="_cosmos_out/photoreal.mp4",
        prompt="a Franka arm stacks marble cubes in a sunlit kitchen",
        seed=42,
        guidance_scale=3,
        steps=35,
        controls={
            "edge": {"control_weight": 0.6},
            "depth": {"control_weight": 0.5},
        },
    )

Endpoints used (reference: NVIDIA NIM docs):
    GET  /v1/health/ready    → liveness gate
    POST /v1/infer           → the inference call (sync, returns b64 video)
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

import requests


def healthz(api_url: str, timeout: float = 5.0) -> dict[str, Any]:
    """Hit NIM's readiness endpoint. Returns the JSON body on success."""
    r = requests.get(f"{api_url.rstrip('/')}/v1/health/ready", timeout=timeout)
    r.raise_for_status()
    return r.json()


def transfer(
    *,
    api_url: str,
    video_path: str | os.PathLike,
    output_path: str | os.PathLike,
    prompt: str,
    name: str | None = None,                    # kept for API compatibility, unused by NIM
    seed: int | None = 2025,
    guidance: int = 3,
    num_steps: int = 35,
    resolution: str = "720",
    negative_prompt: str | None = None,
    sigma_max: float | None = None,
    num_conditional_frames: int = 1,
    image_context: str | None = None,
    controls: dict[str, dict] | None = None,
    poll_seconds: int = 0,                       # kept for API compat (ignored — NIM is sync)
    max_wait_seconds: int = 3600,
    request_timeout: int | None = None,          # HTTP timeout in seconds (default ≈ max_wait)
    verbose: bool = True,
    # Compat shims for parameter renames ───────────────────────────────────
    guidance_scale: int | None = None,           # alias → guidance
    steps: int | None = None,                    # alias → num_steps
    control_files: dict[str, str | os.PathLike] | None = None,  # ignored — NIM takes inline only
) -> dict[str, Any]:
    """Submit a Cosmos Transfer 2.5 NIM inference and block until result is saved.

    Args:
        api_url: e.g. ``http://<h200>:8000`` (NIM's port is 8000, not 5000).
        video_path: input MP4 (the shaded-segmentation video produced by
                    ``notebook_utils.encode_video``).
        output_path: where to save the photoreal MP4 result on this side.
        prompt: text prompt.
        name: ignored by NIM, kept for compatibility with the old client.
        seed, guidance_scale, steps, resolution, negative_prompt: forwarded.
        controls: which controlnet branches to enable, e.g.
                  ``{"edge": {"control_weight": 0.6}, "depth": {"control_weight": 0.5}}``.
                  Branches with no ``control`` value will be computed on-the-fly
                  by NIM from the input video.
        request_timeout: HTTP timeout for the synchronous `/v1/infer` POST.
                         If None, defaults to ``max_wait_seconds``.
        verbose: print progress.

        guidance / num_steps: legacy parameter names — auto-mapped to
                              guidance_scale / steps for backward compat.
        sigma_max, control_files: legacy parameters NIM does not accept;
                                  silently ignored with a verbose note.

    Returns:
        Dict with ``{job_id, status, output_path, seed}``. ``job_id`` is
        synthesised from the response timestamp since NIM doesn't issue
        one — kept for API compatibility.
    """
    # Map legacy parameter names ───────────────────────────────────────────
    if guidance_scale is not None:
        guidance = guidance_scale
    if steps is not None:
        num_steps = steps
    if verbose and control_files:
        print("[cosmos] note: control_files (pre-computed control mp4s) "
              "are not currently passed through; NIM will compute controls on-the-fly")

    api_url = api_url.rstrip("/")
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"input video missing: {video_path}")

    # Build NIM request body ───────────────────────────────────────────────
    with open(video_path, "rb") as fh:
        video_b64 = base64.b64encode(fh.read()).decode("ascii")

    body: dict[str, Any] = {
        "prompt": prompt,
        "video": video_b64,
        "resolution": str(resolution),
        "guidance": int(guidance),
        "num_steps": int(num_steps),
        "num_conditional_frames": int(num_conditional_frames),
    }
    if seed is not None:
        body["seed"] = int(seed)
    if sigma_max is not None:
        body["sigma_max"] = float(sigma_max)
    if negative_prompt is not None:
        body["negative_prompt"] = negative_prompt
    if image_context is not None:
        body["image_context"] = image_context
    if controls:
        for branch, cfg in controls.items():
            if branch not in {"edge", "depth", "seg", "vis"}:
                raise ValueError(f"unknown control branch: {branch}")
            # NIM accepts {"control_weight": float, optional "control": <url|base64>}
            body[branch] = dict(cfg)

    timeout = request_timeout if request_timeout is not None else max_wait_seconds

    if verbose:
        # Don't dump the (huge) base64 in the log.
        log_view = {k: ("<b64 video, %d bytes>" % len(video_b64))
                       if k == "video" else v for k, v in body.items()}
        print(f"[cosmos] POST {api_url}/v1/infer  body={log_view}")

    t0 = time.time()
    r = requests.post(
        f"{api_url}/v1/infer",
        json=body,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    elapsed = time.time() - t0

    if r.status_code != 200:
        raise RuntimeError(
            f"NIM /v1/infer returned {r.status_code} after {elapsed:.0f}s:\n"
            f"{r.text[:2000]}"
        )

    payload = r.json()
    if "b64_video" not in payload:
        raise RuntimeError(f"NIM response missing b64_video: {list(payload.keys())}")

    video_bytes = base64.b64decode(payload["b64_video"])
    output_path.write_bytes(video_bytes)

    if verbose:
        print(f"[cosmos] saved → {output_path}  "
              f"({len(video_bytes):,} bytes, {elapsed:.1f}s, seed={payload.get('seed')})")

    return {
        "job_id": f"nim-{int(t0)}",
        "status": "completed",
        "output_path": str(output_path),
        "seed": payload.get("seed"),
        "elapsed_seconds": elapsed,
    }
