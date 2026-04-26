"""Notebook-side client for the Cosmos Transfer 2.5 Flask API.

Replaces SMMG's `cosmos_request.process_video` (which targeted Transfer 1).
Speaks the T2.5 InferenceArguments schema natively — no parameter translation.

Typical use::

    from cosmos_t25_client import transfer

    out = transfer(
        api_url="http://cosmos-api:5000",
        video_path="_isaaclab_out/shaded_seg.mp4",
        output_path="_cosmos_out/photoreal.mp4",
        prompt="a Franka arm stacks marble cubes in a sunlit kitchen",
        seed=42,
        guidance=3,
        num_steps=35,
        sigma_max=70,
        controls={
            "edge": {"control_weight": 0.6},
            "depth": {"control_weight": 0.5},
        },
    )
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests


def healthz(api_url: str, timeout: float = 5.0) -> dict[str, Any]:
    r = requests.get(f"{api_url.rstrip('/')}/healthz", timeout=timeout)
    r.raise_for_status()
    return r.json()


def transfer(
    *,
    api_url: str,
    video_path: str | os.PathLike,
    output_path: str | os.PathLike,
    prompt: str,
    name: str | None = None,
    seed: int = 2025,
    guidance: int = 3,
    num_steps: int = 35,
    sigma_max: float | str | None = None,
    resolution: str = "720",
    negative_prompt: str | None = None,
    controls: dict[str, dict] | None = None,
    control_files: dict[str, str | os.PathLike] | None = None,
    poll_seconds: int = 10,
    max_wait_seconds: int = 3600,
    verbose: bool = True,
) -> dict[str, Any]:
    """Submit a T2.5 transfer job and block until the result is downloaded.

    Args:
        api_url: e.g. ``http://cosmos-api:5000`` (inside docker network) or
                 ``http://localhost:5001`` (from host).
        video_path: input MP4 (the shaded-segmentation video produced by
                    ``notebook_utils.encode_video``).
        output_path: where to save the photoreal MP4 result on this side.
        prompt: text prompt.
        name: optional run name; auto-derived from filename if not given.
        seed, guidance, num_steps, sigma_max, resolution: forwarded to T2.5.
        negative_prompt: forwarded.
        controls: which controlnet branches to enable, e.g.
                  ``{"edge": {"control_weight": 0.6, "preset_edge_threshold": "medium"}}``.
                  Branches with no ``control_path`` and no upload in
                  ``control_files`` will be computed on-the-fly by T2.5 from
                  the input video.
        control_files: optional pre-computed control MP4s, keyed by branch
                  name ("edge"/"depth"/"seg"/"vis").
        poll_seconds: status polling interval.
        max_wait_seconds: how long to wait before giving up.
        verbose: print progress.

    Returns:
        Dict with at least ``{job_id, status, output_path}``.
    """
    api_url = api_url.rstrip("/")
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"input video missing: {video_path}")

    # ── Build the T2.5 spec ────────────────────────────────────────────────
    spec: dict[str, Any] = {
        "name": name or video_path.stem,
        "prompt": prompt,
        "seed": int(seed),
        "guidance": int(guidance),
        "num_steps": int(num_steps),
        "resolution": str(resolution),
    }
    if sigma_max is not None:
        spec["sigma_max"] = str(sigma_max)
    if negative_prompt is not None:
        spec["negative_prompt"] = negative_prompt

    if controls:
        for branch, cfg in controls.items():
            if branch not in {"edge", "depth", "seg", "vis"}:
                raise ValueError(f"unknown control branch: {branch}")
            spec[branch] = dict(cfg)  # copy

    # ── Submit ─────────────────────────────────────────────────────────────
    files = {"video": (video_path.name, open(video_path, "rb"), "video/mp4")}
    if control_files:
        for branch, p in control_files.items():
            files[branch] = (Path(p).name, open(p, "rb"), "video/mp4")

    if verbose:
        print(f"[cosmos] submitting to {api_url}/v1/transfer  spec={spec}")

    try:
        r = requests.post(
            f"{api_url}/v1/transfer",
            data={"spec": json.dumps(spec)},
            files=files,
            timeout=120,
        )
    finally:
        for _, fh, _ in files.values():
            try:
                fh.close()
            except Exception:
                pass

    if r.status_code != 202:
        raise RuntimeError(f"submit failed [{r.status_code}]: {r.text[:500]}")
    job_id = r.json()["job_id"]
    if verbose:
        print(f"[cosmos] job_id={job_id}, polling…")

    # ── Poll ───────────────────────────────────────────────────────────────
    deadline = time.time() + max_wait_seconds
    last_status = None
    while time.time() < deadline:
        s = requests.get(f"{api_url}/v1/jobs/{job_id}", timeout=30).json()
        if s.get("status") != last_status:
            last_status = s.get("status")
            if verbose:
                print(f"[cosmos] status={last_status}")
        if last_status in ("completed", "failed"):
            break
        time.sleep(poll_seconds)
    else:
        raise TimeoutError(f"job {job_id} not finished after {max_wait_seconds}s")

    if last_status != "completed":
        log = s.get("log_tail", "")
        raise RuntimeError(f"job {job_id} failed:\n{log}")

    # ── Download result ────────────────────────────────────────────────────
    r = requests.get(f"{api_url}/v1/jobs/{job_id}/result", timeout=120, stream=True)
    r.raise_for_status()
    with open(output_path, "wb") as out:
        for chunk in r.iter_content(chunk_size=1 << 20):
            out.write(chunk)
    if verbose:
        print(f"[cosmos] saved → {output_path}")

    return {"job_id": job_id, "status": "completed", "output_path": str(output_path)}
