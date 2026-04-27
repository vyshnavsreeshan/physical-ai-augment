# Cosmos Transfer 2.5 NIM — Deployment Guide

The exact procedure that worked for this project. Tested end-to-end on
an NVIDIA **H200 NVL** host (Hopper, 143 GB HBM3e per GPU); the same
steps work on any cu130-compatible NVIDIA GPU (Ampere / Ada / Hopper /
Blackwell) with sufficient VRAM.

> **Image:** `nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:1.0.0`
> **API base:** `http://<host>:8000`
> **Status as of this writeup:** verified working, ~12 min per 35-step / 720p / 120-frame submission.

---

## 1. Prerequisites

### Hardware
- NVIDIA GPU with **≥ 24 GB VRAM** (35 GB+ recommended for headroom on long videos / future concurrent requests)
- **≥ 32 GB host RAM** during model load (peaks ~25 GB)
- **≥ 100 GB free disk** in `~/.cache/nim/` for the model weights
- NVIDIA driver compatible with CUDA 13.0+ (e.g. `550+`)

### Software
- Docker with `nvidia-container-toolkit` installed and configured
- Outbound internet access (the container downloads ~30 GB of model weights from NGC on first start)

### NGC account + API key
This is the part that's unfamiliar if you've never done it before.

1. Sign in / create an account at https://org.ngc.nvidia.com (free; same as NVIDIA Developer)
2. Open https://org.ngc.nvidia.com/setup/api-key
3. Click **Generate API Key**
4. Copy the key — it looks like `nvapi-...` — and save it somewhere safe; you can't retrieve it later

---

## 2. Authenticate Docker against NGC

```bash
# replace with your real key
export NGC_API_KEY=nvapi-XXXXXXXXXXXXXXXXXXXXX

echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
# expect: Login Succeeded
```

Two things are easy to miss:
- The username is the literal string **`$oauthtoken`** (not your NGC username) — keep the single quotes so the shell doesn't try to expand it.
- This `docker login` writes credentials to `~/.docker/config.json`. If you reset the host or use a CI runner, you'll need to log in again.

---

## 3. Pull the image

```bash
docker pull nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:1.0.0
```

The image itself is roughly **30 GB compressed** plus extracted layers
that take similar disk; expect a download of 10–20 minutes on a fast link.
Subsequent pulls are instant (cached).

If `pull access denied` appears: `docker login nvcr.io` failed silently.
Re-run the login step and check the output literally says
`Login Succeeded`.

---

## 4. Create a persistent cache directory

The container caches all downloaded model weights at `/opt/nim/.cache`.
Mounting a host directory there means subsequent container starts re-use
the cache instead of re-downloading.

```bash
mkdir -p ~/.cache/nim
chmod -R 777 ~/.cache/nim 2>/dev/null || true
```

The `chmod 777` is because the NIM container runs as root and writes
files owned by root; the loose permissions just keep it convenient if
you later poke at the cache from your normal user shell.

---

## 5. Run the container

This is the exact `docker run` we used:

```bash
docker run -d --name cosmos-nim \
    --restart unless-stopped \
    --runtime=nvidia --gpus all \
    --shm-size=32GB --ulimit nofile=65536:65536 \
    -e NGC_API_KEY=$NGC_API_KEY \
    -v ~/.cache/nim:/opt/nim/.cache \
    -p 8000:8000 \
    nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:1.0.0
```

What each flag does:

| Flag | Why it's there |
|---|---|
| `--restart unless-stopped` | Auto-recover after host reboots |
| `--runtime=nvidia --gpus all` | Expose all NVIDIA GPUs to the container |
| `--shm-size=32GB` | Triton uses POSIX shared memory for inter-process IPC; 32 GB is comfortable for 720p video |
| `--ulimit nofile=65536:65536` | Triton opens many file descriptors (one per model instance × one per request); the default 1024 is too low |
| `-e NGC_API_KEY` | The container re-authenticates against NGC on first start to download model weights |
| `-v ~/.cache/nim:/opt/nim/.cache` | Persist downloaded weights so subsequent starts don't re-download |
| `-p 8000:8000` | Expose the HTTP API on the host's port 8000 |

If you're running on a multi-GPU host and want to dedicate one GPU,
replace `--gpus all` with `--gpus '"device=0"'` (or `device=1`, etc.).

---

## 6. Wait for ready

The container takes **5–10 minutes on first start** to download model
weights into `/opt/nim/.cache` and then load them onto GPU. Watch:

```bash
docker logs -f cosmos-nim 2>&1 | grep -E "Application is ready|error|Welcome"
```

What you want to see:

```
INFO ...] Welcome! Application is ready to receive API requests.
```

Once that appears, in another terminal:

```bash
curl http://localhost:8000/v1/health/ready
# expect: {"object":"Triton readiness check","message":"ready","status":"ready"}
```

If the readiness check returns 503 or empty: still loading. Wait a
minute and retry. If `docker logs cosmos-nim` shows model files
downloading at near-zero rate: NGC API key auth is broken (re-do step 2)
or the host has no internet.

Subsequent container starts (after `docker stop` + `docker start`) are
much faster — typically ~2 minutes — because the weights are in the
mounted cache.

---

## 7. Probe the API

The NIM exposes Triton-style HTTP endpoints. The ones you care about:

| Endpoint | Method | What it does |
|---|---|---|
| `/v1/health/live` | GET | Process is alive |
| `/v1/health/ready` | GET | Models loaded, ready to serve inference |
| `/v1/infer` | POST | Run a Cosmos Transfer 2.5 inference |
| `/v1/metrics` | GET | Prometheus metrics |
| `/v1/metadata` | GET | Model name + version info |
| `/v1/manifest` | GET | The full NIM manifest |
| `/openapi.json` | GET | The OpenAPI 3.1 schema for `/v1/infer` |

Pull the inference schema to see exactly what fields the request body
accepts:

```bash
curl -s http://localhost:8000/openapi.json | python -m json.tool | less
# the request body is at:
# components.schemas.Transfer2Request
```

The schema we observed (and which the client in this repo uses):

```jsonc
{
  "prompt":          "string  — required",
  "video":           "string  — required, base64 of input MP4 OR URL",
  "negative_prompt": "string  — optional, has a sensible default",
  "image_context":   "string  — optional b64/URL",
  "seed":            "integer — optional",
  "guidance":        "integer 0–7, default 3",
  "num_steps":       "integer ≥ 1, default 35",
  "resolution":      "enum: 256 | 480 | 512 | 720, default 480",
  "num_conditional_frames": "enum: 0 | 1 | 2, default 1",
  "sigma_max":       "float — optional",
  "edge":            "{ control_weight, control } — optional",
  "depth":           "{ control_weight, control } — optional",
  "seg":             "{ control_weight, control } — optional",
  "vis":             "{ control_weight, control } — optional"
}
```

Response shape:
```jsonc
{
  "b64_video": "<base64 of generated MP4>",
  "seed": <int>
}
```

---

## 8. Send your first inference

Using a small input video (the `.mp4` Mimic produces is ~570 KB):

```bash
VIDEO_B64=$(base64 -w 0 your_input.mp4)

cat > /tmp/req.json <<EOF
{
  "prompt": "A photoreal scene of a Franka arm stacking marble cubes on a wooden table.",
  "video": "${VIDEO_B64}",
  "resolution": "480",
  "num_steps": 4,
  "edge": {"control_weight": 1.0}
}
EOF

curl -H 'Content-Type: application/json' -X POST \
     http://localhost:8000/v1/infer \
     -d @/tmp/req.json \
     | jq -r '.b64_video' | base64 -d > /tmp/out.mp4

ffprobe /tmp/out.mp4 2>&1 | grep -E "Duration|Stream"
```

For the smallest meaningful test: `num_steps=4` and `resolution="480"`
finishes in ~90 seconds and confirms the whole pipeline works.
For real output, use `num_steps=35` and `resolution="720"` —
typically ~12 minutes per 120-frame request.

---

## 9. Performance reference

Numbers we observed on an H200 NVL with cache warm (no re-downloads):

| Settings | Wall clock |
|---|---|
| `num_steps=4`,  `resolution=480`, 33 frames | ~90 sec |
| `num_steps=8`,  `resolution=480`, 120 frames | ~3 min |
| `num_steps=35`, `resolution=480`, 120 frames | ~6 min |
| `num_steps=35`, `resolution=720`, 120 frames | **~12 min** ← this is the published-quality target |

Time goes mostly into:
- Diffusion sampling: `num_steps × ~6 sec/step × num_chunks` (videos > 93 frames are chunked)
- Prompt + output guardrails: ~1.5 min combined
- Edge / depth control map computation: ~2-3 min for 120 frames at 720p

Concurrent requests: NIM serializes them through a single Triton worker
by default (`workers_count: 1`). Submitting two at once means the second
waits for the first. To actually parallelize, run two NIM containers
each pinned to one GPU (`--gpus '"device=0"'` and `'"device=1"'`).

---

## 10. Two-machine deployment

If your Isaac Sim host is RAM-tight (e.g., 24 GB), run NIM on a
different machine. The Cosmos load peaks at ~25 GB RSS during model
deserialization, plus Isaac Sim's own footprint, plus headroom — easy
to OOM on one box.

1. On the **NIM host** (e.g. an H200 instance): follow steps 1-7 above.
2. On the **Isaac Sim host**: skip the NIM steps and instead point your
   client at the NIM host:
   ```bash
   export COSMOS_API_URL=http://<nim-host-ip>:8000
   ```
3. Verify reachability:
   ```bash
   curl http://<nim-host-ip>:8000/v1/health/ready
   ```
4. Make sure no firewall is blocking port 8000 between the two hosts.

This is the topology we used and recommend for serious workloads.

---

## 11. Troubleshooting reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `pull access denied` | NGC login expired or never happened | Re-run step 2; check Login Succeeded literally appears |
| Container starts but `/v1/health/ready` returns 503 forever | Still downloading weights from NGC, OR NGC_API_KEY missing | `docker logs -f cosmos-nim` — watch for the download progress; if no progress for 5+ min, restart with NGC_API_KEY set |
| `Access denied. This repository requires approval.` | Some Cosmos Transfer 2.5 sub-checkpoints (e.g. blur) are gated on HuggingFace, not NGC | The NIM doesn't normally need HF — but if it does, set `-e HF_TOKEN=hf_...` (https://huggingface.co/settings/tokens) AND accept the gated repo's terms first |
| `oom_killed=true` on the container | Host RAM wasn't enough during model load | Move to a bigger host or use the two-machine deployment |
| `400 Bad Request: "Extra inputs are not permitted"` | Your client is sending a field NIM doesn't accept | Compare your request body against the OpenAPI schema (`/openapi.json`); commonly `guidance_scale` should be `guidance` and `steps` should be `num_steps` |
| `500 Internal Server Error` from `/v1/infer` | Server-side issue: video metadata invalid (frame count out of 93–480 range), OOM during diffusion, or transient model crash | Check `docker logs cosmos-nim`; `docker restart cosmos-nim` and retry |
| Inference works but produced video is corrupt | Some browsers don't support VP9 in MP4 — the file IS valid, open it in VLC or ffprobe to confirm | If you need h264 specifically, transcode with `ffmpeg -i input.mp4 -c:v libx264 output.mp4` |

---

## Cleanup

```bash
docker stop cosmos-nim
docker rm cosmos-nim
docker rmi nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:1.0.0   # ~30 GB freed
rm -rf ~/.cache/nim                                          # ~30 GB freed
```

The cached weights are by far the biggest disk consumer. Keep them if
you plan to redeploy soon; nuke them if you're done.

---

## References

- NIM for Cosmos quickstart — https://docs.nvidia.com/nim/cosmos/latest/quickstart-guide.html
- API reference — https://docs.nvidia.com/nim/cosmos/latest/api-reference.html
- Cosmos Transfer 2.5 source (the model under the NIM) — https://github.com/nvidia-cosmos/cosmos-transfer2.5
- Cosmos Transfer 2.5 model card — https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B
- NGC API keys — https://org.ngc.nvidia.com/setup/api-key
