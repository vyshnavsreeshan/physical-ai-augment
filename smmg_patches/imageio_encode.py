"""Replace SMMG's encode_video() with an imageio-based version.

The upstream version uses `omni.videoencoding`'s NVENC wrapper which fails
on Blackwell with NV_ENC_ERR_INVALID_PARAM. Encoding 120 RGB frames at
1280x704 with libx264-software takes <2 s on a modern CPU, so we sidestep
NVENC entirely.

Replaces only the encode_video function in /workspace/notebook_utils.py.
The Warp shading kernel and all other helpers are kept as-is.
"""

from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path("/workspace/notebook_utils.py")


# The replacement function body.
NEW_ENCODE = '''
def encode_video(root_dir: str, start_frame: int, num_frames: int,
                 camera_name: str, output_path: str, env_num: int, trial_num: int) -> None:
    """Encode shaded-segmentation frames into an MP4 using imageio (libx264).

    Software encoding — works on any GPU (incl. Blackwell where omni.videoencoding's
    NVENC wrapper fails with NV_ENC_ERR_INVALID_PARAM). Warp shading still runs
    on the GPU, only the H.264 encoding is on the CPU.
    """
    import imageio.v2 as imageio

    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    frame_name_pattern = "{camera_name}_{modality}_trial_{trial_num}_tile_{env_num}_step_{frame_idx}.png"

    # Validate all frames exist before starting
    for frame_idx in range(start_frame, start_frame + num_frames):
        for modality in ("normals", "semantic_segmentation"):
            fp = os.path.join(root_dir, frame_name_pattern.format(
                camera_name=camera_name, modality=modality,
                trial_num=trial_num, env_num=env_num, frame_idx=frame_idx))
            if not os.path.exists(fp):
                raise ValueError(f"Missing {modality} frame at index {frame_idx} for trial {trial_num}")

    # Get dimensions from first frame
    first_path = os.path.join(root_dir, frame_name_pattern.format(
        camera_name=camera_name, modality="semantic_segmentation",
        trial_num=trial_num, env_num=env_num, frame_idx=start_frame))
    first_frame = np.array(Image.open(first_path))
    height, width = first_frame.shape[:2]

    # Pre-allocate Warp buffers (same shapes as upstream)
    normals_wp = wp.empty((height, width, 3), dtype=wp.float32, device="cuda")
    segmentation_wp = wp.empty((height, width, 4), dtype=wp.uint8, device="cuda")
    shaded_segmentation_wp = wp.empty_like(segmentation_wp)
    light_source = wp.array(DEFAULT_LIGHT_DIRECTION, dtype=wp.vec3f, device="cuda")

    writer = imageio.get_writer(
        output_path,
        fps=DEFAULT_FRAMERATE,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for frame_idx in range(start_frame, start_frame + num_frames):
            normals_path = os.path.join(root_dir, frame_name_pattern.format(
                camera_name=camera_name, modality="normals",
                trial_num=trial_num, env_num=env_num, frame_idx=frame_idx))
            seg_path = os.path.join(root_dir, frame_name_pattern.format(
                camera_name=camera_name, modality="semantic_segmentation",
                trial_num=trial_num, env_num=env_num, frame_idx=frame_idx))

            normals_np = np.array(Image.open(normals_path)).astype(np.float32) / 255.0
            wp.copy(normals_wp, wp.from_numpy(normals_np))
            segmentation_np = np.array(Image.open(seg_path))
            wp.copy(segmentation_wp, wp.from_numpy(segmentation_np))

            wp.launch(
                _shade_segmentation, dim=(height, width),
                inputs=[segmentation_wp, normals_wp, shaded_segmentation_wp, light_source],
            )

            shaded = shaded_segmentation_wp.numpy()  # H,W,4 uint8
            # Drop alpha for libx264 (yuv420p doesn't carry alpha anyway)
            writer.append_data(np.ascontiguousarray(shaded[..., :3]))
    finally:
        writer.close()
'''.lstrip("\n")


def main() -> int:
    if not TARGET.exists():
        print(f"FAIL: target not found: {TARGET}", file=sys.stderr)
        return 1
    src = TARGET.read_text()

    # Find the start of the original `encode_video` function definition
    start = src.find("def encode_video(")
    if start == -1:
        print("FAIL: could not locate `def encode_video(` in notebook_utils.py", file=sys.stderr)
        return 1

    # The original function is the last definition in the file. Replace from
    # `def encode_video(` to end-of-file.
    head = src[:start]
    new = head + NEW_ENCODE.lstrip()
    TARGET.write_text(new)
    print(f"patched {TARGET} (replaced encode_video with imageio version)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
