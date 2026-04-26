"""Smoke test inside SMMG container.

Run via:
  docker run --rm --gpus all -v $(pwd)/scripts:/smoke \
      --entrypoint python3.11 physical-ai-lab/smmg-blackwell:1.0 /smoke/smmg_smoke.py
"""

import sys


def main() -> int:
    print("[1/4] torch + Blackwell sm_120 matmul")
    import torch
    print(f"      torch={torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}")
    if not torch.cuda.is_available():
        print("FAIL: cuda not available")
        return 1
    x = torch.randn(512, 512, device="cuda")
    y = x @ x.T
    torch.cuda.synchronize()
    print(f"      matmul OK shape={tuple(y.shape)} dtype={y.dtype}")

    print("[2/4] launch Isaac Sim AppLauncher (headless, cameras)")
    from isaaclab.app import AppLauncher
    import argparse
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--headless", "--enable_cameras"])
    launcher = AppLauncher(args)
    print("      app launched")

    print("[3/4] import isaaclab + isaaclab_mimic")
    import isaaclab
    import isaaclab_mimic.envs  # registers Blueprint-Mimic
    import gymnasium as gym
    n = sum(1 for k in gym.envs.registry if "Blueprint-Mimic" in k)
    print(f"      isaaclab v={getattr(isaaclab,'__version__','?')}  blueprint-mimic envs={n}")
    if n == 0:
        print("FAIL: Blueprint-Mimic env not registered")
        launcher.app.close()
        return 1

    print("[4/4] gym.make Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0 (real test of Blackwell + Mimic)")
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(
        "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0",
        device="cuda:0", num_envs=1, use_fabric=True,
    )
    env = gym.make("Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0", cfg=env_cfg).unwrapped
    print(f"      env created OK: {type(env).__name__}")
    print(f"      action_space={env.action_space}")
    obs, _ = env.reset()
    print(f"      reset OK; obs keys={list(obs.get('policy', {}).keys())}")

    env.close()
    launcher.app.close()
    print("OK — Mimic env runs on Blackwell in this container")
    return 0


if __name__ == "__main__":
    sys.exit(main())
