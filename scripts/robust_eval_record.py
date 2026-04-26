# Patched copy of IsaacLab/scripts/imitation_learning/robomimic/robust_eval.py that
# records an MP4 of every rollout's table-camera view.
#
# Only additions versus upstream:
#   - --video_dir CLI flag
#   - _RECORDER global + append inside rollout()
#   - per-trial MP4 written from evaluate_model()

"""Run robust evaluation of a robomimic policy, saving rollout MP4s per trial."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate robomimic policy + record rollout MP4s.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--input_dir", type=str, required=True, help="Dir containing .pth checkpoints")
parser.add_argument("--start_epoch", type=int, default=100)
parser.add_argument("--horizon", type=int, default=400)
parser.add_argument("--num_rollouts", type=int, default=15)
parser.add_argument("--num_seeds", type=int, default=3)
parser.add_argument("--seeds", nargs="+", type=int, default=None)
parser.add_argument("--log_dir", type=str, default="/tmp/policy_evaluation_results")
parser.add_argument("--log_file", type=str, default="results")
parser.add_argument("--video_dir", type=str, default=None,
                    help="If set, save one MP4 per rollout here.")
parser.add_argument("--video_fps", type=int, default=30)
parser.add_argument("--image_obs_key", type=str, default="table_cam",
                    help="Obs dict key to record as video.")
parser.add_argument("--norm_factor_min", type=float, default=None)
parser.add_argument("--norm_factor_max", type=float, default=None)
parser.add_argument("--enable_pinocchio", default=False, action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy
import pathlib
import random
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from isaaclab_tasks.utils import parse_env_cfg


_RECORDER = {"frames": None}  # set to a list[np.ndarray H,W,3 uint8] to enable recording


def _tensor_to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    """Accept a H,W,C or C,H,W tensor (uint8 or float [0,1]) and return H,W,3 uint8."""
    a = t.detach().cpu()
    if a.ndim == 4:
        a = a.squeeze(0)
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[-1] not in (1, 3, 4):
        a = a.permute(1, 2, 0)
    arr = a.numpy()
    if arr.dtype != np.uint8:
        arr = (arr.clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def rollout(policy, env: gym.Env, success_term, horizon: int, device: torch.device):
    policy.start_episode()
    obs_dict, _ = env.reset()
    traj = dict(actions=[], obs=[], next_obs=[])

    for _ in range(horizon):
        obs = copy.deepcopy(obs_dict["policy"])
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])

        if hasattr(env.cfg, "image_obs_list"):
            for image_name in env.cfg.image_obs_list:
                if image_name in obs_dict["policy"].keys():
                    image = torch.squeeze(obs_dict["policy"][image_name])
                    image = image.permute(2, 0, 1).clone().float() / 255.0
                    image = image.clip(0.0, 1.0)
                    obs[image_name] = image

        # --- ADDITION: capture frame for video ---
        if _RECORDER["frames"] is not None:
            key = args_cli.image_obs_key
            if key in obs_dict["policy"]:
                _RECORDER["frames"].append(_tensor_to_uint8_hwc(obs_dict["policy"][key]))

        traj["obs"].append(obs)
        actions = policy(obs)
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = ((actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)) / 2 + args_cli.norm_factor_min
        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]
        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs)

        if bool(success_term.func(env, **success_term.params)[0]):
            return True, traj
        elif terminated or truncated:
            return False, traj
    return False, traj


def evaluate_model(model_path, env, device, success_term, num_rollouts, horizon, seed,
                   output_file, setting, video_dir):
    torch.manual_seed(seed)
    env.seed(seed)
    random.seed(seed)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=model_path, device=device, verbose=False)

    results = []
    model_tag = pathlib.Path(model_path).stem  # e.g. model_epoch_600
    for trial in range(num_rollouts):
        print(f"[{setting}][{model_tag}] trial {trial}", flush=True)
        if video_dir:
            _RECORDER["frames"] = []
        terminated, _ = rollout(policy, env, success_term, horizon, device)
        results.append(terminated)

        if video_dir and _RECORDER["frames"]:
            tag = "SUCCESS" if terminated else "FAIL"
            out = os.path.join(video_dir, f"{setting}_seed{seed}_{model_tag}_trial{trial}_{tag}.mp4")
            try:
                imageio.mimsave(out, _RECORDER["frames"], fps=args_cli.video_fps, codec="libx264",
                                quality=7, macro_block_size=None)
                print(f"  wrote {out} ({len(_RECORDER['frames'])} frames, {tag})", flush=True)
            except Exception as e:
                print(f"  video write failed: {e}", flush=True)
            _RECORDER["frames"] = None

        with open(output_file, "a") as fh:
            fh.write(f"[{model_tag}] trial {trial}: {terminated}\n")

    success_rate = results.count(True) / len(results)
    with open(output_file, "a") as fh:
        fh.write(f"[{model_tag}] successful: {results.count(True)}/{len(results)}  rate={success_rate}\n")
        fh.write("-" * 60 + "\n\n")
    print(f"[{model_tag}] success rate {success_rate:.2%}\n", flush=True)
    return success_rate


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1,
                            use_fabric=not args_cli.disable_fabric)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
    env_cfg.eval_mode = True
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    device = TorchUtils.get_torch_device(try_to_use_cuda=False)

    model_checkpoints = [f.name for f in os.scandir(args_cli.input_dir) if f.is_file()]
    seeds = random.sample(range(0, 10000), args_cli.num_seeds) if args_cli.seeds is None else args_cli.seeds
    settings = ["vanilla", "light_intensity", "light_color", "light_texture",
                "table_texture", "robot_texture", "all"]

    os.makedirs(args_cli.log_dir, exist_ok=True)
    if args_cli.video_dir:
        os.makedirs(args_cli.video_dir, exist_ok=True)

    for seed in seeds:
        output_path = os.path.join(args_cli.log_dir, f"{args_cli.log_file}_seed_{seed}")
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        summary = {s: {} for s in settings}
        summary["overall"] = {}

        with open(output_path, "w") as fh:
            for setting in settings:
                env.cfg.eval_type = setting
                fh.write(f"Evaluation setting: {setting}\n" + "=" * 60 + "\n")
                print(f"\n=== setting: {setting} ===", flush=True)
                for model in model_checkpoints:
                    epoch = int(model.split(".")[0].split("_")[-1])
                    if epoch < args_cli.start_epoch:
                        continue
                    sr = evaluate_model(
                        model_path=os.path.join(args_cli.input_dir, model),
                        env=env, device=device, success_term=success_term,
                        num_rollouts=args_cli.num_rollouts, horizon=args_cli.horizon,
                        seed=seed, output_file=output_path,
                        setting=setting, video_dir=args_cli.video_dir,
                    )
                    summary[setting][model] = sr
                    summary["overall"][model] = summary["overall"].get(model, 0.0) + sr
                    env.reset()
                fh.write("=" * 60 + "\n\n")
                env.reset()

            for model in summary["overall"].keys():
                summary["overall"][model] /= len(settings)

            fh.write("\nResults summary (success rate):\n")
            for setting in summary:
                fh.write(f"\nSetting: {setting}\n")
                for model, sr in summary[setting].items():
                    fh.write(f"  {model}: {sr}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
