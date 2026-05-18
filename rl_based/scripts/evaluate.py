"""
Evaluate a trained DLO midpoint reaching policy.

Usage
-----
  python evaluate.py --model logs/dlo_sac/best/best_model.zip
  python evaluate.py --model logs/dlo_sac/best/best_model.zip --render
  python evaluate.py --model logs/dlo_her/best/best_model.zip --her
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch as _th

# PyTorch 2.x requires seekable file objects, but SB3 passes non-seekable
# ZipFile members directly. Wrap them in BytesIO before loading.
_orig_th_load = _th.load
def _patched_th_load(f, *args, **kwargs):
    if hasattr(f, "read"):
        f = io.BytesIO(f.read())
    kwargs["weights_only"] = False
    return _orig_th_load(f, *args, **kwargs)
_th.load = _patched_th_load

from stable_baselines3 import SAC, TD3

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.dlo_env import DLOMidpointEnv, DLOMidpointGoalEnv


def run_eval(model, env, n_episodes: int = 50, render: bool = False):
    successes, dists, ep_lens = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done, step = False, 0
        ep_success = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if isinstance(info, list):
                info = info[0]

            if info.get("success", False):
                ep_success = True

        # Extract final distance
        if isinstance(info, dict):
            final_dist = info.get("dist", np.nan)
        else:
            final_dist = np.nan

        successes.append(float(ep_success))
        dists.append(final_dist)
        ep_lens.append(step)

        if render:
            print(f"  Ep {ep+1:3d} | success={ep_success} | dist={final_dist:.4f}m | steps={step}")

    sr   = 100 * np.mean(successes)
    adist = np.mean(dists)
    alen  = np.mean(ep_lens)

    print(f"\n{'─'*45}")
    print(f"  Episodes     : {n_episodes}")
    print(f"  Success rate : {sr:.1f}%")
    print(f"  Mean dist    : {adist:.4f} m")
    print(f"  Mean ep len  : {alen:.1f} steps")
    print(f"{'─'*45}")
    return sr, adist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True, type=str)
    parser.add_argument("--algo",     default="sac",  choices=["sac", "td3"])
    parser.add_argument("--her",      action="store_true")
    parser.add_argument("--render",   action="store_true")
    parser.add_argument("--episodes", default=50, type=int)
    args = parser.parse_args()

    EnvCls = DLOMidpointGoalEnv if args.her else DLOMidpointEnv
    env = EnvCls(
        render_mode="human" if args.render else None,
        random_goal=True,
    )

    AlgoCls = SAC if args.algo == "sac" else TD3
    model = AlgoCls.load(args.model, env=env)

    print(f"\n[Eval] {args.model}")
    run_eval(model, env, n_episodes=args.episodes, render=args.render)
    env.close()


if __name__ == "__main__":
    main()
