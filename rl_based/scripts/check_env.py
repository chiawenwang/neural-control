"""
Quick sanity check — runs the env for a few random steps without RL.
Run this first to verify MuJoCo + env work before training.

  python scripts/check_env.py
"""

import sys
from pathlib import Path

import numpy as np
from stable_baselines3.common.env_checker import check_env

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.dlo_env import DLOMidpointEnv, DLOMidpointGoalEnv


def main():
    print("─" * 50)
    print("Checking DLOMidpointEnv …")
    env = DLOMidpointEnv(max_steps=100, random_goal=True)
    check_env(env, warn=True)

    obs, _ = env.reset()
    print(f"  obs shape  : {obs.shape}")
    print(f"  act shape  : {env.action_space.shape}")

    total_reward = 0.0
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, _ = env.reset()

    print(f"  50-step reward (random policy): {total_reward:.3f}")
    print(f"  Final dist to goal: {info['dist']:.4f} m")
    env.close()

    print("\nChecking DLOMidpointGoalEnv (HER) …")
    goal_env = DLOMidpointGoalEnv(max_steps=100)
    check_env(goal_env, warn=True)
    obs, _ = goal_env.reset()
    print(f"  obs keys   : {list(obs.keys())}")
    print(f"  achieved_goal shape: {obs['achieved_goal'].shape}")
    goal_env.close()

    print("\n✓ All checks passed.")


if __name__ == "__main__":
    main()
