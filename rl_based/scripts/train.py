"""
Train DLO midpoint reaching with SAC (or TD3 / HER+SAC).

Usage
-----
  # Basic SAC
  python train.py

  # SAC with HER (hindsight experience replay)
  python train.py --her

  # TD3 instead of SAC
  python train.py --algo td3

  # Continue from checkpoint
  python train.py --resume logs/dlo_sac/best_model.zip
"""

import argparse
import os
from pathlib import Path

import torch
import numpy as np
from stable_baselines3 import SAC, TD3, HerReplayBuffer
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    ProgressBarCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.dlo_env import DLOMidpointEnv, DLOMidpointGoalEnv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CFG = dict(
    # Environment
    n_envs          = 4,          # parallel envs (set 1 for debugging)
    max_steps       = 500,        # steps per episode
    success_thresh  = 0.03,       # 3 cm
    action_scale    = 0.05,
    random_goal     = True,

    # SAC / TD3
    total_timesteps = 1_000_000,
    learning_rate   = 3e-4,
    buffer_size     = 300_000,
    batch_size      = 256,
    tau             = 0.005,
    gamma           = 0.99,
    learning_starts = 5_000,
    train_freq      = 1,
    gradient_steps  = 1,

    # SAC-specific
    ent_coef        = "auto",
    target_entropy  = "auto",

    # Policy network
    net_arch        = [256, 256],

    # Eval
    eval_freq       = 10_000,
    eval_episodes   = 20,

    # Logging
    log_dir         = "logs",
    run_name        = "dlo_sac",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(use_goal_env: bool, **env_kwargs):
    cls = DLOMidpointGoalEnv if use_goal_env else DLOMidpointEnv
    return lambda: Monitor(cls(**env_kwargs))


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_agent(algo: str, env, cfg: dict, use_her: bool):
    policy_kwargs = dict(net_arch=cfg["net_arch"])

    replay_buffer_cls    = HerReplayBuffer if use_her else None
    replay_buffer_kwargs = (
        dict(n_sampled_goal=4, goal_selection_strategy="future") if use_her else {}
    )

    common = dict(
        policy             = "MlpPolicy",
        env                = env,
        learning_rate      = cfg["learning_rate"],
        buffer_size        = cfg["buffer_size"],
        batch_size         = cfg["batch_size"],
        tau                = cfg["tau"],
        gamma              = cfg["gamma"],
        learning_starts    = cfg["learning_starts"],
        train_freq         = cfg["train_freq"],
        gradient_steps     = cfg["gradient_steps"],
        policy_kwargs      = policy_kwargs,
        replay_buffer_class  = replay_buffer_cls,
        replay_buffer_kwargs = replay_buffer_kwargs,
        verbose            = 1,
        tensorboard_log    = cfg["log_dir"],
        device             = _get_device(),
    )

    if algo == "sac":
        return SAC(
            **common,
            ent_coef       = cfg["ent_coef"],
            target_entropy = cfg["target_entropy"],
        )
    elif algo == "td3":
        n_actions = env.action_space.shape[0]
        noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )
        return TD3(**common, action_noise=noise)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",    default="sac",  choices=["sac", "td3"])
    parser.add_argument("--her",     action="store_true", help="Use HER replay buffer")
    parser.add_argument("--resume",  default=None,   type=str)
    parser.add_argument("--steps",   default=None,   type=int)
    parser.add_argument("--run_name",default=None,   type=str)
    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()
    if args.steps:
        cfg["total_timesteps"] = args.steps
    if args.run_name:
        cfg["run_name"] = args.run_name

    use_goal_env = args.her  # HER requires GoalEnv
    run_dir = Path(cfg["log_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(
        max_steps          = cfg["max_steps"],
        success_threshold  = cfg["success_thresh"],
        action_scale       = cfg["action_scale"],
        random_goal        = cfg["random_goal"],
    )

    # Training env
    train_env = make_vec_env(
        make_env(use_goal_env, **env_kwargs),
        n_envs=cfg["n_envs"],
    )

    # Eval env (single, deterministic goal for fair comparison)
    eval_env = make_vec_env(
        make_env(use_goal_env, **{**env_kwargs, "random_goal": True}),
        n_envs=1,
    )

    # Build or restore agent
    if args.resume:
        print(f"[INFO] Resuming from {args.resume}")
        AlgoCls = SAC if args.algo == "sac" else TD3
        model   = AlgoCls.load(args.resume, env=train_env)
    else:
        model = build_agent(args.algo, train_env, cfg, use_goal_env)

    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(run_dir / "best"),
        log_path             = str(run_dir / "eval_logs"),
        eval_freq            = cfg["eval_freq"] // cfg["n_envs"],
        n_eval_episodes      = cfg["eval_episodes"],
        deterministic        = True,
        render               = False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq   = 50_000 // cfg["n_envs"],
        save_path   = str(run_dir / "checkpoints"),
        name_prefix = cfg["run_name"],
    )
    callbacks = CallbackList([eval_cb, ckpt_cb, ProgressBarCallback()])

    print(f"\n{'='*60}")
    print(f"  Algorithm : {args.algo.upper()}" + (" + HER" if use_goal_env else ""))
    print(f"  Run dir   : {run_dir}")
    print(f"  Steps     : {cfg['total_timesteps']:,}")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps   = cfg["total_timesteps"],
        callback          = callbacks,
        reset_num_timesteps = not bool(args.resume),
        tb_log_name       = cfg["run_name"],
    )

    model.save(str(run_dir / "final_model"))
    print(f"\n[DONE] Final model saved to {run_dir / 'final_model'}")


if __name__ == "__main__":
    main()
