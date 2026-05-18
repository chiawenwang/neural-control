# DLO Midpoint Reaching — RL with MuJoCo

Reinforcement learning for **deformable linear object (rope) midpoint reaching** using a 2-DOF planar arm in MuJoCo.

## Project structure

```
dlo_rl/
├── envs/
│   ├── dlo_scene.xml     # MuJoCo model: arm + rope + goal marker
│   └── dlo_env.py        # Gymnasium env (DLOMidpointEnv + GoalEnv variant)
├── scripts/
│   ├── check_env.py      # Sanity check (run first!)
│   ├── train.py          # Training: SAC / TD3 / HER+SAC
│   └── evaluate.py       # Evaluation + optional render
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Quickstart

```bash
# 1. Verify the env works
python scripts/check_env.py

# 2. Train SAC (default)
python scripts/train.py

# 3. Train SAC + HER (much better sample efficiency)
python scripts/train.py --her --run_name dlo_her

# 4. Train TD3
python scripts/train.py --algo td3 --run_name dlo_td3

# 5. Evaluate
python scripts/evaluate.py --model logs/dlo_sac/best/best_model.zip

# 6. Monitor with TensorBoard
tensorboard --logdir logs/
```

## Environment details

| Property | Value |
|---|---|
| Obs dim | 24 |
| Action dim | 2 (Δshoulder, Δelbow) |
| Episode length | 500 steps |
| Success threshold | 3 cm |
| Reward | −dist − 0.01 + 1.0 on success |
| Physics dt | 2 ms (5 steps per control) |

### Observation vector (dim=24)

| Slice | Content |
|---|---|
| [0:2] | Shoulder & elbow angles (rad) |
| [2:4] | Shoulder & elbow velocities (rad/s) |
| [4:7] | Rope midpoint XYZ (m) |
| [7:10] | Goal position XYZ (m) |
| [10:24] | Rope hinge joint angles (14 DOF) |

## Algorithm recommendations

| Scenario | Recommended |
|---|---|
| Baseline / easy goal | SAC |
| Sparse reward / hard goal | SAC + HER (`--her`) |
| Deterministic policy needed | TD3 |

## Extending the env

- **3-DOF arm**: Add a wrist joint to `dlo_scene.xml` + actuator, update joint IDs in `dlo_env.py`.
- **Fixed end**: Add a `weld` equality constraint in XML to pin the rope's free end.
- **Longer rope**: Increase the number of rope segments (add more `rope_N` bodies).
- **Image obs**: Add a MuJoCo `camera` and return rendered pixels from `_get_obs`.
- **Curriculum**: Gradually increase `goal_range` and decrease `success_threshold` as training progresses.
