"""
DLO Middle-Point Reaching Environment
=====================================
A 2-DOF planar arm manipulates one end of a rope (DLO).
The task: move the rope's **midpoint** to a target 3-D position.

Observation
-----------
  - shoulder angle, elbow angle          (2,)
  - shoulder velocity, elbow velocity    (2,)
  - midpoint xyz                         (3,)
  - goal xyz                             (3,)
  - rope joint angles (14 hinge dofs)    (14,)
  ──────────────────────────────────────────
  Total obs dim: 24

Action
------
  - Δshoulder, Δelbow  ∈ [-1, 1]  (delta position commands, scaled)

Reward
------
  - Dense: −‖mid − goal‖₂
  - Bonus: +1.0 when within success_threshold
  - Time penalty: −0.01 per step (encourages speed)

Episode termination
-------------------
  - Success (mid within threshold) or max_steps reached.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import mujoco
    _USE_DM = False
except ImportError:
    try:
        import dm_control.mujoco as mujoco
        _USE_DM = True
    except ImportError:
        raise ImportError(
            "Install mujoco (pip install mujoco) or dm_control to use this env."
        )


XML_PATH = Path(__file__).parent / "dlo_scene.xml"

# Rope hinge joint names (x + y for each segment)
ROPE_JOINT_NAMES = [
    f"rj{i}{ax}" for i in range(7) for ax in ["x", "y"]
]
ARM_JOINT_NAMES = ["shoulder", "elbow"]


class DLOMidpointEnv(gym.Env):
    """Gymnasium env: move rope midpoint to a target position."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        goal_range: float = 0.25,
        success_threshold: float = 0.03,
        max_steps: int = 500,
        action_scale: float = 0.05,
        render_mode: Optional[str] = None,
        random_goal: bool = True,
    ):
        super().__init__()

        self.goal_range = goal_range
        self.success_threshold = success_threshold
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.render_mode = render_mode
        self.random_goal = random_goal

        # --- Load MuJoCo model ---
        self._model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self._data  = mujoco.MjData(self._model)

        # --- Cache body / joint IDs ---
        self._shoulder_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder"
        )
        self._elbow_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "elbow"
        )
        self._goal_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "goal_body"
        )
        self._mid_site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "mid_site"
        )
        self._rope_joint_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ROPE_JOINT_NAMES
        ]

        # --- Actuator IDs ---
        self._act_shoulder_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_shoulder"
        )
        self._act_elbow_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_elbow"
        )

        # --- Spaces ---
        # obs: shoulder_angle, elbow_angle, shoulder_vel, elbow_vel,
        #      mid_xyz (3), goal_xyz (3), rope_joints (14)
        obs_dim = 2 + 2 + 3 + 3 + len(ROPE_JOINT_NAMES)  # = 24
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._step_count = 0
        self._goal_pos   = np.zeros(3)

        # Renderer / viewer (lazy init)
        self._renderer = None
        self._viewer   = None  # mujoco.viewer for human mode (Mac-compatible)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)

        # Randomise arm init pose slightly
        self._data.qpos[self._shoulder_id] = self.np_random.uniform(-0.3, 0.3)
        self._data.qpos[self._elbow_id]    = self.np_random.uniform(-0.5, 0.5)

        # Set goal position
        if self.random_goal:
            r   = self.np_random.uniform(0.1, self.goal_range)
            phi = self.np_random.uniform(-np.pi / 2, np.pi / 2)
            z   = self.np_random.uniform(0.1, 0.45)
            self._goal_pos = np.array([r * np.cos(phi), r * np.sin(phi), z])
        else:
            self._goal_pos = np.array([0.3, 0.0, 0.3])

        # Move goal body (visual marker)
        self._model.body_pos[self._goal_body_id] = self._goal_pos

        # Let rope settle
        for _ in range(50):
            mujoco.mj_step(self._model, self._data)

        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)

        # Delta position control
        new_shoulder = (
            self._data.ctrl[self._act_shoulder_id] + action[0] * self.action_scale
        )
        new_elbow = (
            self._data.ctrl[self._act_elbow_id] + action[1] * self.action_scale
        )
        self._data.ctrl[self._act_shoulder_id] = np.clip(new_shoulder, -3.14, 3.14)
        self._data.ctrl[self._act_elbow_id]    = np.clip(new_elbow, -2.8, 2.8)

        # Step simulation (5 physics steps per control step)
        for _ in range(5):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        obs = self._get_obs()
        mid_pos = self._get_midpoint()
        dist    = np.linalg.norm(mid_pos - self._goal_pos)

        # Reward
        reward  = -dist
        reward -= 0.01  # time penalty

        success = dist < self.success_threshold
        if success:
            reward += 1.0

        terminated = bool(success)
        truncated  = self._step_count >= self.max_steps

        info = {
            "dist":    float(dist),
            "success": success,
            "mid_pos": mid_pos.copy(),
            "goal_pos": self._goal_pos.copy(),
        }

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            # mujoco.viewer works correctly on Mac (both Apple Silicon and Intel)
            if self._viewer is None:
                import mujoco.viewer as mjviewer
                self._viewer = mjviewer.launch_passive(self._model, self._data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self._model, height=480, width=640)
            self._renderer.update_scene(self._data)
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_midpoint(self) -> np.ndarray:
        """Return world-frame xyz of the rope midpoint site."""
        return self._data.site_xpos[self._mid_site_id].copy()

    def _get_obs(self) -> np.ndarray:
        shoulder_q   = self._data.qpos[self._shoulder_id]
        elbow_q      = self._data.qpos[self._elbow_id]
        shoulder_v   = self._data.qvel[self._shoulder_id]
        elbow_v      = self._data.qvel[self._elbow_id]
        mid_pos      = self._get_midpoint()
        rope_angles  = np.array([
            self._data.qpos[jid] for jid in self._rope_joint_ids
        ])
        return np.concatenate([
            [shoulder_q, elbow_q],
            [shoulder_v, elbow_v],
            mid_pos,
            self._goal_pos,
            rope_angles,
        ]).astype(np.float32)


    # ------------------------------------------------------------------
    # Goal-conditioned helpers (for HER)
    # ------------------------------------------------------------------

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compatible with HerReplayBuffer (stable-baselines3)."""
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(dist > self.success_threshold).astype(np.float32)


# ---------------------------------------------------------------------------
# GoalEnv wrapper (for HER support)
# ---------------------------------------------------------------------------

class DLOMidpointGoalEnv(gym.GoalEnv, DLOMidpointEnv):
    """
    GoalEnv variant: obs dict has keys 'observation', 'achieved_goal',
    'desired_goal'.  Required for Hindsight Experience Replay.
    """

    def __init__(self, **kwargs):
        DLOMidpointEnv.__init__(self, **kwargs)
        obs_dim = 2 + 2 + 3 + len(ROPE_JOINT_NAMES)  # without goal
        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(-np.inf, np.inf, (obs_dim,),  np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, (3,),         np.float32),
            "desired_goal":  spaces.Box(-np.inf, np.inf, (3,),         np.float32),
        })

    def _get_obs(self):
        shoulder_q  = self._data.qpos[self._shoulder_id]
        elbow_q     = self._data.qpos[self._elbow_id]
        shoulder_v  = self._data.qvel[self._shoulder_id]
        elbow_v     = self._data.qvel[self._elbow_id]
        mid_pos     = self._get_midpoint()
        rope_angles = np.array([
            self._data.qpos[jid] for jid in self._rope_joint_ids
        ])
        obs = np.concatenate([
            [shoulder_q, elbow_q],
            [shoulder_v, elbow_v],
            mid_pos,
            rope_angles,
        ]).astype(np.float32)

        return {
            "observation":   obs,
            "achieved_goal": mid_pos.astype(np.float32),
            "desired_goal":  self._goal_pos.astype(np.float32),
        }

    def reset(self, **kwargs):
        obs_dict, info = DLOMidpointEnv.reset(self, **kwargs)
        return obs_dict, info

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = DLOMidpointEnv.step(self, action)
        return obs_dict, reward, terminated, truncated, info
