"""LIBERO environment wrapper for AdaGRPO.

Provides a gymnasium-compatible interface to LIBERO benchmark tasks.
Supports all four suites: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, LIBERO-Long.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import gymnasium as gym
import numpy as np

logger = logging.getLogger("adagrpo")


class LIBEROWrapper(gym.Env):
    """Gymnasium wrapper for a single LIBERO task."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task_name: str,
        suite: str = "libero_object",
        image_size: int = 128,
        max_episode_steps: int = 300,
        seed: int = 0,
    ):
        super().__init__()
        self.task_name = task_name
        self.suite = suite
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        self._env = None
        self._task_description = ""

        try:
            self._init_libero(task_name, suite, seed)
        except ImportError:
            logger.warning(
                "LIBERO not installed. Using dummy env. Install with: pip install libero"
            )
            self._use_dummy = True
            self.observation_space = gym.spaces.Dict({
                "agentview_image": gym.spaces.Box(0, 255, (3, image_size, image_size), np.uint8),
                "robot0_eef_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
                "robot0_eef_quat": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
                "robot0_gripper_qpos": gym.spaces.Box(-np.inf, np.inf, (2,), np.float32),
            })
            self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
            return

        self._use_dummy = False

    def _init_libero(self, task_name: str, suite: str, seed: int) -> None:
        """Initialize the actual LIBERO environment."""
        from libero.libero import benchmark

        bench = benchmark.get_benchmark(suite)
        task_names = bench.get_task_names()
        if task_name not in task_names:
            raise ValueError(
                f"Task '{task_name}' not found in {suite}. "
                f"Available: {task_names}"
            )
        task_idx = task_names.index(task_name)
        task = bench.get_task(task_idx)
        self._task_description = task.language

        from libero.libero.envs import OffScreenRenderEnv

        env_args = {
            "bddl_file_name": task.bddl_file,
            "camera_heights": self.image_size,
            "camera_widths": self.image_size,
        }
        self._env = OffScreenRenderEnv(**env_args)
        self._env.seed(seed)

        self.observation_space = gym.spaces.Dict({
            "agentview_image": gym.spaces.Box(
                0, 255, (3, self.image_size, self.image_size), np.uint8
            ),
            "robot0_eef_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
            "robot0_eef_quat": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
            "robot0_gripper_qpos": gym.spaces.Box(-np.inf, np.inf, (2,), np.float32),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)

    @property
    def task_description(self) -> str:
        return self._task_description

    def _convert_obs(self, raw_obs: dict) -> dict:
        """Convert LIBERO observation to standard format."""
        img = raw_obs.get("agentview_image", np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
        if img.shape[-1] == 3:
            img = img.transpose(2, 0, 1)  # HWC -> CHW
        return {
            "agentview_image": img,
            "robot0_eef_pos": raw_obs.get("robot0_eef_pos", np.zeros(3, dtype=np.float32)),
            "robot0_eef_quat": raw_obs.get("robot0_eef_quat", np.zeros(4, dtype=np.float32)),
            "robot0_gripper_qpos": raw_obs.get("robot0_gripper_qpos", np.zeros(2, dtype=np.float32)),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict]:
        self._step_count = 0
        if self._use_dummy:
            obs = {k: space.sample() for k, space in self.observation_space.spaces.items()}
            return obs, {"task_description": self.task_name}
        raw_obs = self._env.reset()
        return self._convert_obs(raw_obs), {"task_description": self._task_description}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps

        if self._use_dummy:
            obs = {k: space.sample() for k, space in self.observation_space.spaces.items()}
            reward = float(np.random.random() > 0.95)
            terminated = reward > 0.5
            return obs, reward, terminated, truncated, {}

        raw_obs, reward, done, info = self._env.step(action)
        obs = self._convert_obs(raw_obs)
        terminated = bool(info.get("success", False))
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._env is not None:
            self._env.close()


def make_libero_env(
    task_name: str,
    suite: str = "libero_object",
    image_size: int = 128,
    max_episode_steps: int = 300,
    seed: int = 0,
) -> LIBEROWrapper:
    return LIBEROWrapper(
        task_name=task_name,
        suite=suite,
        image_size=image_size,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )
