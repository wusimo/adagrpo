"""RoboMimic environment wrapper for AdaGRPO.

Provides gymnasium-compatible interface to RoboMimic benchmark tasks:
Square, Transport, ToolHang, Lift, Can.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import gymnasium as gym
import numpy as np

logger = logging.getLogger("adagrpo")


class RoboMimicWrapper(gym.Env):
    """Gymnasium wrapper for RoboMimic environments."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task_name: str = "Square",
        image_size: int = 84,
        max_episode_steps: int = 400,
        use_image_obs: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.task_name = task_name
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        self._env = None

        try:
            self._init_robomimic(task_name, image_size, use_image_obs, seed)
            self._use_dummy = False
        except (ImportError, Exception) as e:
            logger.warning("RoboMimic not available (%s). Using dummy env.", e)
            self._use_dummy = True
            obs_spaces = {
                "agentview_image": gym.spaces.Box(0, 255, (3, image_size, image_size), np.uint8),
                "robot0_eef_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
                "robot0_eef_quat": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
                "robot0_gripper_qpos": gym.spaces.Box(-np.inf, np.inf, (2,), np.float32),
            }
            self.observation_space = gym.spaces.Dict(obs_spaces)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)

    def _init_robomimic(self, task_name: str, image_size: int, use_image_obs: bool, seed: int) -> None:
        import robomimic.utils.env_utils as EnvUtils

        env_meta = {
            "env_name": task_name,
            "type": 1,  # robosuite type
            "env_kwargs": {
                "has_renderer": False,
                "has_offscreen_renderer": use_image_obs,
                "use_camera_obs": use_image_obs,
                "camera_heights": image_size,
                "camera_widths": image_size,
                "camera_names": ["agentview"],
                "reward_shaping": False,
            },
        }
        self._env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False)
        self._env.env.seed(seed)

        obs_spaces = {
            "agentview_image": gym.spaces.Box(0, 255, (3, image_size, image_size), np.uint8),
            "robot0_eef_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
            "robot0_eef_quat": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
            "robot0_gripper_qpos": gym.spaces.Box(-np.inf, np.inf, (2,), np.float32),
        }
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)

    def _convert_obs(self, raw_obs: dict) -> dict:
        img = raw_obs.get("agentview_image", np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.transpose(2, 0, 1)
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
            return obs, {}
        raw_obs = self._env.reset()
        return self._convert_obs(raw_obs), {}

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
        terminated = bool(done)
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._env is not None:
            self._env.close()


def make_robomimic_env(
    task_name: str = "Square",
    image_size: int = 84,
    max_episode_steps: int = 400,
    seed: int = 0,
) -> RoboMimicWrapper:
    return RoboMimicWrapper(
        task_name=task_name,
        image_size=image_size,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )
