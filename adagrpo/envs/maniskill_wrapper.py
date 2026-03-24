"""ManiSkill3 environment wrapper for AdaGRPO.

ManiSkill3 already provides gymnasium-compatible environments. This wrapper
standardises the observation format to match our pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import gymnasium as gym
import numpy as np

logger = logging.getLogger("adagrpo")


class ManiSkillWrapper(gym.Wrapper):
    """Thin wrapper around ManiSkill3 envs to standardise observations."""

    def __init__(
        self,
        env: gym.Env,
        image_size: int = 128,
        max_episode_steps: int = 200,
    ):
        super().__init__(env)
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

    def _convert_obs(self, obs: Any) -> dict:
        """Normalise ManiSkill observation dict."""
        if isinstance(obs, dict):
            result = {}
            # ManiSkill provides images under "sensor_data" or "image"
            for key in ["sensor_data", "image"]:
                if key in obs and isinstance(obs[key], dict):
                    for cam_name, cam_data in obs[key].items():
                        if isinstance(cam_data, dict) and "rgb" in cam_data:
                            img = cam_data["rgb"]
                            if hasattr(img, "cpu"):
                                img = img.cpu().numpy()
                            if img.ndim == 3 and img.shape[-1] == 3:
                                img = img.transpose(2, 0, 1)
                            result["image"] = img
                            break

            # State observations
            if "agent" in obs:
                agent = obs["agent"]
                if isinstance(agent, dict):
                    result["robot0_eef_pos"] = np.array(
                        agent.get("eef_pos", np.zeros(3)), dtype=np.float32
                    )
                    result["robot0_eef_quat"] = np.array(
                        agent.get("eef_quat", np.zeros(4)), dtype=np.float32
                    )
            # Flatten extra if present
            if "extra" in obs and isinstance(obs["extra"], dict):
                for k, v in obs["extra"].items():
                    result[f"extra_{k}"] = np.array(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v
            return result
        # If obs is a flat array
        return {"flat_obs": np.array(obs, dtype=np.float32)}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict]:
        self._step_count = 0
        obs, info = self.env.reset(seed=seed, options=options)
        return self._convert_obs(obs), info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self._step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        truncated = truncated or (self._step_count >= self.max_episode_steps)
        return self._convert_obs(obs), float(reward), terminated, truncated, info


def make_maniskill_env(
    task_name: str = "PickCube-v1",
    obs_mode: str = "rgbd",
    control_mode: str = "pd_ee_delta_pose",
    image_size: int = 128,
    max_episode_steps: int = 200,
    seed: int = 0,
) -> ManiSkillWrapper:
    """Create a ManiSkill3 environment wrapped for AdaGRPO.

    Args:
        task_name: ManiSkill task ID (e.g. 'PickCube-v1', 'StackCube-v1').
        obs_mode: observation mode ('rgbd', 'pointcloud', 'state').
        control_mode: robot control mode.
        image_size: camera resolution.
        max_episode_steps: episode length limit.
        seed: random seed.

    Returns:
        Wrapped ManiSkill environment.
    """
    try:
        import mani_skill.envs  # noqa: F401 — register envs

        env = gym.make(
            task_name,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode="rgb_array",
            max_episode_steps=max_episode_steps,
        )
        return ManiSkillWrapper(env, image_size=image_size, max_episode_steps=max_episode_steps)
    except ImportError:
        logger.warning("ManiSkill not installed. Returning dummy env.")
        # Return a minimal dummy
        env = gym.make("Pendulum-v1")
        return ManiSkillWrapper(env, image_size=image_size, max_episode_steps=max_episode_steps)
