import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from src.config import CommandConfig, EnvConfig, ObsConfig, RewardConfig


class NullEnv(VecEnv):
    """
    An environment for using RSL_RL without training.
    """

    def __init__(self, num_envs, env_cfg: EnvConfig, obs_cfg: ObsConfig):
        self.device = "cpu"
        self.cfg = env_cfg
        self.obs = obs_cfg
        self.num_envs = num_envs
        self.num_actions = len(env_cfg["joints"])
        self.max_episode_length = 1
        self.observations = TensorDict(
            {"main": torch.zeros((num_envs, obs_cfg["num_obs"]), dtype=torch.float64)}, batch_size=num_envs
        )
        self.rewards = torch.empty(
            (self.num_envs,), dtype=torch.float64, device=self.device
        )
        self.reset = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)

    def get_observations(self) -> TensorDict:
        return self.observations

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        return self.observations, self.rewards, self.reset, {}
