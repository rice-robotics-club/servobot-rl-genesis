from typing import Callable, override

import torch

from src.config import RewardConfig
from src.modules.module import Module

RewardFunc = Callable[[], torch.Tensor]


class RewardManager(Module):
    cfg: RewardConfig
    rewards: dict[str, float]
    reward_functions: dict[str, RewardFunc] = {}
    episode_sums: dict[str, torch.Tensor] = {}

    def __init__(self, num_envs: int, cfg: RewardConfig, device: torch.device | str):
        super().__init__(num_envs, device)
        self.cfg = cfg
        self.rewards = cfg.rewards
        self.buffers["reward"] = torch.zeros((num_envs,), device=self.device)

        for name in self.rewards:
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), dtype=torch.float, device=self.device
            )

    def add_reward_function(self, name: str, func: RewardFunc) -> None:
        self.reward_functions[name] = func

    @override
    def step(self, actions: torch.Tensor):
        Module.step(self, actions)

        self.buffers["reward"].zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.rewards[name]
            self.buffers["reward"] += rew
        self.episode_sums += self.buffers["reward"]
