import genesis as gs
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from src.config import Config
from src.modules.module import Module
from src.propogator.propogator import Propogator


class BaseEnv(Module, VecEnv):
    num_actions: int
    num_observations: int
    propogator: Propogator

    extras = {}

    def __init__(
        self,
        num_envs: int,
        cfg: Config,
        device: torch.device | str,
    ):
        super().__init__(num_envs, device)

        self.cfg = cfg.env
        self.num_actions = self.cfg.num_actions
        self.max_episode_length = self.cfg.max_episode_length
        self.episode_length_buf = torch.empty(
            (self.num_envs,), dtype=torch.int, device=self.device
        )

        self.buffers["obs"] = TensorDict(
            {}, batch_size=(self.num_envs), device=self.device
        )
        self.buffers["reset"] = torch.ones(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self.buffers["all_envs"] = torch.arange(self.num_envs, device=self.device)

    def step(self, actions: torch.Tensor):
        Module.step(self, actions)

        self.episode_length_buf += 1

        # check termination and reset
        self.buffers["reset"] = self.episode_length_buf > self.max_episode_length

        self.extras["time_outs"] = (
            self.episode_length_buf > self.max_episode_length
        ).to(dtype=gs.tc_float)

        self.reset(self.buffers["reset"])

        return (
            self.buffers["obs"],
            self.buffers["reward"],
            self.buffers["reset"],
            self.extras,
        )

    def get_observations(self) -> TensorDict:
        return self.buffers["obs"]

    def reset(self, envs_idx: torch.Tensor | None = None):
        if envs_idx is None:
            envs_idx = self.buffers["all_envs"]

        self.buffers["episode_length"].masked_fill_(envs_idx, 0)
        self.buffers["reward"].masked_fill_(envs_idx, 0.0)
