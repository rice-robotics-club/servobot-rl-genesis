from typing import Any

import torch
from tensordict import TensorDict


class Module:
    """
    A module is a base class for sharing training data between different modules via a shared TensorDict.
    """

    device: torch.device | str
    buffers: TensorDict
    num_envs: int
    children: list["Module"]

    def __init__(self, num_envs: int, device: torch.device | str):
        self.device = device
        self.buffers = TensorDict(batch_size=(num_envs,))
        self.num_envs = num_envs
        self.children = []

    def use(self, module: "Module") -> None:
        if module.buffers.batch_size != self.buffers.batch_size:
            raise ValueError("Batch sizes do not match between modules")

        self.buffers.add(module.buffers)
        module.buffers = self.buffers
        self.children.append(module)

    def step(self, actions: torch.Tensor) -> Any:
        for child in self.children:
            child.step(actions)
