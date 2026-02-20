from abc import abstractmethod

from torch import Tensor

from src.modules.module import Module


class Propogator(Module):
    """
    The propogator abstract class defines an interface for a physics propopator,
    (simulation, IRL, etc.).
    """

    @abstractmethod
    def step(self, actions: Tensor, commands: Tensor | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self, idx: Tensor | None = None) -> None:
        raise NotImplementedError
