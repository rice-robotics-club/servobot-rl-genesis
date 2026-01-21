from abc import ABC, abstractmethod


class BaseInput(ABC):
    """Base class for input methods providing updated twist velocity commands for model evaluation."""

    @property
    @abstractmethod
    def command(self) -> list[float]:
        """Get the current twist command from the gamepad, """
        raise NotImplementedError
