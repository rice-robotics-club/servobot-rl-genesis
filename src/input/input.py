from abc import ABC


class BaseInput(ABC):
    """Base class for input methods providing updated twist velocity commands for model evaluation."""

    _command: list[float]
    """Twist field to update on input"""

    @property
    def command(self) -> list[float]:
        """Get the current twist command from the gamepad."""
        return self._command
