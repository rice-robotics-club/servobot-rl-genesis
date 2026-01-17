from abc import ABC


class Twist2D:
    """Class for the 2D twist (velocity) data type, representing x and y axis translation and z axis rotation."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class BaseInput(ABC):
    """Base class for input methods providing updated twist velocity commands for model evaluation."""

    _command: Twist2D
    """Twist field to update on input"""

    @property
    def command(self) -> Twist2D:
        """Get the current twist command from the gamepad."""
        return self._command
