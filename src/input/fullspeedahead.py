from .input import BaseInput


class FullSpeedAhead(BaseInput):
    """Input class that always commands full speed ahead."""

    @property
    def command(self) -> list[float]:
        return [0.0, 1.0, 0.0]