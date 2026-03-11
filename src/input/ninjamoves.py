import time

from .input import BaseInput


class NinjaMoves(BaseInput):
    """Input class that rapidly cycles through direction changes."""

    SEQUENCE = [
        [0.0, 1.0, 0.0],   # forward
        [0.0, -1.0, 0.0],  # backward
        [1.0, 0.0, 0.0],   # strafe left
        [-1.0, 0.0, 0.0],  # strafe right
        [0.0, 1.0, 1.0],   # forward + turn left
        [0.0, 1.0, -1.0],  # forward + turn right
    ]
    INTERVAL = 0.5  # seconds per direction

    def __init__(self):
        self._start = time.monotonic()

    @property
    def command(self) -> list[float]:
        elapsed = time.monotonic() - self._start
        idx = int(elapsed / self.INTERVAL) % len(self.SEQUENCE)
        return self.SEQUENCE[idx]
