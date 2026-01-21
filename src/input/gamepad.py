import pygame

from .input import BaseInput


class Gamepad(BaseInput):
    """Input class for gamepads/controllers with joysticks."""

    def __init__(self, name: None | str = None):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise ValueError("No joysticks found")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

    @property
    def command(self) -> list[float]:
        pygame.event.pump()
        return [
            self._joystick.get_axis(3),
            -self._joystick.get_axis(4),
            -self._joystick.get_axis(0),
        ]
