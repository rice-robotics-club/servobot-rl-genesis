import threading

import evdev

from .input import BaseInput


class Gamepad(BaseInput):
    """Input class for gamepads/controllers with joysticks."""

    def __init__(self, name: None | str = None):
        self._devices: list[evdev.InputDevice[str]] = [
            evdev.InputDevice(dev) for dev in evdev.list_devices()
        ]
        self._gamepads: list[evdev.InputDevice[str]] = list(
            filter(
                lambda d: evdev.ecodes.EV_ABS in d.capabilities()
                and (d.name == name if name else True),
                self._devices,
            )
        )

        if len(self._gamepads) < 1:
            raise Exception("No gamepad found")

        self._gamepad: evdev.InputDevice[str] = self._gamepads.pop()
        self._gamepad.grab()

        print(f"Gamepad initialized: {self._gamepad.name}")

        self._command = [0.0, 0.0, 0.0]

        t = threading.Thread(target=self._read_loop)
        t.daemon = True
        t.start()

    def _read_loop(self) -> None:
        for event in self._gamepad.read_loop():
            if event.type == evdev.ecodes.EV_ABS:
                match event.code:
                    case 4:
                        self._command[0] = (65535.0 - event.value) / 65535.0
                    case 3:
                        self._command[1] = (event.value) / 65535.0
                    case 0:
                        self._command[2] = (65535.0 - event.value) / 65535.0
