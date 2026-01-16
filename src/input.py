import evdev

class Twist2D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

class Gamepad:
    def __init__(self, name: None | str = None):
        self._devices: list[evdev.InputDevice[str]] = [evdev.InputDevice(dev) for dev in evdev.list_devices()]  # pyright: ignore[reportUnknownMemberType]
        self._gamepads: list[evdev.InputDevice[str]] = list(
            filter(
                lambda d: evdev.ecodes.EV_ABS in d.capabilities() and (d.name == name if name else True), 
                self._devices
            )
        )

        if len(self._gamepads) < 1:
            raise Exception("No gamepad found")

        self._gamepad: evdev.InputDevice[str] = self._gamepads.pop()
        self._gamepad.grab()
        self._command: Twist2D = Twist2D()
        
    async def read_loop(self) -> None:
        """Call via asyncio.run(gamepad.read_loop()) to start updating command field asynchronously."""
        async for event in self._gamepad.async_read_loop():
            if event.type == evdev.ecodes.EV_ABS:
                match event.code:
                    case 3:
                        self._command.x = (event.value - 32767.0) / 32768.0
                    case 4:
                        self._command.y = (32767.0-event.value) / 32768.0
                    case 0:
                        self._command.z = (32767.0 -event.value) / 32768.0

    @property
    def command(self) -> Twist2D:
        """Get the current twist command from the gamepad."""
        return self._command