import sys

from input import BaseInput

if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty

class Keyboard(BaseInput):
    """Input class for keyboards."""

    def __init__(self, name: None | str = None):

        self._settings = saveTerminalSettings()

    def getKey(self, settings):
        if sys.platform == 'win32':
            # getwch() returns a string on Windows
            key = msvcrt.getwch()
        else:
            tty.setraw(sys.stdin.fileno())
            # sys.stdin.read() returns a string on Linux
            key = sys.stdin.read(1)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    def command(self):
        key = self.getKey(self._settings)
        if key == 'w':
            return [1.0, 0.0, 0.0]
        elif key == 's':
            return [-1.0, 0.0, 0.0]
        elif key == 'a':
            return [0.0, 0.0, 1.0]
        elif key == 'd':
            return [0.0, 0.0, -1.0]
        else:
            return [0.0, 0.0, 0.0]
        

