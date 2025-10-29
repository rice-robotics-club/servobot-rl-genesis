import numpy as np
import typing

_DType = typing.TypeVar("_DType", bound=np.generic)

# Type for a numpy array of length 12, representing desired IK for each leg as [leg1..., leg2..., leg3..., leg4...]
PosArray = typing.Annotated[np.typing.NDArray[_DType], typing.Literal[12]]

# Type for a numpy array of length 12, representing configuration of ServoBot
CfgArray = typing.Annotated[np.typing.NDArray[_DType], typing.Literal[12]]

JOINT_NAMES = [
    "FL_Hip",
    "FL_TopLeg",
    "FL_BotLeg",
    "FR_Hip",
    "FR_TopLeg",
    "FR_BotLeg",
    "BL_Hip",
    "BL_TopLeg",
    "BL_BotLeg",
    "BR_Hip",
    "BR_TopLeg",
    "BR_BotLeg",
]


class IK:
    """
    Class for analytically solving IK for ServoBot.
    """

    def __init__(self, off1=0.04064, off2=0.0254, off3=0.01524, thigh=0.109855, calf=0.0762):
        """
        Constructor for IK class, establishing the relevant link parameters for the ServoBot.

        :param off1: offset 1 length in meters
        :param off2: offset 2 length in meters
        :param off3: offset 3 length in meters
        :param thigh: thigh length in meters
        :param calf: calf length in meters
        """
        self.last_valid_cfg = np.zeros(shape=(12,))

        self.off1 = off1
        self.off2 = off2
        self.off3 = off3

        self.thigh = thigh
        self.calf = calf

        # creates pre-defined constants to prevent repeated arithmetic
        self.a = self.thigh ** 2 + self.calf ** 2
        self.b = 2 * self.thigh * self.calf
        self.c = 2 * self.off1
        self.d = self.off1 ** 2

        self.input_off = np.array([self.off3, self.off1, 0] * 4)
        self.output_mult = np.array([1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1])

    def solve(self, positions: PosArray) -> CfgArray:
        """
        Function to solve for IK configuration given input leg positions.

        :param positions: numpy array of shape (12,)
        :return: numpy array of shape (12,)
        """
        output_cfg = []

        positions += self.input_off

        for i in range(4):
            x, y, z = positions[3 * i:3 * i + 3]
            x2_z2 = x ** 2 + z ** 2
            dist_xy = np.sqrt(x2_z2)
            th1 = np.asin(x / dist_xy) - np.asin(self.off3 / dist_xy)
            th3 = np.acos(
                (
                        (y - self.off1) ** 2
                        + (np.sqrt(x2_z2 - self.off3 ** 2) - self.off2) ** 2
                        - self.a
                )
                / self.b
            )
            a = self.b * np.cos(th3) + self.a
            b = np.sqrt(
                (self.thigh + self.calf * np.cos(th3)) ** 2
                * (a + y * self.c - y ** 2 - self.d)
            )
            th2 = np.acos((np.sin(th3) * self.calf * (y - self.off1) + b) / a)
            output_cfg.extend([th1, th2, th3])

        return np.array(output_cfg) * self.output_mult

    def get_idle_cfg(self, height=0.13) -> dict[str, float]:
        """
        Returns a configuration array for
        :param height: robot height in meters
        :return: numpy array of shape (12,)
        """
        positions = np.array([0.0, 0.0, -height] * 4)
        config = self.solve(positions)
        return {JOINT_NAMES[i]: c for (i, c) in enumerate(config.tolist())}


if __name__ == "__main__":
    ik = IK()
    print(ik.get_idle_cfg())
