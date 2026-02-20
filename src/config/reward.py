from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class RewardConfig:
    tracking_sigma: float = 0.1
    """ Standard deviation of the tracking reward """

    rewards: dict[str, float] = MISSING
    """ List of rewards and their corresponding scales """

    targets: dict[str, int | float] = field(default_factory=dict)
    """ List of target values """
