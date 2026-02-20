from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class ObsConfig:
    r"""
    Obs_Config.

    List of observation names, with optional scale values
    """

    num_obs: int = MISSING
    r""" Required property """

    scales: dict[str, float] = field(default_factory=dict)
    r""" Required property """
