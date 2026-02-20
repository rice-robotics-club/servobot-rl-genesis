from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class EnvConfig:
    """Configuration for the training environment"""

    class_name: str = MISSING
    """ Name of the environment class """

    max_episode_length: int = 1000
    """ Maximum episode length in steps """

    num_actions: int = MISSING
    """ Number of actions """

    urdf_path: str = MISSING
    """ Path to the robot URDF file """

    joints: dict[str, float] = MISSING
    """ Joint names and their default angles """

    kp: float = 20.0
    """ Position proportional gain for joint PD controller """

    kv: float = 0.5
    """ Velocity derivative gain for joint PD controller """

    base_init_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """ Initial base position [x, y, z] in meters """

    base_init_quat: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    """ Initial base orientation quaternion [w, x, y, z] """

    episode_length: float = 10.0
    """ Maximum episode length in seconds """

    resampling_time: float = 1.0
    """ Time interval for resampling goals/actions in seconds """

    action_scale: float = 1.0
    """ Scaling multiplier applied to actions """

    simulate_action_latency: bool = True
    """ Whether to simulate action latency """

    clip_actions: float = 100.0
    """ Action clipping magnitude """

    dt: float = 1e-2
    """ Time duration for each simulation step in seconds, defaults to 1e-2 """
