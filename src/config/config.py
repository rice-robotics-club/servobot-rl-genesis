from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from .env import EnvConfig
from .obs import ObsConfig
from .reward import RewardConfig
from .runner import RunnerConfig


@dataclass
class Config:
    """
    JSON Schema for rsl_rl configuration YAML files.  The schema uses class_name fields to determine which sub-schema applies.
    """

    num_envs: int = 4096
    """ Number of environments to run in parallel """

    max_iterations: int = 10000
    """ Maximum number of iterations to run """

    log_dir: str | None = None
    """ Directory to log to """

    load_path: str | None = None
    """ Path to load a model from """

    debug: bool = False
    """ Whether to run in debug mode """

    headless: bool = False
    """ Whether to run in headless mode """

    no_print: bool = False
    """ Whether to suppress RSL RL printing """

    runner: RunnerConfig = field(default_factory=RunnerConfig)
    """ Configuration for the RSL RL Runner """

    env: EnvConfig = field(default_factory=EnvConfig)
    """ Configuration for the training environment """

    obs: ObsConfig = field(default_factory=ObsConfig)
    """ List of observation names, with optional scale values """

    reward: RewardConfig = field(default_factory=RewardConfig)
    """ Configuration for the reward system """

    commands: dict[str, tuple[float, float]] = field(default_factory=dict)
    """ List of commands and their associated ranges """


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
