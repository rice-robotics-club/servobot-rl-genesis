from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PPOConfig:
    class_name: Literal["PPO"] = "PPO"
    """
    The algorithm class to use.
    """


@dataclass
class RunnerConfigObsGroups:
    """Maps observation groups to sets of observation types"""

    policy: list[str] = field(default_factory=lambda: ["policy"])
    """ Observation sets for the policy/actor """

    critic: list[str] = field(default_factory=lambda: ["policy"])
    """ Observation sets for the critic (PPO only) """

    teacher: list[str] = field(default_factory=list)
    """ Observation sets for the teacher (Distillation only) """


@dataclass
class RunnerConfig:
    """
    Runner configuration. Use class_name to select between OnPolicyRunner and DistillationRunner.
    """

    class_name: Literal["OnPolicyRunner", "DistillationRunner"] = "OnPolicyRunner"
    """
    The runner class to use.
    """

    num_steps_per_env: int = 24
    """
    Number of steps per environment per iteration
    """

    max_iterations: int = 1000
    """
    Number of policy updates
    """

    seed: int = 0
    """ Random seed for reproducibility """

    obs_groups: "RunnerConfigObsGroups" = field(default_factory=RunnerConfigObsGroups)
    """ Maps observation groups to sets of observation types """

    save_interval: int = 100
    """ Check for potential saves every n iterations """

    experiment_name: str = "default_experiment"
    """ Name of the experiment """

    run_name: str = "default_run"
    """ Name of the specific run """

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """ Logging backend to use """

    neptune_project: str = ""
    """ Neptune project name (when logger=neptune) """

    wandb_project: str = ""
    """ Weights & Biases project name (when logger=wandb) """

    # policy: Required["_RunnerFullStopJsonNumberSignDefsPolicy"]
    # """ Policy configuration (generic) """

    # algorithm: Required["_RunnerFullStopJsonNumberSignDefsAlgorithm"]
    # """ Algorithm configuration (generic) """
