from typing import Literal, Required, TypedDict


CommandConfig = dict[str, "_CommandConfigAdditionalproperties"]
r"""
Command_Config.

List of commands and their associated ranges
"""



class Config(TypedDict, total=False):
    r"""
    Config.

    JSON Schema for rsl_rl configuration YAML files.  The schema uses class_name fields to determine which sub-schema applies.

    $comment: Produced with Claude Opus 4.5, use with caution.
    """

    runner: Required["RunnerConfig"]
    r""" Required property """

    env: Required["EnvConfig"]
    r"""
    Env_Config.

    Configuration for the training environment

    Required property
    """

    obs: Required["ObsConfig"]
    r"""
    Obs_Config.

    List of observation names, with optional scale values

    Required property
    """

    reward: Required["RewardConfig"]
    r"""
    Reward_Config.

    Required property
    """

    commands: Required["CommandConfig"]
    r"""
    Command_Config.

    List of commands and their associated ranges

    Required property
    """



class EnvConfig(TypedDict, total=False):
    r"""
    Env_Config.

    Configuration for the training environment
    """

    class_name: Required[str]
    r"""
    Name of the environment class

    Required property
    """

    urdf_path: Required[str]
    r"""
    Path to the robot URDF file

    Required property
    """

    joints: Required[dict[str, int | float]]
    r"""
    Joint names and their default angles

    Required property
    """

    kp: int | float
    r""" Position proportional gain for joint PD controller """

    kd: int | float
    r""" Velocity derivative gain for joint PD controller """

    termination_if_roll_greater_than: int | float
    r""" Terminate episode if base roll angle exceeds this value (degrees) """

    termination_if_pitch_greater_than: int | float
    r""" Terminate episode if base pitch angle exceeds this value (degrees) """

    base_init_pos: list[int | float]
    r"""
    Initial base position [x, y, z] in meters

    minItems: 3
    maxItems: 3
    """

    base_init_quat: list[int | float]
    r"""
    Initial base orientation quaternion [w, x, y, z]

    minItems: 4
    maxItems: 4
    """

    episode_length: int | float
    r""" Maximum episode length in seconds """

    resampling_time: int | float
    r""" Time interval for resampling goals/actions in seconds """

    action_scale: int | float
    r""" Scaling multiplier applied to actions """

    simulate_action_latency: bool
    r""" Whether to simulate action latency """

    clip_actions: int | float
    r""" Action clipping magnitude """

    dt: int | float
    r""" Time duration for each simulation step in seconds, defaults to 1e-2 """



class ObsConfig(TypedDict, total=False):
    r"""
    Obs_Config.

    List of observation names, with optional scale values
    """

    num_obs: Required[int]
    r""" Required property """

    scales: Required[dict[str, int | float]]
    r""" Required property """



class RewardConfig(TypedDict, total=False):
    r""" Reward_Config. """

    tracking_sigma: int | float
    r""" Standard deviation of the tracking reward """

    rewards: Required[dict[str, int | float]]
    r"""
    List of rewards and their corresponding scales

    Required property
    """

    targets: Required[dict[str, int | float]]
    r"""
    List of target values

    Required property
    """



class RunnerConfig(TypedDict, total=False):
    r"""
    Runner_Config.

    Runner configuration. Use class_name to select between OnPolicyRunner and DistillationRunner.

    $comment: Produced with Claude Opus 4.5, use with caution.
    allOf:
      - if:
          properties:
            class_name:
              const: OnPolicyRunner
        then:
          properties:
            algorithm:
              $ref: runner.json#/$defs/ppo_algorithm
            policy:
              $ref: runner.json#/$defs/ppo_policy
      - if:
          properties:
            class_name:
              const: DistillationRunner
        then:
          properties:
            algorithm:
              $ref: runner.json#/$defs/distillation_algorithm
            policy:
              $ref: runner.json#/$defs/distillation_policy
    """

    class_name: Required["_RunnerConfigClassName"]
    r"""
    The runner class to use

    Required property
    """

    num_steps_per_env: Required[int]
    r"""
    Number of steps per environment per iteration

    Required property
    """

    max_iterations: Required[int]
    r"""
    Number of policy updates

    Required property
    """

    seed: int
    r""" Random seed for reproducibility """

    obs_groups: "_RunnerConfigObsGroups"
    r""" Maps observation groups to sets of observation types """

    save_interval: int
    r""" Check for potential saves every n iterations """

    experiment_name: Required[str]
    r"""
    Name of the experiment

    Required property
    """

    run_name: str
    r""" Name of the specific run """

    logger: "_RunnerConfigLogger"
    r""" Logging backend to use """

    neptune_project: str
    r""" Neptune project name (when logger=neptune) """

    wandb_project: str
    r""" Weights & Biases project name (when logger=wandb) """

    policy: Required["_RunnerFullStopJsonNumberSignDefsPolicy"]
    r"""
    Policy configuration (generic)

    Required property
    """

    algorithm: Required["_RunnerFullStopJsonNumberSignDefsAlgorithm"]
    r"""
    Algorithm configuration (generic)

    Required property
    """



_CommandConfigAdditionalproperties = list[int | float]
r"""
minLength: 2
maxLength: 2
"""



_RunnerConfigClassName = Literal['OnPolicyRunner'] | Literal['DistillationRunner']
r""" The runner class to use """
_RUNNERCONFIGCLASSNAME_ONPOLICYRUNNER: Literal['OnPolicyRunner'] = "OnPolicyRunner"
r"""The values for the 'The runner class to use' enum"""
_RUNNERCONFIGCLASSNAME_DISTILLATIONRUNNER: Literal['DistillationRunner'] = "DistillationRunner"
r"""The values for the 'The runner class to use' enum"""



_RunnerConfigLogger = Literal['tensorboard'] | Literal['neptune'] | Literal['wandb']
r""" Logging backend to use """
_RUNNERCONFIGLOGGER_TENSORBOARD: Literal['tensorboard'] = "tensorboard"
r"""The values for the 'Logging backend to use' enum"""
_RUNNERCONFIGLOGGER_NEPTUNE: Literal['neptune'] = "neptune"
r"""The values for the 'Logging backend to use' enum"""
_RUNNERCONFIGLOGGER_WANDB: Literal['wandb'] = "wandb"
r"""The values for the 'Logging backend to use' enum"""



class _RunnerConfigObsGroups(TypedDict, total=False):
    r""" Maps observation groups to sets of observation types """

    policy: list[str]
    r""" Observation sets for the policy/actor """

    critic: list[str]
    r""" Observation sets for the critic (PPO only) """

    teacher: list[str]
    r""" Observation sets for the teacher (Distillation only) """



class _RunnerFullStopJsonNumberSignDefsAlgorithm(TypedDict, total=False):
    r""" Algorithm configuration (generic) """

    class_name: Required["_RunnerFullStopJsonNumberSignDefsAlgorithmClassName"]
    r""" Required property """



_RunnerFullStopJsonNumberSignDefsAlgorithmClassName = Literal['PPO'] | Literal['Distillation']
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSALGORITHMCLASSNAME_PPO: Literal['PPO'] = "PPO"
r"""The values for the '_RunnerFullStopJsonNumberSignDefsAlgorithmClassName' enum"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSALGORITHMCLASSNAME_DISTILLATION: Literal['Distillation'] = "Distillation"
r"""The values for the '_RunnerFullStopJsonNumberSignDefsAlgorithmClassName' enum"""



class _RunnerFullStopJsonNumberSignDefsPolicy(TypedDict, total=False):
    r""" Policy configuration (generic) """

    class_name: Required["_RunnerFullStopJsonNumberSignDefsPolicyClassName"]
    r""" Required property """



_RunnerFullStopJsonNumberSignDefsPolicyClassName = Literal['ActorCritic'] | Literal['ActorCriticCNN'] | Literal['ActorCriticRecurrent'] | Literal['StudentTeacher'] | Literal['StudentTeacherRecurrent']
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSPOLICYCLASSNAME_ACTORCRITIC: Literal['ActorCritic'] = "ActorCritic"
r"""The values for the '_RunnerFullStopJsonNumberSignDefsPolicyClassName' enum"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSPOLICYCLASSNAME_ACTORCRITICCNN: Literal['ActorCriticCNN'] = "ActorCriticCNN"
r"""The values for the '_RunnerFullStopJsonNumberSignDefsPolicyClassName' enum"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSPOLICYCLASSNAME_ACTORCRITICRECURRENT: Literal['ActorCriticRecurrent'] = "ActorCriticRecurrent"
r"""The values for the '_RunnerFullStopJsonNumberSignDefsPolicyClassName' enum"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSPOLICYCLASSNAME_STUDENTTEACHER: Literal['StudentTeacher'] = "StudentTeacher"
r"""The values for the '_RunnerFullStopJsonNumberSignDefsPolicyClassName' enum"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSPOLICYCLASSNAME_STUDENTTEACHERRECURRENT: Literal['StudentTeacherRecurrent'] = "StudentTeacherRecurrent"
r"""The values for the '_RunnerFullStopJsonNumberSignDefsPolicyClassName' enum"""

