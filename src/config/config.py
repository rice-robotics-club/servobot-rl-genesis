from typing import Literal, Required, TypeAlias, TypedDict, Union


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

    robot_description_path: Required[str]
    r"""
    Path to the robot description file

    Required property
    """

    robot_description_type: Required["_EnvConfigRobotDescriptionType"]
    r"""
    Type of the robot file

    Required property
    """

    joints: Required[dict[str, int | float]]
    r"""
    Joint names and their default angles

    Required property
    """

    kp: Required[int | float]
    r"""
    Position proportional gain for joint PD controller

    Required property
    """

    kd: Required[int | float]
    r"""
    Velocity derivative gain for joint PD controller

    Required property
    """

    termination_if_roll_greater_than: int | float
    r""" Terminate episode if base roll angle exceeds this value (radians) """

    termination_if_pitch_greater_than: int | float
    r""" Terminate episode if base pitch angle exceeds this value (radians) """

    base_init_pos: Required[list[int | float]]
    r"""
    Initial base position [x, y, z] in meters

    minItems: 3
    maxItems: 3

    Required property
    """

    base_init_quat: Required[list[int | float]]
    r"""
    Initial base orientation quaternion [w, x, y, z]

    minItems: 4
    maxItems: 4

    Required property
    """

    episode_length: Required[int | float]
    r"""
    Maximum episode length in seconds

    Required property
    """

    resampling_time: Required[int | float]
    r"""
    Time interval for resampling goals/actions in seconds

    Required property
    """

    action_scale: Required[int | float]
    r"""
    Scaling multiplier applied to actions

    Required property
    """

    simulate_action_latency: bool
    r""" Whether to simulate action latency """

    clip_actions: Required[int | float]
    r"""
    Action clipping magnitude

    Required property
    """

    dt: Required[int | float]
    r"""
    Time duration for each simulation step in seconds, defaults to 1e-2

    Required property
    """



class ObsConfig(TypedDict, total=False):
    r"""
    Obs_Config.

    List of observation names, with optional scale values
    """

    num_obs: Required[dict[str, int]]
    r""" Required property """

    scales: Required[dict[str, int | float]]
    r""" Required property """



class RewardConfig(TypedDict, total=False):
    r""" Reward_Config. """

    tracking_sigma: Required[int | float]
    r"""
    Standard deviation of the tracking reward

    Required property
    """

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

    $comment: Updated to reflect the nested RSL-RL structure with discrete models (actor/critic/student/teacher).
    allOf:
      - if:
          properties:
            class_name:
              const: OnPolicyRunner
        then:
          properties:
            algorithm:
              $ref: runner.json#/$defs/ppo_algorithm
          required:
          - actor
          - critic
      - if:
          properties:
            class_name:
              const: DistillationRunner
        then:
          properties:
            algorithm:
              $ref: runner.json#/$defs/distillation_algorithm
          required:
          - student
          - teacher
      - if:
          properties:
            logger:
              const: wandb
        then:
          required:
          - wandb_project
      - if:
          properties:
            logger:
              const: neptune
        then:
          required:
          - neptune_project
    """

    class_name: Required["_RunnerConfigClassName"]
    r"""
    The runner class to use

    Required property
    """

    num_steps_per_env: Required[int]
    r"""
    Number of environment steps collected per iteration.

    Required property
    """

    obs_groups: Required["_RunnerConfigObsGroups"]
    r"""
    Mapping from observation sets to observation groups coming from the environment.

    Required property
    """

    save_interval: Required[int]
    r"""
    Number of iterations between checkpoints.

    Required property
    """

    logger: "_RunnerConfigLogger"
    r"""
    Logging service to use.

    default: tensorboard
    """

    wandb_project: str
    r""" W&B project name used by the W&B writer. """

    neptune_project: str
    r""" Neptune project name used by the Neptune writer. """

    run_name: str
    r""" Optional run label shown in the console output. """

    check_for_nan: bool
    r"""
    Whether to check for NaN values coming from the environment.

    default: True
    """

    algorithm: Required["_RunnerFullStopJsonNumberSignDefsAlgorithm"]
    r"""
    Algorithm configuration (generic)

    Required property
    """

    actor: "_RunnerFullStopJsonNumberSignDefsModel"
    r"""
    Model configuration (actor, critic, student, or teacher).

    Aggregation type: oneOf
    """

    critic: "_RunnerFullStopJsonNumberSignDefsModel"
    r"""
    Model configuration (actor, critic, student, or teacher).

    Aggregation type: oneOf
    """

    student: "_RunnerFullStopJsonNumberSignDefsModel"
    r"""
    Model configuration (actor, critic, student, or teacher).

    Aggregation type: oneOf
    """

    teacher: "_RunnerFullStopJsonNumberSignDefsModel"
    r"""
    Model configuration (actor, critic, student, or teacher).

    Aggregation type: oneOf
    """



_CommandConfigAdditionalproperties = list[int | float]
r"""
minLength: 2
maxLength: 2
"""



_EnvConfigRobotDescriptionType = Literal['URDF'] | Literal['MJCF']
r""" Type of the robot file """
_ENVCONFIGROBOTDESCRIPTIONTYPE_URDF: Literal['URDF'] = "URDF"
r"""The values for the 'Type of the robot file' enum"""
_ENVCONFIGROBOTDESCRIPTIONTYPE_MJCF: Literal['MJCF'] = "MJCF"
r"""The values for the 'Type of the robot file' enum"""



_RUNNER_CONFIG_CHECK_FOR_NAN_DEFAULT = True
r""" Default value of the field path 'Runner_Config check_for_nan' """



_RUNNER_CONFIG_LOGGER_DEFAULT = 'tensorboard'
r""" Default value of the field path 'Runner_Config logger' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_CONFIG_ACTIVATION_DEFAULT = 'elu'
r""" Default value of the field path 'runner.json# $defs cnn_config activation' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_CONFIG_DILATION_DEFAULT = 1
r""" Default value of the field path 'runner.json# $defs cnn_config dilation' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_CONFIG_DILATION_ONEOF0_DEFAULT = 1
r""" Default value of the field path 'runner.json# $defs cnn_config dilation oneof0' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_CONFIG_DILATION_ONEOF1_DEFAULT = 1
r""" Default value of the field path 'runner.json# $defs cnn_config dilation oneof1' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_CONFIG_FLATTEN_DEFAULT = True
r""" Default value of the field path 'runner.json# $defs cnn_config flatten' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_CONFIG_GLOBAL_POOL_DEFAULT = 'none'
r""" Default value of the field path 'runner.json# $defs cnn_config global_pool' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_CONFIG_MAX_POOL_DEFAULT = False
r""" Default value of the field path 'runner.json# $defs cnn_config max_pool' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_CONFIG_MAX_POOL_ONEOF0_DEFAULT = False
r""" Default value of the field path 'runner.json# $defs cnn_config max_pool oneof0' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_CONFIG_MAX_POOL_ONEOF1_DEFAULT = False
r""" Default value of the field path 'runner.json# $defs cnn_config max_pool oneof1' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_CONFIG_NORM_DEFAULT = 'none'
r""" Default value of the field path 'runner.json# $defs cnn_config norm' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_CONFIG_NORM_ONEOF0_DEFAULT = 'none'
r""" Default value of the field path 'runner.json# $defs cnn_config norm oneof0' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_CONFIG_NORM_ONEOF1_DEFAULT = 'none'
r""" Default value of the field path 'runner.json# $defs cnn_config norm oneof1' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_CONFIG_PADDING_DEFAULT = 'none'
r""" Default value of the field path 'runner.json# $defs cnn_config padding' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_CONFIG_STRIDE_DEFAULT = 1
r""" Default value of the field path 'runner.json# $defs cnn_config stride' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_CONFIG_STRIDE_ONEOF0_DEFAULT = 1
r""" Default value of the field path 'runner.json# $defs cnn_config stride oneof0' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_CONFIG_STRIDE_ONEOF1_DEFAULT = 1
r""" Default value of the field path 'runner.json# $defs cnn_config stride oneof1' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_MODEL_ACTIVATION_DEFAULT = 'elu'
r""" Default value of the field path 'runner.json# $defs cnn_model activation' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_MODEL_CNN_CFG_DEFAULT = None
r""" Default value of the field path 'runner.json# $defs cnn_model cnn_cfg' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_MODEL_CNN_CFG_ONEOF0_DEFAULT = None
r""" Default value of the field path 'runner.json# $defs cnn_model cnn_cfg oneof0' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_MODEL_CNN_CFG_ONEOF1_DEFAULT = None
r""" Default value of the field path 'runner.json# $defs cnn_model cnn_cfg oneof1' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_MODEL_CNN_CFG_ONEOF2_DEFAULT = None
r""" Default value of the field path 'runner.json# $defs cnn_model cnn_cfg oneof2' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_MODEL_DISTRIBUTION_CFG_DEFAULT = None
r""" Default value of the field path 'runner.json# $defs cnn_model distribution_cfg' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_MODEL_DISTRIBUTION_CFG_ONEOF0_DEFAULT = None
r""" Default value of the field path 'runner.json# $defs cnn_model distribution_cfg oneof0' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_MODEL_DISTRIBUTION_CFG_ONEOF1_DEFAULT = None
r""" Default value of the field path 'runner.json# $defs cnn_model distribution_cfg oneof1' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_MODEL_HIDDEN_DIMS_DEFAULT = [256, 256, 256]
r""" Default value of the field path 'runner.json# $defs cnn_model hidden_dims' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_CNN_MODEL_OBS_NORMALIZATION_DEFAULT = False
r""" Default value of the field path 'runner.json# $defs cnn_model obs_normalization' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_DISTRIBUTION_CFG_INIT_STD_DEFAULT = 1.0
r""" Default value of the field path 'runner.json# $defs distribution_cfg init_std' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_DISTRIBUTION_CFG_STD_TYPE_DEFAULT = 'scalar'
r""" Default value of the field path 'runner.json# $defs distribution_cfg std_type' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_MLP_MODEL_ACTIVATION_DEFAULT = 'elu'
r""" Default value of the field path 'runner.json# $defs mlp_model activation' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_MLP_MODEL_DISTRIBUTION_CFG_DEFAULT = None
r""" Default value of the field path 'runner.json# $defs mlp_model distribution_cfg' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_MLP_MODEL_DISTRIBUTION_CFG_ONEOF0_DEFAULT = None
r""" Default value of the field path 'runner.json# $defs mlp_model distribution_cfg oneof0' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_MLP_MODEL_DISTRIBUTION_CFG_ONEOF1_DEFAULT = None
r""" Default value of the field path 'runner.json# $defs mlp_model distribution_cfg oneof1' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_MLP_MODEL_HIDDEN_DIMS_DEFAULT = [256, 256, 256]
r""" Default value of the field path 'runner.json# $defs mlp_model hidden_dims' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_MLP_MODEL_OBS_NORMALIZATION_DEFAULT = False
r""" Default value of the field path 'runner.json# $defs mlp_model obs_normalization' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_RNN_MODEL_ACTIVATION_DEFAULT = 'elu'
r""" Default value of the field path 'runner.json# $defs rnn_model activation' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_RNN_MODEL_DISTRIBUTION_CFG_DEFAULT = None
r""" Default value of the field path 'runner.json# $defs rnn_model distribution_cfg' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_RNN_MODEL_DISTRIBUTION_CFG_ONEOF0_DEFAULT = None
r""" Default value of the field path 'runner.json# $defs rnn_model distribution_cfg oneof0' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_RNN_MODEL_DISTRIBUTION_CFG_ONEOF1_DEFAULT = None
r""" Default value of the field path 'runner.json# $defs rnn_model distribution_cfg oneof1' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_RNN_MODEL_HIDDEN_DIMS_DEFAULT = [256, 256, 256]
r""" Default value of the field path 'runner.json# $defs rnn_model hidden_dims' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_RNN_MODEL_OBS_NORMALIZATION_DEFAULT = False
r""" Default value of the field path 'runner.json# $defs rnn_model obs_normalization' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_RNN_MODEL_RNN_HIDDEN_DIM_DEFAULT = 256
r""" Default value of the field path 'runner.json# $defs rnn_model rnn_hidden_dim' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_RNN_MODEL_RNN_NUM_LAYERS_DEFAULT = 1
r""" Default value of the field path 'runner.json# $defs rnn_model rnn_num_layers' """



_RUNNER_FULL_STOP_JSON_NUMBER_SIGN___DEFS_RNN_MODEL_RNN_TYPE_DEFAULT = 'lstm'
r""" Default value of the field path 'runner.json# $defs rnn_model rnn_type' """



_RunnerConfigClassName = Literal['OnPolicyRunner'] | Literal['DistillationRunner']
r""" The runner class to use """
_RUNNERCONFIGCLASSNAME_ONPOLICYRUNNER: Literal['OnPolicyRunner'] = "OnPolicyRunner"
r"""The values for the 'The runner class to use' enum"""
_RUNNERCONFIGCLASSNAME_DISTILLATIONRUNNER: Literal['DistillationRunner'] = "DistillationRunner"
r"""The values for the 'The runner class to use' enum"""



_RunnerConfigLogger = Literal['tensorboard'] | Literal['wandb'] | Literal['neptune']
r"""
Logging service to use.

default: tensorboard
"""
_RUNNERCONFIGLOGGER_TENSORBOARD: Literal['tensorboard'] = "tensorboard"
r"""The values for the 'Logging service to use' enum"""
_RUNNERCONFIGLOGGER_WANDB: Literal['wandb'] = "wandb"
r"""The values for the 'Logging service to use' enum"""
_RUNNERCONFIGLOGGER_NEPTUNE: Literal['neptune'] = "neptune"
r"""The values for the 'Logging service to use' enum"""



class _RunnerConfigObsGroups(TypedDict, total=False):
    r""" Mapping from observation sets to observation groups coming from the environment. """

    actor: list[str]
    r""" Observations used as input to the actor model. """

    critic: list[str]
    r""" Observations used as input to the critic model. """

    student: list[str]
    r""" Observations used as input to the student model. """

    teacher: list[str]
    r""" Observations used as input to the teacher model. """

    rnd_state: list[str]
    r""" Observations used as input to the RND extension. """



class _RunnerFullStopJsonNumberSignDefsAlgorithm(TypedDict, total=False):
    r""" Algorithm configuration (generic) """

    class_name: Required["_RunnerFullStopJsonNumberSignDefsAlgorithmClassName"]
    r""" Required property """



_RunnerFullStopJsonNumberSignDefsAlgorithmClassName = Literal['PPO'] | Literal['Distillation']
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSALGORITHMCLASSNAME_PPO: Literal['PPO'] = "PPO"
r"""The values for the '_RunnerFullStopJsonNumberSignDefsAlgorithmClassName' enum"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSALGORITHMCLASSNAME_DISTILLATION: Literal['Distillation'] = "Distillation"
r"""The values for the '_RunnerFullStopJsonNumberSignDefsAlgorithmClassName' enum"""



class _RunnerFullStopJsonNumberSignDefsCnnConfig(TypedDict, total=False):
    r""" CNN network configuration. """

    output_channels: Required[list[int]]
    r"""
    Output channels for each convolutional layer.

    Required property
    """

    kernel_size: Required[int | list[int]]
    r"""
    Kernel size for each convolutional layer or a single kernel size for all layers.

    Aggregation type: oneOf

    Required property
    """

    stride: Union["_RunnerFullStopJsonNumberSignDefsCnnConfigStrideOneof0", "_RunnerFullStopJsonNumberSignDefsCnnConfigStrideOneof1"]
    r"""
    Stride for each convolutional layer or a single stride for all layers.

    default: 1

    Aggregation type: oneOf
    """

    dilation: Union["_RunnerFullStopJsonNumberSignDefsCnnConfigDilationOneof0", "_RunnerFullStopJsonNumberSignDefsCnnConfigDilationOneof1"]
    r"""
    Dilation for each convolutional layer or a single dilation for all layers.

    default: 1

    Aggregation type: oneOf
    """

    padding: "_RunnerFullStopJsonNumberSignDefsCnnConfigPadding"
    r"""
    Padding type to use.

    default: none
    """

    norm: Union["_RunnerFullStopJsonNumberSignDefsCnnConfigNormOneof0", "_RunnerFullStopJsonNumberSignDefsCnnConfigNormOneof1"]
    r"""
    Normalization type for each convolutional layer or a single normalization type for all layers.

    default: none

    Aggregation type: oneOf
    """

    activation: str
    r"""
    Activation function to use.

    default: elu
    """

    max_pool: Union["_RunnerFullStopJsonNumberSignDefsCnnConfigMaxPoolOneof0", "_RunnerFullStopJsonNumberSignDefsCnnConfigMaxPoolOneof1"]
    r"""
    Whether to apply max pooling after each convolutional layer or a single boolean for all layers.

    default: False

    Aggregation type: oneOf
    """

    global_pool: "_RunnerFullStopJsonNumberSignDefsCnnConfigGlobalPool"
    r"""
    Global pooling type to apply at the end.

    default: none
    """

    flatten: bool
    r"""
    Whether to flatten the output tensor.

    default: True
    """



_RunnerFullStopJsonNumberSignDefsCnnConfigDilationOneof0 = int
r""" default: 1 """



_RunnerFullStopJsonNumberSignDefsCnnConfigDilationOneof1 = list[int]
r""" default: 1 """



_RunnerFullStopJsonNumberSignDefsCnnConfigGlobalPool = Literal['none'] | Literal['max'] | Literal['avg']
r"""
Global pooling type to apply at the end.

default: none
"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSCNNCONFIGGLOBALPOOL_NONE: Literal['none'] = "none"
r"""The values for the 'Global pooling type to apply at the end' enum"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSCNNCONFIGGLOBALPOOL_MAX: Literal['max'] = "max"
r"""The values for the 'Global pooling type to apply at the end' enum"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSCNNCONFIGGLOBALPOOL_AVG: Literal['avg'] = "avg"
r"""The values for the 'Global pooling type to apply at the end' enum"""



_RunnerFullStopJsonNumberSignDefsCnnConfigMaxPoolOneof0 = bool
r""" default: False """



_RunnerFullStopJsonNumberSignDefsCnnConfigMaxPoolOneof1 = list[bool]
r""" default: False """



_RunnerFullStopJsonNumberSignDefsCnnConfigNormOneof0 = Literal['none'] | Literal['batch'] | Literal['layer']
r""" default: none """
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSCNNCONFIGNORMONEOF0_NONE: Literal['none'] = "none"
r"""The values for the 'default: none' enum"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSCNNCONFIGNORMONEOF0_BATCH: Literal['batch'] = "batch"
r"""The values for the 'default: none' enum"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSCNNCONFIGNORMONEOF0_LAYER: Literal['layer'] = "layer"
r"""The values for the 'default: none' enum"""



_RunnerFullStopJsonNumberSignDefsCnnConfigNormOneof1 = list["_RunnerFullStopJsonNumberSignDefsCnnConfigNormOneof1Item"]
r""" default: none """



_RunnerFullStopJsonNumberSignDefsCnnConfigNormOneof1Item = Literal['none'] | Literal['batch'] | Literal['layer']
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSCNNCONFIGNORMONEOF1ITEM_NONE: Literal['none'] = "none"
r"""The values for the '_RunnerFullStopJsonNumberSignDefsCnnConfigNormOneof1Item' enum"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSCNNCONFIGNORMONEOF1ITEM_BATCH: Literal['batch'] = "batch"
r"""The values for the '_RunnerFullStopJsonNumberSignDefsCnnConfigNormOneof1Item' enum"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSCNNCONFIGNORMONEOF1ITEM_LAYER: Literal['layer'] = "layer"
r"""The values for the '_RunnerFullStopJsonNumberSignDefsCnnConfigNormOneof1Item' enum"""



_RunnerFullStopJsonNumberSignDefsCnnConfigPadding = Literal['none'] | Literal['zeros'] | Literal['reflect'] | Literal['replicate'] | Literal['circular']
r"""
Padding type to use.

default: none
"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSCNNCONFIGPADDING_NONE: Literal['none'] = "none"
r"""The values for the 'Padding type to use' enum"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSCNNCONFIGPADDING_ZEROS: Literal['zeros'] = "zeros"
r"""The values for the 'Padding type to use' enum"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSCNNCONFIGPADDING_REFLECT: Literal['reflect'] = "reflect"
r"""The values for the 'Padding type to use' enum"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSCNNCONFIGPADDING_REPLICATE: Literal['replicate'] = "replicate"
r"""The values for the 'Padding type to use' enum"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSCNNCONFIGPADDING_CIRCULAR: Literal['circular'] = "circular"
r"""The values for the 'Padding type to use' enum"""



_RunnerFullStopJsonNumberSignDefsCnnConfigStrideOneof0 = int
r""" default: 1 """



_RunnerFullStopJsonNumberSignDefsCnnConfigStrideOneof1 = list[int]
r""" default: 1 """



class _RunnerFullStopJsonNumberSignDefsCnnModel(TypedDict, total=False):
    r""" CNN Model configuration. """

    class_name: Required[Literal['CNNModel']]
    r""" Required property """

    hidden_dims: list[int]
    r"""
    Hidden dimensions of the MLP.

    default:
      - 256
      - 256
      - 256
    """

    activation: str
    r"""
    Activation function of the MLP.

    default: elu
    """

    obs_normalization: bool
    r"""
    Whether to normalize the observations before passing them to the MLP.

    default: False
    """

    distribution_cfg: Union["_RunnerFullStopJsonNumberSignDefsCnnModelDistributionCfgOneof0", "_RunnerFullStopJsonNumberSignDefsDistributionCfg"]
    r"""
    Optional output distribution configuration.

    default: None

    Aggregation type: oneOf
    """

    cnn_cfg: Union["_RunnerFullStopJsonNumberSignDefsCnnModelCnnCfgOneof0", "_RunnerFullStopJsonNumberSignDefsCnnConfig", "_RunnerFullStopJsonNumberSignDefsCnnModelCnnCfgOneof2"]
    r"""
    Configuration of the CNN encoder(s).

    default: None

    Aggregation type: oneOf
    """



_RunnerFullStopJsonNumberSignDefsCnnModelCnnCfgOneof0: TypeAlias = None
r""" default: None """



_RunnerFullStopJsonNumberSignDefsCnnModelCnnCfgOneof2 = dict[str, "_RunnerFullStopJsonNumberSignDefsCnnConfig"]
r"""
Per-observation-group CNN configurations

default: None
"""



_RunnerFullStopJsonNumberSignDefsCnnModelDistributionCfgOneof0: TypeAlias = None
r""" default: None """



class _RunnerFullStopJsonNumberSignDefsDistributionCfg(TypedDict, total=False):
    r""" Distribution configuration for stochastic outputs. """

    class_name: Required["_RunnerFullStopJsonNumberSignDefsDistributionCfgClassName"]
    r"""
    Distribution class name.

    Required property
    """

    init_std: int | float
    r"""
    Initial standard deviation.

    default: 1.0
    """

    std_type: "_RunnerFullStopJsonNumberSignDefsDistributionCfgStdType"
    r"""
    Parameterization of the standard deviation.

    default: scalar
    """



_RunnerFullStopJsonNumberSignDefsDistributionCfgClassName = Literal['GaussianDistribution'] | Literal['HeteroscedasticGaussianDistribution']
r""" Distribution class name. """
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSDISTRIBUTIONCFGCLASSNAME_GAUSSIANDISTRIBUTION: Literal['GaussianDistribution'] = "GaussianDistribution"
r"""The values for the 'Distribution class name' enum"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSDISTRIBUTIONCFGCLASSNAME_HETEROSCEDASTICGAUSSIANDISTRIBUTION: Literal['HeteroscedasticGaussianDistribution'] = "HeteroscedasticGaussianDistribution"
r"""The values for the 'Distribution class name' enum"""



_RunnerFullStopJsonNumberSignDefsDistributionCfgStdType = Literal['scalar'] | Literal['log']
r"""
Parameterization of the standard deviation.

default: scalar
"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSDISTRIBUTIONCFGSTDTYPE_SCALAR: Literal['scalar'] = "scalar"
r"""The values for the 'Parameterization of the standard deviation' enum"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSDISTRIBUTIONCFGSTDTYPE_LOG: Literal['log'] = "log"
r"""The values for the 'Parameterization of the standard deviation' enum"""



class _RunnerFullStopJsonNumberSignDefsMlpModel(TypedDict, total=False):
    r""" MLP Model configuration. """

    class_name: Required[Literal['MLPModel']]
    r""" Required property """

    hidden_dims: list[int]
    r"""
    Hidden dimensions of the MLP.

    default:
      - 256
      - 256
      - 256
    """

    activation: str
    r"""
    Activation function of the MLP.

    default: elu
    """

    obs_normalization: bool
    r"""
    Whether to normalize the observations before passing them to the MLP.

    default: False
    """

    distribution_cfg: Union["_RunnerFullStopJsonNumberSignDefsMlpModelDistributionCfgOneof0", "_RunnerFullStopJsonNumberSignDefsDistributionCfg"]
    r"""
    Optional output distribution configuration.

    default: None

    Aggregation type: oneOf
    """



_RunnerFullStopJsonNumberSignDefsMlpModelDistributionCfgOneof0: TypeAlias = None
r""" default: None """



_RunnerFullStopJsonNumberSignDefsModel = Union["_RunnerFullStopJsonNumberSignDefsMlpModel", "_RunnerFullStopJsonNumberSignDefsRnnModel", "_RunnerFullStopJsonNumberSignDefsCnnModel"]
r"""
Model configuration (actor, critic, student, or teacher).

Aggregation type: oneOf
"""



class _RunnerFullStopJsonNumberSignDefsRnnModel(TypedDict, total=False):
    r""" RNN Model configuration. """

    class_name: Required[Literal['RNNModel']]
    r""" Required property """

    hidden_dims: list[int]
    r"""
    Hidden dimensions of the MLP.

    default:
      - 256
      - 256
      - 256
    """

    activation: str
    r"""
    Activation function of the MLP.

    default: elu
    """

    obs_normalization: bool
    r"""
    Whether to normalize the observations before passing them to the MLP.

    default: False
    """

    distribution_cfg: Union["_RunnerFullStopJsonNumberSignDefsRnnModelDistributionCfgOneof0", "_RunnerFullStopJsonNumberSignDefsDistributionCfg"]
    r"""
    Optional output distribution configuration.

    default: None

    Aggregation type: oneOf
    """

    rnn_type: "_RunnerFullStopJsonNumberSignDefsRnnModelRnnType"
    r"""
    Type of RNN network.

    default: lstm
    """

    rnn_hidden_dim: int
    r"""
    Hidden dimension of the RNN.

    default: 256
    """

    rnn_num_layers: int
    r"""
    Number of RNN layers.

    default: 1
    """



_RunnerFullStopJsonNumberSignDefsRnnModelDistributionCfgOneof0: TypeAlias = None
r""" default: None """



_RunnerFullStopJsonNumberSignDefsRnnModelRnnType = Literal['lstm'] | Literal['gru']
r"""
Type of RNN network.

default: lstm
"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSRNNMODELRNNTYPE_LSTM: Literal['lstm'] = "lstm"
r"""The values for the 'Type of RNN network' enum"""
_RUNNERFULLSTOPJSONNUMBERSIGNDEFSRNNMODELRNNTYPE_GRU: Literal['gru'] = "gru"
r"""The values for the 'Type of RNN network' enum"""

