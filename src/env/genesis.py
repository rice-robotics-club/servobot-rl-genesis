import math
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Mapping, Sequence, TypeVar

import genesis as gs
import tensordict
import torch
from genesis.utils.geom import (
    inv_quat,
    transform_by_quat,
)
from rsl_rl import env

from src.config import CommandConfig, EnvConfig, ObsConfig, RewardConfig
from src.config.config import CurriculumConfig

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity


class GenesisEnv(env.VecEnv):
    """Provides base functionality of an RSL_RL VecEnv in Genesis."""

    def __init__(
        self,
        num_envs: int,
        env_cfg: EnvConfig,
        obs_cfg: ObsConfig,
        reward_cfg: RewardConfig,
        command_cfg: CommandConfig,
        curriculum_cfg: CurriculumConfig,
        headless: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__()

        self.cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.rewards = reward_cfg["rewards"]
        self.targets = reward_cfg["targets"]
        self.command_cfg = command_cfg
        self.curriculum_cfg = curriculum_cfg

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.obs_scales = obs_cfg["scales"]
        self.num_commands = len(self.command_cfg)
        self.num_actions = len(env_cfg["joints"])

        self.debug = debug

        # init default values
        self.device = gs.device if gs.device else torch.device("gpu")
        self.dt = get_or_default(env_cfg, "dt", 0.02)
        self.joint_names = sorted(env_cfg["joints"].keys())
        self.base_init_pos = get_or_default(env_cfg, "base_init_pos", [0.0, 0.0, 0.2])
        self.base_init_quat = get_or_default(
            env_cfg, "base_init_quat", [1.0, 0.0, 0.0, 0.0]
        )
        self.resampling_time = get_or_default(env_cfg, "resampling_time", 4.0)
        self.max_episode_length = math.ceil(
            get_or_default(env_cfg, "max_episode_length", 20) / self.dt
        )

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                # For this locomotion policy, there are usually no more than 20 collision pairs. Setting a low value
                # can save memory. Violating this condition will raise an exception.
                max_collision_pairs=100,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=int(1.0 / self.dt),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=not headless,
        )

        self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True,
            )
        )

        self.robot: RigidEntity = self.scene.add_entity(
            gs.morphs.URDF(
                file=self.cfg["urdf_path"],
                pos=self.base_init_pos,
                quat=self.base_init_quat,
            ),
        )  # pyright: ignore

        self.scene.build(n_envs=num_envs)

        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dofs_idx_local[0] for name in self.joint_names],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device
        )

        self.init_base_pos = torch.tensor(
            self.base_init_pos, dtype=gs.tc_float, device=gs.device
        )
        self.init_base_quat = torch.tensor(
            self.base_init_quat, dtype=gs.tc_float, device=gs.device
        )
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(
            [env_cfg["joints"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_qpos = torch.concatenate(
            (self.init_base_pos, self.init_base_quat, self.init_dof_pos)
        )
        self.init_projected_gravity: torch.Tensor = transform_by_quat(
            self.global_gravity, self.inv_base_init_quat
        )  # pyright: ignore

        self.base_lin_vel: torch.Tensor = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.base_ang_vel: torch.Tensor = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.projected_gravity = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.policy_buf = torch.empty(
            (self.num_envs, self.num_obs), dtype=gs.tc_float, device=gs.device
        )
        self.obs_dict = tensordict.TensorDict(
            {}, batch_size=[self.num_envs], device=gs.device
        )
        self.rew_buf = torch.empty(
            (self.num_envs,), dtype=gs.tc_float, device=gs.device
        )
        self.reset_buf = torch.ones(
            (self.num_envs,), dtype=gs.tc_bool, device=gs.device
        )
        self.episode_length_buf = torch.empty(
            (self.num_envs,), dtype=gs.tc_int, device=gs.device
        )
        self.commands = torch.empty(
            (self.num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device
        )
        self.commands_scale = torch.tensor(
            [self.obs_scales[name] for name in self.command_cfg],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.commands_limits: tuple[torch.Tensor, torch.Tensor] = tuple(
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(*self.command_cfg.values())
        )  # pyright: ignore
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.base_quat = torch.empty(
            (self.num_envs, 4), dtype=gs.tc_float, device=gs.device
        )
        self.base_euler: torch.Tensor = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.default_dof_pos = torch.tensor(
            [env_cfg["joints"][name] for name in self.joint_names],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()
        self.reward_functions: dict = {}
        self.episode_sums: dict[str, torch.Tensor] = {}

    @abstractmethod
    def step(
        self, actions: torch.Tensor, command: Sequence[float] | None = None
    ) -> tuple[tensordict.TensorDict, torch.Tensor, torch.Tensor, dict]:
        raise NotImplementedError

    @abstractmethod
    def update_observations(self) -> None:
        raise NotImplementedError

    def get_observations(self) -> tensordict.TensorDict:
        return self.obs_dict

    def init_reward_functions(self):
        for name in self.rewards:
            self.rewards[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), dtype=gs.tc_float, device=gs.device
            )

    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def reset_idx(self, envs_idx: torch.Tensor | None = None):
        # reset state
        self.robot.set_qpos(
            self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True
        )

        # reset buffers
        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_pos.copy_(self.init_base_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
            return
        else:
            torch.where(
                envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos
            )
            torch.where(
                envs_idx[:, None],
                self.init_base_quat,
                self.base_quat,
                out=self.base_quat,
            )
            torch.where(
                envs_idx[:, None],
                self.init_projected_gravity,
                self.projected_gravity,
                out=self.projected_gravity,
            )
            torch.where(
                envs_idx[:, None], self.init_dof_pos, self.dof_pos, out=self.dof_pos
            )
            torch.where(
                envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos
            )
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        # fill extras
        n_envs = envs_idx.sum()
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0)
            self.extras["episode"]["rew_" + key] = mean / self.max_episode_length
            value.masked_fill_(envs_idx, 0.0)

        # random sample command upon reset
        self._resample_commands(envs_idx)

    def reset(self) -> tensordict.TensorDict:
        self.reset_idx()
        self.update_observations()
        return self.obs_dict


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(
        size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device
    ) + lower


T = TypeVar("T")


def get_or_default(cfg: Mapping[str, Any], key: str, default: T) -> T:
    return cfg[key] if key in cfg else default
