import math
from typing import TYPE_CHECKING, Any, Mapping, Sequence, TypeVar

import genesis as gs
import tensordict
import torch
from genesis.utils.geom import (
    inv_quat,
    quat_to_xyz,
    transform_by_quat,
    transform_quat_by_quat,
)
from rsl_rl import env

from src.config import CommandConfig, EnvConfig, ObsConfig, RewardConfig

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity


class ServobotStudentEnv(env.VecEnv):
    def __init__(
        self,
        num_envs: int,
        env_cfg: EnvConfig,
        obs_cfg: ObsConfig,
        reward_cfg: RewardConfig,
        command_cfg: CommandConfig,
        headless: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__()

        self.cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.rewards = reward_cfg["rewards"]
        self.targets = reward_cfg["targets"]
        self.command_cfg = command_cfg

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.obs_scales = obs_cfg["scales"]
        self.num_commands = len(self.command_cfg)
        self.num_actions = len(env_cfg["joints"])

        self.debug = debug
        self.headless = headless

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
        self.kp = get_or_default(env_cfg, "kp", 20.0)
        self.kv = get_or_default(env_cfg, "kv", 0.5)
        self.tracking_sigma = get_or_default(reward_cfg, "tracking_sigma", 0.25)
        self.clip_actions = get_or_default(env_cfg, "clip_actions", 100.0)
        self.simulate_action_latency = get_or_default(
            env_cfg, "simulate_action_latency", False
        )
        self.action_scale = get_or_default(env_cfg, "action_scale", 0.25)
        self.termination_if_pitch_greater_than = get_or_default(
            env_cfg, "termination_if_pitch_greater_than", 45
        )
        self.termination_if_roll_greater_than = get_or_default(
            env_cfg, "termination_if_roll_greater_than", 45
        )

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
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

        self.imu = self.scene.add_sensor(
            gs.sensors.IMU(
                entity_idx=self.robot.idx,
                link_idx_local=self.robot.base_link_idx,
            )
        )

        self.scene.build(n_envs=num_envs)

        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dofs_idx_local[0] for name in self.joint_names],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        self.robot.set_dofs_kp([self.kp] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.kv] * self.num_actions, self.motors_dof_idx)

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
        self.policy_buf = torch.zeros(self.num_envs, self.num_obs, device=gs.device)
        self.obs_dict = tensordict.TensorDict(
            {"policy": self.policy_buf}, batch_size=[self.num_envs], device=gs.device
        )

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()
        self.reward_functions: dict = {}
        self.episode_sums: dict[str, torch.Tensor] = {}

        self.init_reward_functions()

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

    def update_observations(self):
        self.obs_dict["policy"] = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel_z"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos)
                * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ),
            dim=-1,
        )

    def step(self, actions: torch.Tensor, command: Sequence[float] | None = None):
        self.actions = torch.clip(actions, -self.clip_actions, self.clip_actions)
        exec_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )
        target_dof_pos = exec_actions * self.action_scale + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)

        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=False,
        )  # pyright: ignore
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)  # pyright: ignore
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)  # pyright: ignore
        self.projected_gravity: torch.Tensor = transform_by_quat(
            self.global_gravity, inv_base_quat
        )  # pyright: ignore
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # compute reward
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.rewards[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        if command:
            self.commands[:, 0] = (
                (command[0] * 0.5 + 0.5)
                * (self.command_cfg["lin_vel_x"][1] - self.command_cfg["lin_vel_x"][0])
            ) + self.command_cfg["lin_vel_x"][0]
            self.commands[:, 1] = (
                (command[1] * 0.5 + 0.5)
                * (self.command_cfg["lin_vel_y"][1] - self.command_cfg["lin_vel_y"][0])
            ) + self.command_cfg["lin_vel_y"][0]
            self.commands[:, 2] = (
                (command[2] * 0.5 + 0.5)
                * (self.command_cfg["ang_vel_z"][1] - self.command_cfg["ang_vel_z"][0])
            ) + self.command_cfg["ang_vel_z"][0]
        else:
            # resample commands
            self._resample_commands(
                self.episode_length_buf % int(self.resampling_time / self.dt) == 0
            )

        # visualize commanded and actual velocity
        self.scene.clear_debug_objects()

        cmd_vec = torch.zeros(3)
        cmd_vec[:2] = self.commands[0, :2]
        cmd_vec[2] = 0.0
        cmd_vec: torch.Tensor = transform_by_quat(cmd_vec, self.base_quat[0, :])  # type: ignore

        self.cmd_debug_arrow = self.scene.draw_debug_arrow(
            self.base_pos[0, :].cpu(),
            cmd_vec.cpu(),
            color=(0, 0, 1, 0.5),
        )
        self.vel_debug_arrow = self.scene.draw_debug_arrow(
            self.base_pos[0, :].cpu(),
            self.base_lin_vel[0, :].cpu(),
            color=(1, 0, 0, 0.5),
        )

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length

        self.reset_buf |= (
            torch.abs(self.base_euler[:, 1]) > self.termination_if_pitch_greater_than
        )
        self.reset_buf |= (
            torch.abs(self.base_euler[:, 0]) > self.termination_if_roll_greater_than
        )

        self.extras["time_outs"] = (
            self.episode_length_buf > self.max_episode_length
        ).to(dtype=gs.tc_float)

        self.reset_idx(self.reset_buf)

        # update observations
        self.update_observations()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.tracking_sigma)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_roll_angle(self):
        # Penalize roll angle
        return torch.square(self.base_euler[:, 0])

    def _reward_pitch_angle(self):
        # Penalize pitch angle
        return torch.square(self.base_euler[:, 1])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.targets["base_height"])

    def _reward_energy(self):
        # Penalize energy consumption (torque * velocity)
        # For PD control: torque = kp * (target - current) + kv * (0 - vel)
        # This is inspired by this paper: https://arxiv.org/pdf/2111.01674
        # Should help the robot develop more efficient and 'natural' gaits over time

        # Calculate target positions from actions
        exec_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )
        target_dof_pos = exec_actions * self.action_scale + self.default_dof_pos

        # Calculate PD torques
        pos_error = target_dof_pos - self.dof_pos
        vel_error = -self.dof_vel  # target velocity is 0
        # These are actually different for each env if domain randomization is on
        torques = self.kp * pos_error + self.kv * vel_error

        # Energy = |torque * velocity|
        return torch.sum(torch.abs(torques * self.dof_vel), dim=1)

    def _reward_survival(self):
        # Small constant reward for survival
        # Scales with target velocity magnitude, which is inspired by https://arxiv.org/pdf/2111.01674
        speed_magnitude = torch.norm(self.commands[:, :2], dim=1)
        return (
            torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float)
            * speed_magnitude
        )


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(
        size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device
    ) + lower


T = TypeVar("T")


def get_or_default(cfg: Mapping[str, Any], key: str, default: T) -> T:
    return cfg[key] if key in cfg else default
