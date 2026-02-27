import math
from typing import TYPE_CHECKING

import genesis as gs
import torch
from genesis.utils.geom import (
    inv_quat,
    quat_to_xyz,
    transform_by_quat,
    transform_quat_by_quat,
)
from tensordict import TensorDict

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(
        size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device
    ) + lower


class CatbotEnv:
    """
    Locomotion environment for Catbot, mirroring the structure of go2_env.py.

    Config key differences vs. go2_env.py:
      - env_cfg["joints"]        : dict of joint_name -> default_angle (replaces joint_names + default_joint_angles)
      - env_cfg["episode_length"]: episode duration in seconds       (replaces episode_length_s)
      - env_cfg["resampling_time"]: command resample period in seconds (replaces resampling_time_s)
      - obs_cfg["scales"]        : obs scale dict                    (replaces obs_scales)
          keys: lin_vel_x, lin_vel_y, ang_vel_z, dof_pos, dof_vel
      - reward_cfg["rewards"]    : reward scale dict                 (replaces reward_scales)
      - reward_cfg["targets"]    : target value dict (e.g. base_height)
      - command_cfg              : dict with keys lin_vel_x, lin_vel_y, ang_vel_z,
                                   each a [min, max] list              (replaces *_range keys + num_commands)

    termination_if_pitch/roll_greater_than values in env_cfg are interpreted in DEGREES,
    consistent with go2_env.py
    """

    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        headless=False,
        debug=False,
    ):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.joint_names = sorted(env_cfg["joints"].keys())
        self.num_actions = len(self.joint_names)
        self.num_commands = len(command_cfg)  # lin_vel_x, lin_vel_y, ang_vel_z
        self.device = gs.device

        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.dt = env_cfg.get("dt", 0.02)
        self.max_episode_length = math.ceil(
            env_cfg.get("episode_length", 20.0) / self.dt
        )

        self.cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["scales"]
        self.reward_scales = reward_cfg["rewards"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                max_collision_pairs=20,
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

        # add plane
        self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True,
            )
        )

        # add robot
        self.robot: RigidEntity = self.scene.add_entity(
            getattr(gs.morphs, env_cfg["robot_description_type"])(
                file=env_cfg["robot_description_path"],
                pos=env_cfg["base_init_pos"],
                quat=env_cfg["base_init_quat"],
            ),
        )  # pyright: ignore

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices (local DOF indices relative to the robot entity)
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dofs_idx_local[0] for name in self.joint_names],
            dtype=gs.tc_int,
            device=gs.device,
        )

        # zero out PD gains for non-motor (linkage) DOFs
        all_dof_idx = torch.arange(self.robot.n_dofs, device=gs.device)
        linkage_dof_idx = all_dof_idx[~torch.isin(all_dof_idx, self.motors_dof_idx)]
        self.robot.set_dofs_kp([0.0] * len(linkage_dof_idx), linkage_dof_idx)
        self.robot.set_dofs_kv([0.0] * len(linkage_dof_idx), linkage_dof_idx)

        self.robot.set_dofs_kp(
            [env_cfg.get("kp", 20.0)] * self.num_actions, self.motors_dof_idx
        )
        self.robot.set_dofs_kv(
            [env_cfg.get("kd", 0.5)] * self.num_actions, self.motors_dof_idx
        )

        self.default_dof_pos = torch.tensor(
            [env_cfg["joints"][name] for name in self.joint_names],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.robot.set_dofs_position(self.default_dof_pos, self.motors_dof_idx)

        # Define global gravity direction vector
        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device
        )

        # Initial state
        self.init_base_pos = torch.tensor(
            env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device
        )
        self.init_base_quat = torch.tensor(
            env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device
        )
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_qpos = self.robot.get_qpos()[0]
        self.init_projected_gravity = transform_by_quat(
            self.global_gravity, self.inv_base_init_quat
        )

        # initialize buffers
        self.base_lin_vel = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.base_ang_vel = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.projected_gravity = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.obs_dict = TensorDict(
            {
                "main": torch.empty(
                    (self.num_envs, self.num_obs), dtype=gs.tc_float, device=gs.device
                )
            },
            batch_size=(self.num_envs,),
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
            [
                self.obs_scales["lin_vel_x"],
                self.obs_scales["lin_vel_y"],
                self.obs_scales["ang_vel_z"],
            ],
            device=gs.device,
            dtype=gs.tc_float,
        )
        # commands_limits: list of [lower_tensor, upper_tensor], each shape (num_commands,)
        self.commands_limits = [
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                command_cfg["lin_vel_x"],
                command_cfg["lin_vel_y"],
                command_cfg["ang_vel_z"],
            )
        ]
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
        self.extras = dict()  # extra information for logging

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
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

    def step(self, actions):
        self.actions = torch.clip(
            actions,
            -self.cfg.get("clip_actions", 100.0),
            self.cfg.get("clip_actions", 100.0),
        )
        exec_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )
        target_dof_pos = (
            exec_actions * self.cfg.get("action_scale", 0.25) + self.default_dof_pos
        )
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # compute reward
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # resample commands
        self._resample_commands(
            self.episode_length_buf
            % int(self.cfg.get("resampling_time", 4.0) / self.dt)
            == 0
        )

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.cfg.get(
            "termination_if_pitch_greater_than", 10
        )
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.cfg.get(
            "termination_if_roll_greater_than", 10
        )

        # compute timeout
        self.extras["time_outs"] = (
            self.episode_length_buf > self.max_episode_length
        ).to(dtype=gs.tc_float)

        # reset environments that need it
        self._reset_idx(self.reset_buf)

        # update observations
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_dict

    def get_privileged_observations(self):
        return None

    def _reset_idx(self, envs_idx=None):
        # reset state
        self.robot.set_qpos(
            self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True
        )

        # reset buffers
        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.default_dof_pos)
            self.base_pos.copy_(self.init_base_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
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
                envs_idx[:, None], self.default_dof_pos, self.dof_pos, out=self.dof_pos
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
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        episode_length_s = self.cfg.get("episode_length", 20.0)
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
                value.zero_()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
                value.masked_fill_(envs_idx, 0.0)
            self.extras["episode"]["rew_" + key] = mean / episode_length_s

        # resample commands on reset
        self._resample_commands(envs_idx)

    def _update_observation(self):
        self.obs_dict["main"] = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel_z"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos)
                * self.obs_scales["dof_pos"],  # num_actions
                self.dof_vel * self.obs_scales["dof_vel"],  # num_actions
                self.actions,  # num_actions
            ),  # type: ignore
            dim=-1,
        )

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.obs_dict

    # ------------ reward functions ----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.reward_cfg.get("tracking_sigma", 0.25))

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg.get("tracking_sigma", 0.25))

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(
            self.base_pos[:, 2] - self.reward_cfg["targets"]["base_height"]
        )

    def _reward_roll_angle(self):
        # Penalize roll angle (radians, converted from degrees via base_euler)
        return torch.square(self.base_euler[:, 0] * (math.pi / 180.0))

    def _reward_pitch_angle(self):
        # Penalize pitch angle (radians, converted from degrees via base_euler)
        return torch.square(self.base_euler[:, 1] * (math.pi / 180.0))

    def _reward_energy(self):
        # Penalize energy consumption: |torque * velocity|
        # torque estimated from PD law: kp*(target - pos) + kd*(0 - vel)
        exec_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )
        target_dof_pos = (
            exec_actions * self.cfg.get("action_scale", 0.25) + self.default_dof_pos
        )
        torques = self.cfg.get("kp", 20.0) * (
            target_dof_pos - self.dof_pos
        ) + self.cfg.get("kd", 0.5) * (-self.dof_vel)
        return torch.sum(torch.abs(torques * self.dof_vel), dim=1)

    def _reward_survival(self):
        # Small constant reward scaled by commanded speed magnitude
        speed_magnitude = torch.norm(self.commands[:, :2], dim=1)
        return (
            torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float)
            * speed_magnitude
        )
