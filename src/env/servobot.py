from typing import Sequence

import genesis as gs
import tensordict
import torch
from genesis.utils.geom import (
    inv_quat,
    quat_to_xyz,
    transform_by_quat,
    transform_quat_by_quat,
)

from src.env.genesis import GenesisEnv, get_or_default


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class ServobotEnv(GenesisEnv):
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        headless: bool = False,
        debug: bool = False,
    ):
        super().__init__(
            num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, headless, debug
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

        self.robot.set_dofs_kp([self.kp] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.kv] * self.num_actions, self.motors_dof_idx)

        self.policy_buf = torch.zeros(self.num_envs, self.num_obs, device=gs.device)
        self.obs_dict = tensordict.TensorDict(
            {"policy": self.policy_buf}, batch_size=[self.num_envs], device=gs.device
        )

        self.init_reward_functions()

    def update_observations(self):
        self.policy_buf = torch.concatenate(
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

        self.obs_dict["policy"] = self.policy_buf

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

        if self.debug:
            # print("base_pos", self.base_pos)
            # print("base_quat", self.base_quat)
            print("base_euler_pitch", self.base_euler[:, 1])
            # print("base_lin_vel", self.base_lin_vel)
            # print("base_ang_vel", self.base_ang_vel)
            # print("projected_gravity", self.projected_gravity)
            # print("dof_pos", self.dof_pos)
            # print("dof_vel", self.dof_vel)

        # compute reward
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.rewards[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        if command:
            # set command to input [-1.0, 1.0], scaled by command ranges
            # rearranged this to match the physical orientation of servobot
            self.commands[:, 0] = (
                command[0]
                * (self.command_cfg["lin_vel_x"][1] - self.command_cfg["lin_vel_x"][0])
            ) + self.command_cfg["lin_vel_x"][0]
            self.commands[:, 1] = (
                command[1]
                * (self.command_cfg["lin_vel_y"][1] - self.command_cfg["lin_vel_y"][0])
            ) + self.command_cfg["lin_vel_y"][0]
            self.commands[:, 2] = (
                command[2]
                * (self.command_cfg["ang_vel_z"][1] - self.command_cfg["ang_vel_z"][0])
            ) + self.command_cfg["ang_vel_z"][0]
            print(self.commands)
        else:
            # resample commands
            self._resample_commands(
                self.episode_length_buf % int(self.resampling_time / self.dt) == 0
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
