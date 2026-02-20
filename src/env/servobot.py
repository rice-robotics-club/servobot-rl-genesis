from typing import Sequence

import genesis as gs
import torch
from genesis.utils.geom import (
    inv_quat,
    quat_to_xyz,
    transform_by_quat,
    transform_quat_by_quat,
)

from src.config import Config
from src.env.base import BaseEnv


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class ServobotEnv(BaseEnv):
    def __init__(self, num_envs: int, cfg: Config, device: torch.device | str):
        super().__init__(num_envs, cfg, device)

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
        out = super().step(actions)

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

        # check termination and reset

        self.reset_buf |= (
            torch.abs(self.buffers["base_euler"][:, 1])
            > self.cfg.termination_if_pitch_greater_than
        )
        self.reset_buf |= (
            torch.abs(self.buffers["base_euler"][:, 0])
            > self.cfg.termination_if_roll_greater_than
        )

        return out

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
