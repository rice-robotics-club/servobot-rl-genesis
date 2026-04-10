import math
from typing import TYPE_CHECKING

import genesis as gs
import numpy as np
import torch
from genesis.engine.sensors.imu import IMUData, IMUSensor
from genesis.utils.geom import (
    inv_quat,
    quat_to_xyz,
    transform_by_quat,
    transform_quat_by_quat,
)
from tensordict import TensorDict

from src.config import (
    CommandConfig,
    DomainRandConfig,
    EnvConfig,
    ObsConfig,
    RewardConfig,
)
from src.env.genesis import get_or_default

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity
    from genesis.engine.entities.rigid_entity import RigidLink


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(
        size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device
    ) + lower


class CatbotLegEnv:
    def __init__(
        self,
        num_envs,
        env_cfg: EnvConfig,
        obs_cfg: ObsConfig,
        reward_cfg: RewardConfig,
        command_cfg: CommandConfig,
        headless=False,
        debug=False,
        domain_rand_cfg: DomainRandConfig | None = None,
        **kwargs,
    ):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_commands = len(command_cfg)
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length"] / self.dt)
        self.joint_names = sorted(env_cfg["joints"].keys())
        self.num_actions = len(self.joint_names)

        self.cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["scales"]
        self.reward_scales = reward_cfg["rewards"]
        self.targets = reward_cfg["targets"]
        self.dr_cfg: DomainRandConfig = domain_rand_cfg or {}

        dr_on_reset = (
            self.dr_cfg.get("on_reset", {}) if self.dr_cfg.get("enabled", False) else {}
        )
        _needs_dof_batching = "kp" in dr_on_reset or "kd" in dr_on_reset

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
                batch_dofs_info=_needs_dof_batching,
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

        # add plain
        self.plane: RigidEntity = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True,
            )
        )  # type: ignore

        # add robot
        self.robot: RigidEntity = self.scene.add_entity(
            gs.morphs.URDF(
                file=self.cfg["robot_description_path"],
                recompute_inertia=True,
                links_to_keep=[
                    "imu_link",
                    "closing_BR_leg0_1 (1) (1)_1",
                    "closing_BR_leg0_1 (1) (1)_2",
                    "closing_BR_leg0_1 (1) (1)_1_z",
                    "closing_BR_leg0_1 (1) (1)_2_z",
                    "closing_BR_leg3_1 (1) (1)_1",
                    "closing_BR_leg3_1 (1) (1)_2",
                    "closing_BR_leg3_1 (1) (1)_1_z",
                    "closing_BR_leg3_1 (1) (1)_2_z",
                ],
                decompose_robot_error_threshold=0.0,
                fixed=True,
            )
        )  # type: ignore

        self.add_equality_constraints()

        self.imu_link: RigidLink = self.robot.get_link("imu_link")

        self.imu: IMUSensor = self.scene.add_sensor(
            gs.sensors.IMU(
                entity_idx=self.robot.idx,  # type: ignore
                link_idx_local=self.imu_link.idx_local,  # type: ignore
                pos_offset=(0.0, 0.0, 0.0),  # type: ignore
                # sensor characteristics
                acc_cross_axis_coupling=(0.0, 0.01, 0.02),
                gyro_cross_axis_coupling=(0.03, 0.04, 0.05),
                acc_noise=(0.01, 0.01, 0.01),
                gyro_noise=(0.01, 0.01, 0.01),
                acc_random_walk=(0.001, 0.001, 0.001),
                gyro_random_walk=(0.001, 0.001, 0.001),
                delay=0.02,
                jitter=0.01,  # type: ignore
                interpolate=True,  # type: ignore
                draw_debug=True,
            )
        )

        # build
        self.scene.build(n_envs=num_envs)

        # self.scene.sim.rigid_solver.add_weld_constraint(self.plane.idx, self.robot.idx)

        # names to indices
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dofs_idx_local[0] for name in self.joint_names],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.yaw_idx = torch.tensor(
            [self.robot.get_joint("yaw").dofs_idx_local[0]],
            dtype=gs.tc_int,
            device=gs.device,
        )

        all_dof_idx = torch.arange(self.robot.n_dofs, device=gs.device)
        linkage_dof_idx = all_dof_idx[~torch.isin(all_dof_idx, self.motors_dof_idx)]
        self.robot.set_dofs_kp([0.0] * len(linkage_dof_idx), linkage_dof_idx)
        self.robot.set_dofs_kv([0.0] * len(linkage_dof_idx), linkage_dof_idx)

        self.robot.set_dofs_kp([self.cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.cfg["kd"]] * self.num_actions, self.motors_dof_idx)

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
        self.init_imu_pos = self.imu_link.get_pos()
        self.init_imu_quat = self.imu_link.get_quat()
        self.inv_imu_init_quat = inv_quat(self.init_imu_quat)
        self.init_qpos = self.robot.get_qpos()[0]
        self.init_projected_gravity: torch.Tensor = transform_by_quat(
            self.global_gravity, self.inv_imu_init_quat
        )  # type: ignore

        # initialize buffers
        self.imu_lin_vel = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.imu_ang_vel = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.projected_gravity = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.obs_dict = TensorDict(
            {
                group: torch.empty(
                    (self.num_envs, num), dtype=gs.tc_float, device=gs.device
                )
                for group, num in self.num_obs.items()
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
            [self.obs_scales["vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.commands_limits: tuple[torch.Tensor, torch.Tensor] = (
            torch.tensor(
                [self.command_cfg["vel"][0]], dtype=gs.tc_float, device=gs.device
            ),
            torch.tensor(
                [self.command_cfg["vel"][1]], dtype=gs.tc_float, device=gs.device
            ),
        )

        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.gait_phase = torch.zeros(
            (self.num_envs,), dtype=gs.tc_float, device=gs.device
        )
        self.gait_period = get_or_default(env_cfg, "gait_period", 0.5)  # seconds
        self.yaw_vel = torch.zeros(
            (self.num_envs,), dtype=gs.tc_float, device=gs.device
        )
        self.yaw_pos = torch.zeros(
            (self.num_envs,), dtype=gs.tc_float, device=gs.device
        )
        self.goal_yaw_pos = torch.zeros(
            (self.num_envs,), dtype=gs.tc_float, device=gs.device
        )
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.imu_pos = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.imu_quat = torch.empty(
            (self.num_envs, 4), dtype=gs.tc_float, device=gs.device
        )
        self.extras = dict()  # extra information for logging

        # foot position buffer — populated after foot_link_idx is set up below
        self.foot_pos = torch.zeros(
            (self.num_envs, 1, 3), dtype=gs.tc_float, device=gs.device
        )

        self.kp = get_or_default(env_cfg, "kp", 20.0)
        self.kv = get_or_default(env_cfg, "kd", 0.5)

        foot_link_names = get_or_default(env_cfg, "foot_link_names", [])
        foot_idx_local_set = {
            self.robot.get_link(name).idx_local for name in foot_link_names
        }
        self.foot_link_idx = (
            torch.tensor(
                [self.robot.get_link(name).idx_local for name in foot_link_names],
                dtype=gs.tc_int,
                device=gs.device,
            )
            if foot_link_names
            else None
        )
        self.non_foot_link_idx = torch.tensor(
            [
                link.idx_local
                for link in self.robot._links
                if link.idx_local not in foot_idx_local_set
            ],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.body_contact_height_threshold = get_or_default(
            env_cfg, "termination_if_body_contact_height", None
        )

        # Joint limits for safety penalty
        # 4-bar linkage lengths for stretching pain calculation
        self.l0, self.l1, self.l2, self.l3 = 8.5, 6.2, 8.7, 16.63
        from src.env.ik_solver import convert_ta_tl_to_160_225, get_tc_limits

        self.convert_ta_tl_to_160_225 = convert_ta_tl_to_160_225

        # Compute safe limits for t_c
        self.tc_min, self.tc_max = get_tc_limits(
            self.l0, self.l1, self.l2, self.l3, step=2.0
        )
        if self.tc_min is None or self.tc_max is None:
            self.tc_min, self.tc_max = 0.0, 3.14

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
            self.yaw_pos.zero_()
            self.goal_yaw_pos.zero_()
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)
            self.yaw_pos.masked_fill_(envs_idx, 0.0)
            self.goal_yaw_pos.masked_fill_(envs_idx, 0.0)

    def step(self, actions, command=None):
        self.actions = torch.clip(
            actions, -self.cfg["clip_actions"], self.cfg["clip_actions"]
        )
        exec_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )
        target_dof_pos = exec_actions * self.cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # apply stumbling perturbations
        self._apply_stumbles()

        # update buffers
        self.episode_length_buf += 1
        self.gait_phase = (
            self.gait_phase + 2 * math.pi * self.dt / self.gait_period
        ) % (2 * math.pi)
        self.imu_pos = self.imu_link.get_pos()
        self.imu_quat = self.imu_link.get_quat()
        self.imu_euler: torch.Tensor = quat_to_xyz(
            transform_quat_by_quat(self.inv_imu_init_quat, self.imu_quat),
            rpy=True,
            degrees=False,
        )  # type: ignore
        inv_imu_quat = inv_quat(self.imu_quat)
        self.imu_lin_vel: torch.Tensor = transform_by_quat(
            self.imu_link.get_vel(), inv_imu_quat
        )  # type: ignore
        self.imu_ang_vel: torch.Tensor = transform_by_quat(
            self.imu_link.get_ang(), inv_imu_quat
        )  # type: ignore
        self.projected_gravity: torch.Tensor = transform_by_quat(
            self.global_gravity, inv_imu_quat
        )  # type: ignore
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)
        yaw_vel_raw = self.robot.get_dofs_velocity(self.yaw_idx)
        self.yaw_vel = yaw_vel_raw.squeeze(-1) if yaw_vel_raw.ndim > 1 else yaw_vel_raw
        self.yaw_pos = self.yaw_pos + self.yaw_vel * self.dt
        self.goal_yaw_pos = self.goal_yaw_pos + self.commands[:, 0] * self.dt
        if self.foot_link_idx is not None:
            self.foot_pos = self.robot.get_links_pos(self.foot_link_idx)

        # compute reward
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # DEBUG: print yaw signals every 100 steps
        if self.episode_length_buf[0] % 100 == 0 and self.scene.viewer:
            obs_yaw = -self.yaw_vel[0].item()  # What policy sees
            print(
                f"DEBUG: cmd={self.commands[0, 0].item():.3f}, obs_yaw={obs_yaw:.3f}, raw_yaw_vel={self.yaw_vel[0].item():.3f}, rew={self.rew_buf[0].item():.4f}"
            )

        if command:
            self.commands[:, 0] = (
                (command[0] * 0.5 + 0.5)
                * (self.command_cfg["vel"][1] - self.command_cfg["vel"][0])
            ) + self.command_cfg["vel"][0]
        else:
            # resample commands
            self._resample_commands(
                self.episode_length_buf % int(self.cfg["resampling_time"] / self.dt)
                == 0
            )

        if self.scene.viewer:
            # visualize commanded and actual velocity
            self.scene.clear_debug_objects()

            cmd_vec = torch.zeros(3)
            cmd_vec[1] = self.commands[0, 0]
            cmd_vec: torch.Tensor = transform_by_quat(cmd_vec, self.imu_quat[0, :])  # type: ignore

            yaw_vec = torch.zeros(3)
            yaw_vec[1] = self.imu_ang_vel[0, 2]  # Use IMU yaw (what robot sees)
            yaw_vec: torch.Tensor = transform_by_quat(yaw_vec, self.imu_quat[0, :])  # type: ignore

            self.cmd_debug_arrow = self.scene.draw_debug_arrow(
                self.imu_pos[0, :].cpu(),
                cmd_vec.cpu(),
                color=(0, 0, 1, 0.5),
            )
            self.vel_debug_arrow = self.scene.draw_debug_arrow(
                self.imu_pos[0, :].cpu(),
                yaw_vec.cpu(),
                color=(1, 0, 0, 0.5),
            )

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        if (
            self.body_contact_height_threshold is not None
            and len(self.non_foot_link_idx) > 0
        ):
            non_foot_pos = self.robot.get_links_pos(self.non_foot_link_idx)
            self.reset_buf |= (
                non_foot_pos[..., 2] < self.body_contact_height_threshold
            ).any(dim=-1)

        # Compute timeout
        self.extras["time_outs"] = (
            self.episode_length_buf > self.max_episode_length
        ).to(dtype=gs.tc_float)

        # Reset environment if necessary
        self._reset_idx(self.reset_buf)

        # update observations
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_dict

    def _reset_idx(self, envs_idx=None):
        # reset state
        self.robot.set_qpos(
            self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True
        )

        # reset buffers
        if envs_idx is None:
            self.imu_pos.copy_(self.init_imu_pos)
            self.imu_quat.copy_(self.init_imu_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.default_dof_pos)
            self.imu_pos.copy_(self.init_imu_pos)
            self.imu_lin_vel.zero_()
            self.imu_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.gait_phase.zero_()
            self.yaw_pos.zero_()
            self.goal_yaw_pos.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
            self.foot_pos.zero_()
        else:
            torch.where(
                envs_idx[:, None], self.init_imu_pos, self.imu_pos, out=self.imu_pos
            )
            torch.where(
                envs_idx[:, None],
                self.init_imu_quat,
                self.imu_quat,
                out=self.imu_quat,
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
                envs_idx[:, None], self.init_imu_pos, self.imu_pos, out=self.imu_pos
            )
            self.imu_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.imu_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.gait_phase.masked_fill_(envs_idx, 0.0)
            self.yaw_pos.masked_fill_(envs_idx, 0.0)
            self.goal_yaw_pos.masked_fill_(envs_idx, 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        # fill extras
        n_envs = self.num_envs if envs_idx is None else envs_idx.sum()
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.cfg["episode_length"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

        # domain randomization on reset
        self._randomize_on_reset(envs_idx)

        # random sample command upon reset
        self._resample_commands(envs_idx)

    def _update_observation(self):
        data: IMUData = self.imu.read()  # type: ignore
        ang_vel = self.imu_ang_vel.clone()
        ang_vel[:, 2] = -ang_vel[:, 2]  # Fix IMU yaw sign to match joint velocity
        dof_pos = self.dof_pos - self.default_dof_pos
        dof_vel = self.dof_vel

        noise_cfg = (
            self.dr_cfg.get("obs_noise", {})
            if self.dr_cfg.get("enabled", False)
            else {}
        )
        if noise_cfg:
            if "ang_vel" in noise_cfg:
                ang_vel = ang_vel + torch.randn_like(ang_vel) * noise_cfg["ang_vel"]
            if "dof_pos" in noise_cfg:
                dof_pos = dof_pos + torch.randn_like(dof_pos) * noise_cfg["dof_pos"]
            if "dof_vel" in noise_cfg:
                dof_vel = dof_vel + torch.randn_like(dof_vel) * noise_cfg["dof_vel"]

        self.obs_dict["shared"] = torch.concatenate(
            (
                ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 1
                dof_pos * self.obs_scales["dof_pos"],  # 2
                dof_vel * self.obs_scales["dof_vel"],  # 2
                self.actions,  # 2
            ),
            dim=-1,
        )
        self.obs_dict["teacher"] = torch.concatenate(
            (
                self.imu_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
            ),
            dim=-1,
        )
        self.obs_dict["student"] = torch.concatenate(
            (
                data.ang_vel,  # 3
                data.lin_acc,  # 3
                torch.sin(self.gait_phase).unsqueeze(-1),  # 1
                torch.cos(self.gait_phase).unsqueeze(-1),  # 1
            ),
            dim=-1,
        )

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.obs_dict

    def add_equality_constraints(self):
        eq_data = np.array([0.0] * 11)
        sol_params = np.array(
            [
                2.22044605e-16,
                1.00000000e00,
                9.00000000e-01,
                9.50000000e-01,
                1.00000000e-03,
                5.00000000e-01,
                2.00000000e00,
            ]
        )
        self.robot._add_equality(
            "closing_BR_leg0_1 (1) (1)",
            gs.EQUALITY_TYPE.CONNECT,
            ["closing_BR_leg0_1 (1) (1)_1", "closing_BR_leg0_1 (1) (1)_2"],
            eq_data,
            sol_params,
        )
        self.robot._add_equality(
            "closing_BR_leg0_1 (1) (1)_z",
            gs.EQUALITY_TYPE.CONNECT,
            ["closing_BR_leg0_1 (1) (1)_1_z", "closing_BR_leg0_1 (1) (1)_2_z"],
            eq_data,
            sol_params,
        )
        self.robot._add_equality(
            "closing_BR_leg3_1 (1) (1)",
            gs.EQUALITY_TYPE.CONNECT,
            ["closing_BR_leg3_1 (1) (1)_1", "closing_BR_leg3_1 (1) (1)_2"],
            eq_data,
            sol_params,
        )
        self.robot._add_equality(
            "closing_BR_leg3_1 (1) (1)_z",
            gs.EQUALITY_TYPE.CONNECT,
            ["closing_BR_leg3_1 (1) (1)_1_z", "closing_BR_leg3_1 (1) (1)_2_z"],
            eq_data,
            sol_params,
        )

    # ------------ domain randomization ----------------

    def _randomize_on_reset(self, envs_idx):
        if not self.dr_cfg.get("enabled", False):
            return
        on_reset = self.dr_cfg.get("on_reset", {})
        if not on_reset:
            return

        if envs_idx is None:
            env_indices = None
            n = self.num_envs
        else:
            env_indices = torch.where(envs_idx)[0]
            n = len(env_indices)
            if n == 0:
                return

        if "kp" in on_reset:
            lo, hi = on_reset["kp"]
            kp_rand = torch.empty(n, self.num_actions, device=gs.device).uniform_(
                lo, hi
            )
            self.robot.set_dofs_kp(kp_rand, self.motors_dof_idx, envs_idx=env_indices)

        if "kd" in on_reset:
            lo, hi = on_reset["kd"]
            kd_rand = torch.empty(n, self.num_actions, device=gs.device).uniform_(
                lo, hi
            )
            self.robot.set_dofs_kv(kd_rand, self.motors_dof_idx, envs_idx=env_indices)

        if "friction" in on_reset and self.foot_link_idx is not None:
            lo, hi = on_reset["friction"]
            n_feet = len(self.foot_link_idx)
            friction_rand = torch.empty(n, n_feet, device=gs.device).uniform_(lo, hi)
            self.robot.set_friction_ratio(
                friction_rand, links_idx_local=self.foot_link_idx, envs_idx=env_indices
            )

        if "mass_shift" in on_reset:
            lo, hi = on_reset["mass_shift"]
            n_links = self.robot.n_links
            mass_rand = torch.empty(n, n_links, device=gs.device).uniform_(lo, hi)
            self.robot.set_mass_shift(mass_rand, envs_idx=env_indices)

        if "com_shift" in on_reset:
            std = on_reset["com_shift"]
            n_links = self.robot.n_links
            com_rand = torch.randn(n, n_links, 3, device=gs.device) * std
            self.robot.set_COM_shift(com_rand, envs_idx=env_indices)

    def _apply_stumbles(self):
        """Apply random joint velocity impulses during swing to simulate foot catching on obstacles."""
        stumbles_cfg = (
            self.dr_cfg.get("stumbles", {}) if self.dr_cfg.get("enabled", False) else {}
        )
        if not stumbles_cfg.get("enabled", False):
            return

        probability = stumbles_cfg.get("probability", 0.05)
        max_impulse = stumbles_cfg.get("max_vel_impulse", 2.0)

        # Only stumble envs where the leg is actively swinging
        swing_mask = torch.abs(self.yaw_vel) > 0.05
        stumble_mask = swing_mask & (
            torch.rand(self.num_envs, device=gs.device) < probability
        )
        stumble_indices = torch.where(stumble_mask)[0]
        if len(stumble_indices) == 0:
            return

        n = len(stumble_indices)
        impulse = (
            torch.rand(n, self.num_actions, device=gs.device) * 2 - 1
        ) * max_impulse
        current_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)
        new_vel = current_vel.clone()
        new_vel[stumble_indices] += impulse
        self.robot.set_dofs_velocity(
            new_vel, self.motors_dof_idx, envs_idx=stumble_indices
        )

    # ------------ reward functions----------------
    def _reward_foot_clearance(self):
        # Reward foot height when the leg is actively swinging (yaw velocity as proxy).
        # Gates on command magnitude so standing still gives no reward.
        if self.foot_link_idx is None:
            return torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)

        foot_height = self.foot_pos[:, 0, 2]  # (n_envs,) — single foot
        max_height = self.targets.get("foot_clearance_max_height", 0.12)
        foot_height_clamped = foot_height.clamp(max=max_height)
        swing_mask = (torch.abs(self.yaw_vel) > 0.05).float()
        cmd_magnitude = torch.abs(self.commands[:, 0])
        return foot_height_clamped * swing_mask * cmd_magnitude

    def _reward_energy(self):
        exec_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )
        target_dof_pos = exec_actions * self.cfg["action_scale"] + self.default_dof_pos
        pos_error = target_dof_pos - self.dof_pos
        vel_error = -self.dof_vel
        torques = self.kp * pos_error + self.kv * vel_error
        return torch.sum(torch.abs(torques * self.dof_vel), dim=1)

    def _reward_survival(self):
        # Scales with command magnitude so standing still gives no survival reward
        return torch.abs(self.commands[:, 0])

    def _reward_tracking_vel(self):
        # Use negated yaw_vel to match IMU convention (IMU yaw is negated in _update_observation)
        obs_yaw_vel = -self.yaw_vel
        vel_error = self.commands - obs_yaw_vel
        lin_vel_error = torch.sum(torch.square(vel_error), dim=1)
        magnitude_reward = torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

        cmd_sign = torch.sign(self.commands[:, 0])
        vel_sign = torch.sign(obs_yaw_vel + 1e-6)
        same_direction = (cmd_sign * vel_sign) > 0

        command_issued = torch.abs(self.commands[:, 0]) > 0.05
        velocity_moving = torch.abs(obs_yaw_vel) > 0.05

        eligible = same_direction & command_issued & velocity_moving

        return torch.where(
            eligible,
            magnitude_reward + 1.0,
            torch.zeros_like(magnitude_reward),
        )

    def _reward_tracking_yaw_displacement(self):
        error = self.goal_yaw_pos - self.yaw_pos
        tracking_reward = torch.exp(
            -torch.square(error) / self.reward_cfg["tracking_sigma"]
        )

        cmd_sign = torch.sign(self.commands[:, 0])
        vel_sign = torch.sign(self.yaw_vel + 1e-6)
        same_direction = (cmd_sign * vel_sign) > 0
        command_issued = torch.abs(self.commands[:, 0]) > 0.05
        velocity_moving = torch.abs(self.yaw_vel) > 0.05

        eligible = same_direction & command_issued & velocity_moving

        return torch.where(
            eligible,
            tracking_reward + 1.0,
            torch.zeros_like(tracking_reward),
        )

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.imu_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.imu_pos[:, 2] - self.targets["base_height"])

    def _reward_soft_landing(self):
        # Reward for soft landing after a jump or fall, based on rapid positive z acceleration with a negative z velocity
        # This encourages the robot to learn to land softly instead of crashing to the ground
        # Only gives reward when the robot is moving downwards (negative z velocity),
        # and has a rapid positive z acceleration (which indicates a landing impact).
        # this should have a negative weight in config
        z_acc = (self.imu_lin_vel[:, 2] - self.last_dof_vel[:, 1]) / self.dt
        return torch.where(
            self.imu_lin_vel[:, 2] < 0,  # only consider when moving downwards
            torch.clamp(z_acc, min=0.0),  # reward positive acceleration (impact)
            torch.zeros_like(z_acc),  # no reward when not moving downwards
        )

    def _reward_stretching_pain(self):
        """
        Penalizes the robot when the leg approaches the physical bounds of the t_c angle.
        The penalty ramps up ^2 the closer it gets to the boundary.
        """
        t_a_rad = self.dof_pos[:, 0]
        t_l_rad = self.dof_pos[:, 1]

        # Convert to degrees for ik_solver and apply -90 deg offset
        # (Sim 0 is straight down, IK 0 is straight forward)
        t_a_deg = t_a_rad * (180.0 / math.pi) - 90.0
        t_l_deg = t_l_rad * (180.0 / math.pi) - 90.0

        # Vectorized Freudenstein Inverse
        target_theta3_rad = (t_a_deg - 180.0) * (math.pi / 180.0)

        l0, l1, l2, l3 = self.l0, self.l1, self.l2, self.l3

        K1_inv = l0 / l1
        K2_inv = l0 / l3
        K3_inv = (l1**2 - l2**2 + l3**2 + l0**2) / (2.0 * l1 * l3)

        cos_th1 = torch.cos(target_theta3_rad)
        sin_th1 = torch.sin(target_theta3_rad)

        A = cos_th1 - K1_inv - K2_inv * cos_th1 + K3_inv
        B = -2.0 * sin_th1
        C = K1_inv - (K2_inv + 1.0) * cos_th1 + K3_inv

        disc = B**2 - 4.0 * A * C
        valid_mask = disc >= 0.0
        safe_disc = torch.clamp(disc, min=0.0)

        t_false = (-B - torch.sqrt(safe_disc)) / (2.0 * A)
        t_3_false = 2.0 * torch.atan(t_false)

        t_true = (-B + torch.sqrt(safe_disc)) / (2.0 * A)
        t_3_true = 2.0 * torch.atan(t_true)

        # Forward check: calculate_theta3(l0, l3, l2, l1, t_3, open_mode=False)
        K1_fwd = l0 / l3
        K2_fwd = l0 / l1
        K3_fwd = (l3**2 - l2**2 + l1**2 + l0**2) / (2.0 * l3 * l1)

        def forward_pass(t_3_in):
            c_th = torch.cos(t_3_in)
            s_th = torch.sin(t_3_in)
            A_f = c_th - K1_fwd - K2_fwd * c_th + K3_fwd
            B_f = -2.0 * s_th
            C_f = K1_fwd - (K2_fwd + 1.0) * c_th + K3_fwd
            d_f = torch.clamp(B_f**2 - 4.0 * A_f * C_f, min=0.0)
            t_f = (-B_f - torch.sqrt(d_f)) / (2.0 * A_f)
            return 2.0 * torch.atan(t_f)

        out_false = forward_pass(t_3_false)
        out_true = forward_pass(t_3_true)

        err_false = torch.abs(out_false - target_theta3_rad)
        err_true = torch.abs(out_true - target_theta3_rad)

        t_3 = torch.where(err_false < err_true, t_3_false, t_3_true)

        t_c = (t_l_deg - 180.0) * (math.pi / 180.0) - t_3
        t_c = (t_c + math.pi) % (2.0 * math.pi) - math.pi

        dist_min = torch.abs(t_c - self.tc_min)
        dist_max = torch.abs(self.tc_max - t_c)
        min_dist = torch.minimum(dist_min, dist_max)

        threshold = 0.3
        pain = torch.where(
            min_dist < threshold,
            -(((threshold - min_dist) / threshold) ** 2),
            torch.zeros_like(t_a_rad),
        )

        return torch.where(valid_mask, pain, -1.0)
