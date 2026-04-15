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


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(
        size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device
    ) + lower


class CatbotEnv:

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
        num_obs_raw = obs_cfg["num_obs"]
        self.num_obs = num_obs_raw["main"] if isinstance(num_obs_raw, dict) else num_obs_raw
        self.num_privileged_obs = None
        self.num_commands = len(command_cfg)
        self.device = gs.device

        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length"] / self.dt)
        self.joint_names = sorted(env_cfg["joints"].keys())
        self.num_actions = len(self.joint_names)

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

        self.kp = get_or_default(env_cfg, "kp", 20.0)
        self.kv = get_or_default(env_cfg, "kd", 0.5)
        self.tracking_sigma = get_or_default(reward_cfg, "tracking_sigma", 0.25)
        self.clip_actions = get_or_default(env_cfg, "clip_actions", 100.0)
        self.simulate_action_latency = get_or_default(
            env_cfg, "simulate_action_latency", False
        )
        self.action_scale = get_or_default(env_cfg, "action_scale", 0.25)

        self.cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.dr_cfg: DomainRandConfig = domain_rand_cfg or {}

        self.obs_scales = obs_cfg["scales"]
        self.reward_scales = reward_cfg["rewards"]
        self.targets = reward_cfg["targets"]

        # kp/kd per-env randomization requires batch_dofs_info storage in Genesis
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
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(
                    range(min(kwargs.get("num_rendered_envs", 1), num_envs))
                ),
                **(
                    {"background_color": (0.471, 0.655, 1.0)}
                    if kwargs.get("minecraft")
                    else {}
                ),
            ),
            show_viewer=not headless,
        )

        # add plain
        if kwargs.get("minecraft"):
            self.scene.add_entity(
                gs.morphs.Plane(),
                surface=gs.surfaces.Plastic(
                    roughness=1.0,
                    ior=1.0,
                    diffuse_texture=gs.textures.ImageTexture(
                        image_path="assets/grass_texture.jpg",
                    ),
                ),
            )
        else:
            self.scene.add_entity(
                gs.morphs.URDF(
                    file="urdf/plane/plane.urdf",
                    fixed=True,
                )
            )

        # add robot
        self.robot: RigidEntity = self.scene.add_entity(
            gs.morphs.MJCF(
                file=self.cfg["robot_description_path"],
                pos=self.cfg["base_init_pos"],
                quat=self.cfg["base_init_quat"],
            ),
        )  # type: ignore

        # allow subclasses to attach sensors before the scene is compiled
        self._add_sensors_before_build()

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dofs_idx_local[0] for name in self.joint_names],
            dtype=gs.tc_int,
            device=gs.device,
        )

        # masks into dof_pos for hip vs non-hip joints (used by split default-pose rewards)
        self.hip_dof_mask = torch.tensor(
            ["hip" in name for name in self.joint_names], dtype=torch.bool, device=gs.device
        )
        self.leg_dof_mask = ~self.hip_dof_mask

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

        # Per-env calibration offset buffers (zeroed until DR randomizes them).
        # obs_offset: added to observed dof_pos — simulates encoder miscalibration.
        # init_offset: added to the reset joint pose — simulates physical misalignment.
        self.dof_pos_obs_offset = torch.zeros(
            (self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device
        )
        self.dof_pos_init_offset = torch.zeros(
            (self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device
        )

        # PD control parameters

        # Define global gravity direction vector
        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device
        )

        # Initial state
        self.init_base_pos = torch.tensor(
            self.cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device
        )
        self.init_base_quat = torch.tensor(
            self.cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device
        )
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_qpos = self.robot.get_qpos()[0]
        self.init_projected_gravity: torch.Tensor = transform_by_quat(
            self.global_gravity, self.inv_base_init_quat
        )  # type: ignore

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
        self.commands_limits: tuple[torch.Tensor, torch.Tensor] = tuple(
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                self.command_cfg["lin_vel_x"],
                self.command_cfg["lin_vel_y"],
                self.command_cfg["ang_vel_z"],
            )
        )  # type: ignore
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
        self.push_buf = torch.zeros((self.num_envs,), dtype=gs.tc_int, device=gs.device)

        # Set up end effector (foot) link tracking for body contact termination
        foot_link_names = get_or_default(env_cfg, "foot_link_names", [])
        foot_idx_local_set = {
            self.robot.get_link(name).idx_local for name in foot_link_names
        }
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

        # Foot position tracking for clearance reward
        self.foot_link_idx = (
            torch.tensor(
                [self.robot.get_link(name).idx_local for name in foot_link_names],
                dtype=gs.tc_int,
                device=gs.device,
            )
            if foot_link_names
            else None
        )
        n_feet = len(foot_link_names)
        self.foot_pos = torch.zeros(
            (self.num_envs, n_feet, 3), dtype=gs.tc_float, device=gs.device
        )
        self.last_foot_pos = torch.zeros_like(self.foot_pos)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), dtype=gs.tc_float, device=gs.device
            )

    def _add_sensors_before_build(self) -> None:
        """Hook for subclasses to attach Genesis sensors before scene.build()."""
        pass

    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def step(self, actions, command=None):
        self.actions = torch.clip(actions, -self.clip_actions, self.clip_actions)
        exec_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )
        target_dof_pos = exec_actions * self.cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # apply random pushes to base and stumbles to feet
        self._apply_pushes()
        self._apply_stumbles()

        # update foot positions
        if self.foot_link_idx is not None:
            self.last_foot_pos.copy_(self.foot_pos)
            self.foot_pos = self.robot.get_links_pos(self.foot_link_idx)

        # update buffers
        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler: torch.Tensor = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=False,
        )  # type: ignore
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel: torch.Tensor = transform_by_quat(
            self.robot.get_vel(), inv_base_quat
        )  # type: ignore
        self.base_ang_vel: torch.Tensor = transform_by_quat(
            self.robot.get_ang(), inv_base_quat
        )  # type: ignore
        self.projected_gravity: torch.Tensor = transform_by_quat(
            self.global_gravity, inv_base_quat
        )  # type: ignore
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # compute reward
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
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
                self.episode_length_buf % int(self.cfg["resampling_time"] / self.dt)
                == 0
            )

        if self.scene.viewer:
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
            vel_vec: torch.Tensor = transform_by_quat(
                self.base_lin_vel[0, :], self.base_quat[0, :]
            )  # type: ignore
            self.vel_debug_arrow = self.scene.draw_debug_arrow(
                self.base_pos[0, :].cpu(),
                vel_vec.cpu(),
                color=(1, 0, 0, 0.5),
            )

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= (
            torch.abs(self.base_euler[:, 1])
            > self.cfg["termination_if_pitch_greater_than"]
        )
        self.reset_buf |= (
            torch.abs(self.base_euler[:, 0])
            > self.cfg["termination_if_roll_greater_than"]
        )
        if (
            self.body_contact_height_threshold is not None
            and len(self.non_foot_link_idx) > 0
        ):
            # Terminate if any non-foot link touches the ground
            non_foot_pos = self.robot.get_links_pos(
                self.non_foot_link_idx
            )  # (n_envs, n_links, 3)
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
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.default_dof_pos + self.dof_pos_init_offset)
            self.base_pos.copy_(self.init_base_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.foot_pos.zero_()
            self.last_foot_pos.zero_()
            self.episode_length_buf.zero_()
            self.push_buf.zero_()
            self.reset_buf.fill_(True)
            self._randomize_on_reset(None)
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
                envs_idx[:, None],
                self.default_dof_pos + self.dof_pos_init_offset,
                self.dof_pos,
                out=self.dof_pos,
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
            self.foot_pos.masked_fill_(envs_idx[:, None, None], 0.0)
            self.last_foot_pos.masked_fill_(envs_idx[:, None, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.push_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        # domain randomization on reset
        self._randomize_on_reset(envs_idx)

        # fill extras
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.cfg["episode_length"]
            value.masked_fill_(envs_idx, 0.0)

        # random sample command upon reset
        self._resample_commands(envs_idx)

    def _update_observation(self):
        ang_vel = self.base_ang_vel
        dof_pos = self.dof_pos - self.default_dof_pos + self.dof_pos_obs_offset
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

        self.obs_dict["main"] = torch.concatenate(
            (
                ang_vel * self.obs_scales["ang_vel_z"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                dof_pos * self.obs_scales["dof_pos"],  # 12
                dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ),
            dim=-1,
        )

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.obs_dict

    # ------------ domain randomization ----------------

    def _randomize_on_reset(self, envs_idx):
        """Re-randomize per-env physics parameters for the given environments.

        envs_idx: bool mask (n_envs,) or None (all envs).
        Genesis setters expect integer indices, so we convert here.
        """
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

        if "calibration_offset" in on_reset:
            lo, hi = on_reset["calibration_offset"]
            new_obs = torch.empty(n, self.num_actions, device=gs.device).uniform_(lo, hi)
            new_init = torch.empty(n, self.num_actions, device=gs.device).uniform_(lo, hi)
            if env_indices is None:
                self.dof_pos_obs_offset.copy_(new_obs)
                self.dof_pos_init_offset.copy_(new_init)
            else:
                self.dof_pos_obs_offset[env_indices] = new_obs
                self.dof_pos_init_offset[env_indices] = new_init

    def _apply_pushes(self):
        """Apply random velocity impulses to the robot base on a fixed interval."""
        pushes_cfg = (
            self.dr_cfg.get("pushes", {}) if self.dr_cfg.get("enabled", False) else {}
        )
        if not pushes_cfg.get("enabled", False):
            return

        interval_steps = max(1, int(pushes_cfg.get("interval_s", 10.0) / self.dt))
        max_vel = pushes_cfg.get("max_vel_xy", 0.5)

        self.push_buf += 1
        push_mask = self.push_buf >= interval_steps
        if not push_mask.any():
            return

        push_indices = torch.where(push_mask)[0]
        n = len(push_indices)

        angles = torch.rand(n, device=gs.device) * 2 * math.pi
        mags = torch.rand(n, device=gs.device) * max_vel
        delta_vel = torch.zeros(n, 2, device=gs.device)
        delta_vel[:, 0] = mags * torch.cos(angles)
        delta_vel[:, 1] = mags * torch.sin(angles)

        # For a free-floating MJCF robot the first two DOFs are base x,y velocity (world frame)
        base_xy_dof_idx = torch.tensor([0, 1], dtype=gs.tc_int, device=gs.device)
        current_vel = self.robot.get_dofs_velocity(base_xy_dof_idx)  # (n_envs, 2)
        push_vel = current_vel[push_indices] + delta_vel  # (n_push, 2)
        self.robot.set_dofs_velocity(push_vel, base_xy_dof_idx, envs_idx=push_indices)

        self.push_buf.masked_fill_(push_mask, 0)

    def _apply_stumbles(self):
        """Apply random joint velocity impulses when the robot is moving to simulate feet catching on obstacles."""
        stumbles_cfg = (
            self.dr_cfg.get("stumbles", {}) if self.dr_cfg.get("enabled", False) else {}
        )
        if not stumbles_cfg.get("enabled", False):
            return

        probability = stumbles_cfg.get("probability", 0.05)
        max_impulse = stumbles_cfg.get("max_vel_impulse", 2.0)

        # Only stumble envs where the robot is actively moving
        moving_mask = self.base_lin_vel[:, :2].norm(dim=-1) > 0.1
        stumble_mask = moving_mask & (
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
        push_vel = current_vel[stumble_indices] + impulse
        self.robot.set_dofs_velocity(
            push_vel, self.motors_dof_idx, envs_idx=stumble_indices
        )

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

    def _perceived_dof_pos(self):
        """dof_pos relative to default as the policy observes it (includes calibration offset)."""
        return self.dof_pos - self.default_dof_pos + self.dof_pos_obs_offset

    def _reward_similar_to_default(self):
        # Penalize joint poses far from default, measured in perceived space
        return torch.sum(torch.abs(self._perceived_dof_pos()), dim=1)

    def _reward_similar_to_default_legs(self):
        # Penalize non-hip joint poses far from default (a and l joints)
        diff = torch.abs(self._perceived_dof_pos())
        return torch.sum(diff[:, self.leg_dof_mask], dim=1)

    def _reward_similar_to_default_hips(self):
        # Penalize hip joint poses far from default
        diff = torch.abs(self._perceived_dof_pos())
        return torch.sum(diff[:, self.hip_dof_mask], dim=1)

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

    def _reward_foot_clearance(self):
        # Reward feet lifting during swing phase.
        # Uses horizontal foot speed as a proxy for swing: a foot moving
        # horizontally relative to the last timestep is likely in swing, not stance.
        # Only active when the robot has a non-zero velocity command so it doesn't
        # learn to prance on the spot.
        if self.foot_link_idx is None:
            return torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)

        # Foot displacement in world frame, rotated into base-local frame so the
        # forward direction stays correct when the robot turns.
        foot_disp_world = self.foot_pos - self.last_foot_pos  # (n_envs, n_feet, 3)
        inv_base_quat = inv_quat(self.base_quat)  # (n_envs, 4)
        # rotate each foot's displacement vector into the base frame
        foot_disp_local = transform_by_quat(
            foot_disp_world.view(self.num_envs, -1, 3),
            inv_base_quat[:, None, :].expand(-1, foot_disp_world.shape[1], -1),
        )  # (n_envs, n_feet, 3)
        foot_vel_forward = (
            torch.abs(foot_disp_local[..., 0]) / self.dt
        )  # (n_envs, n_feet)

        foot_height = self.foot_pos[..., 2]  # (n_envs, n_feet)

        # Gate clearance reward on the foot moving forwards/backwards in robot frame
        swing_mask = (foot_vel_forward > 0.05).float()
        clearance = foot_height * swing_mask  # (n_envs, n_feet)

        # Scale by command speed so reward is zero when standing still
        cmd_speed = torch.norm(self.commands[:, :2], dim=1, keepdim=True)  # (n_envs, 1)
        return (clearance * cmd_speed).sum(dim=1)

def _reward_stretching_pain(self):
        """
        Penalizes the robot when the leg approaches the physical bounds of the t_c angle.
        The penalty ramps up ^2 the closer it gets to the boundary.
        """
        pain = torch.zeros(self.num_envs, device=self.device)

        # We need to process all legs. We'll identify the 'a' and 'l' joints for each leg based on their names.
        a_indices = []
        l_indices = []
        for i, name in enumerate(self.joint_names):
            if (
                name.endswith("_a")
                or name == "a"
                or "_a_" in name
                or name.split("_")[-1] == "a"
            ):
                a_indices.append(i)
            elif (
                name.endswith("_l")
                or name == "l"
                or "_l_" in name
                or name.split("_")[-1] == "l"
            ):
                l_indices.append(i)

        # If we can't cleanly match the pairs, just return zero penalty
        if len(a_indices) != len(l_indices) or len(a_indices) == 0:
            return pain

        for a_idx, l_idx in zip(a_indices, l_indices):
            t_a_rad = self.dof_pos[:, a_idx]
            t_l_rad = self.dof_pos[:, l_idx]

            t_a_deg = t_a_rad * (180.0 / 3.141592653589793) - 90.0
            t_l_deg = t_l_rad * (180.0 / 3.141592653589793) - 90.0

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
            leg_pain = torch.where(
                min_dist < threshold,
                -(((threshold - min_dist) / threshold) ** 2),
                torch.zeros_like(t_a_rad),
            )

            pain += torch.where(valid_mask, leg_pain, -1.0)

        return pain