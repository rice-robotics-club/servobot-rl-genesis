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
        from src.env.ik_solver import convert_ta_tl_to_160_225

        self.convert_ta_tl_to_160_225 = convert_ta_tl_to_160_225

        # Physical t_c limits derived from catbot_leg hardware constants.
        # t_c = l_pos - theta3; bounds match the linkage hard stops.
        self.tc_min = -1.22173   # -70°
        self.tc_max =  1.047198  # +60°

        self.kp = get_or_default(env_cfg, "kp", 20.0)
        self.kv = get_or_default(env_cfg, "kd", 0.5)
        hip_limits = get_or_default(env_cfg, "hip_limits", [-0.3491, 0.7854])
        self.hip_min = hip_limits[0]
        self.hip_max = hip_limits[1]
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
        # left/right hip masks — motors face opposite directions so limits are mirrored
        self.left_hip_mask = torch.tensor(
            ["hip" in name and name[1] == "L" for name in self.joint_names], dtype=torch.bool, device=gs.device
        )
        self.right_hip_mask = torch.tensor(
            ["hip" in name and name[1] == "R" for name in self.joint_names], dtype=torch.bool, device=gs.device
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

        # Chassis link index for body mass correction / payload DR
        self.chassis_link_idx = torch.tensor(
            [self.robot._links[0].idx_local], dtype=gs.tc_int, device=gs.device
        )

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
        # clamp hip joints
        target_dof_pos[:, self.right_hip_mask] = target_dof_pos[:, self.right_hip_mask].clamp(
            self.hip_min, self.hip_max
        )
        target_dof_pos[:, self.left_hip_mask] = target_dof_pos[:, self.left_hip_mask].clamp(
            -self.hip_max, -self.hip_min
        )
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

        body_mass_offset = on_reset.get("body_mass_offset", 0.0)
        body_mass_lo, body_mass_hi = on_reset.get("body_mass_shift", [0.0, 0.0])
        if body_mass_offset != 0.0 or body_mass_lo != 0.0 or body_mass_hi != 0.0:
            body_mass_rand = (
                torch.empty(n, 1, device=gs.device).uniform_(body_mass_lo, body_mass_hi)
                + body_mass_offset
            )
            self.robot.set_mass_shift(
                body_mass_rand, links_idx_local=self.chassis_link_idx, envs_idx=env_indices
            )

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
        Penalizes each leg when its t_c angle (between 160mm and 225mm_top links)
        enters the danger zone within 5° of the physical limits.
        Returns a positive value; apply a negative weight in the config.

        Uses the same direct calculate_theta3 approach as catbot_leg to avoid
        Freudenstein branch-selection ambiguity.
        """
        from src.env.catbot_leg import calculate_theta3 as _calc_theta3
        pain = torch.zeros(self.num_envs, device=self.device)

        a_indices = [i for i, n in enumerate(self.joint_names) if n.split("_")[-1] == "a" or n == "a"]
        l_indices = [i for i, n in enumerate(self.joint_names) if n.split("_")[-1] == "l" or n == "l"]

        if len(a_indices) != len(l_indices) or len(a_indices) == 0:
            return pain

        threshold = 5.0 * math.pi / 180.0  # 5° danger zone

        for a_idx, l_idx in zip(a_indices, l_indices):
            a_pos = self.dof_pos[:, a_idx]
            l_pos = self.dof_pos[:, l_idx]

            # Same convention as catbot_leg._clamp_target_dof_pos
            theta3 = _calc_theta3(a_pos - (torch.pi / 2)) + torch.pi
            l_min = theta3 - 1.22173
            l_max = theta3 + 1.047198

            # t_c = 0 at center of range, negative toward l_min, positive toward l_max
            t_c = l_pos - theta3
            dist_min_side = torch.abs(l_pos - l_min)   # distance to lower limit
            dist_max_side = torch.abs(l_max - l_pos)   # distance to upper limit
            min_dist = torch.minimum(dist_min_side, dist_max_side)

            normalized = (threshold - min_dist) / threshold
            leg_pain = torch.where(
                min_dist < threshold,
                torch.exp(normalized * 6.0) - 1.0,
                torch.zeros_like(a_pos),
            )
            pain += leg_pain

        return pain

    def _reward_hip_pain(self):
        """
        Penalizes hip joints entering the danger zone within 5° of their physical limits.
        Limits: -20° (-0.3491 rad) to +45° (0.7854 rad).
        Returns a positive value; apply a negative weight in the config.
        """
        threshold = 5.0 * math.pi / 180.0

        def _hip_pain(hip_pos, lo, hi):
            dist_min = torch.abs(hip_pos - lo)
            dist_max = torch.abs(hi - hip_pos)
            min_dist = torch.minimum(dist_min, dist_max)
            normalized = (threshold - min_dist) / threshold
            return torch.where(
                min_dist < threshold,
                torch.exp(normalized * 6.0) - 1.0,
                torch.zeros_like(hip_pos),
            )

        right_pain = _hip_pain(self.dof_pos[:, self.right_hip_mask], self.hip_min, self.hip_max)
        left_pain  = _hip_pain(self.dof_pos[:, self.left_hip_mask], -self.hip_max, -self.hip_min)
        return right_pain.sum(dim=1) + left_pain.sum(dim=1)