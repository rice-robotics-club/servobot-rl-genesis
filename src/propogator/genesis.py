from typing import TYPE_CHECKING

import genesis as gs
import torch
from genesis.utils.geom import (
    inv_quat,
    quat_to_xyz,
    transform_by_quat,
    transform_quat_by_quat,
)

from src.config import EnvConfig, ObsConfig, RewardConfig
from src.propogator.propogator import Propogator

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity


class GenesisPropogator(Propogator):
    def __init__(
        self,
        num_envs: int,
        env_cfg: EnvConfig,
        obs_cfg: ObsConfig,
        reward_cfg: RewardConfig,
        device: torch.device | str,
        headless: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(num_envs, device)

        self.num_envs = num_envs
        self.headless = headless
        self.debug = debug
        self.num_actions = len(env_cfg.joints)
        self.joint_names = sorted(env_cfg.joints.keys())

        self.obs_cfg = obs_cfg

        # init default values
        self.dt = env_cfg.dt
        self.base_init_pos = env_cfg.base_init_pos
        self.base_init_quat = env_cfg.base_init_quat
        self.kp = env_cfg.kp
        self.kv = env_cfg.kv
        self.clip_actions = env_cfg.clip_actions
        self.simulate_action_latency = env_cfg.simulate_action_latency
        self.action_scale = env_cfg.action_scale

        gs.init(precision="32", logging_level="warning", performance_mode=True)

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
                file=env_cfg.urdf_path,
                pos=self.base_init_pos,
                quat=self.base_init_quat,
            ),
        )  # pyright: ignore

        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dofs_idx_local[0] for name in self.joint_names],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        self.robot.set_dofs_kp([self.kp] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.kv] * self.num_actions, self.motors_dof_idx)

        self.scene.build(n_envs=num_envs)

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
            [env_cfg.joints[joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_qpos = torch.concatenate(
            (self.init_base_pos, self.init_base_quat, self.init_dof_pos)
        )
        self.init_projected_gravity: torch.Tensor = transform_by_quat(
            self.global_gravity, self.inv_base_init_quat
        )  # pyright: ignore

        self.buffers["base_pos"] = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.buffers["base_quat"] = torch.empty(
            (self.num_envs, 4), dtype=gs.tc_float, device=gs.device
        )
        self.buffers["base_euler"] = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.buffers["base_lin_vel"] = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.buffers["base_ang_vel"] = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.buffers["projected_gravity"] = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.buffers["actions"] = torch.zeros(
            (self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device
        )
        self.buffers["last_actions"] = torch.zeros(
            (self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device
        )
        self.buffers["dof_pos"] = torch.empty(
            (self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device
        )
        self.buffers["dof_vel"] = torch.empty(
            (self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device
        )
        self.buffers["last_dof_vel"] = torch.zeros(
            (self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device
        )
        self.buffers["default_dof_pos"] = torch.tensor(
            [env_cfg.joints[name] for name in self.joint_names],
            dtype=gs.tc_float,
            device=gs.device,
        )

    def step(self, actions: torch.Tensor, commands: torch.Tensor | None = None):
        self.buffers["actions"] = torch.clip(
            actions, -self.clip_actions, self.clip_actions
        )
        exec_actions = (
            self.buffers["last_actions"]
            if self.simulate_action_latency
            else self.buffers["actions"]
        )
        target_dof_pos = (
            exec_actions * self.action_scale + self.buffers["default_dof_pos"]
        )
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)

        self.scene.step()

        # update buffers
        self.buffers["base_pos"] = self.robot.get_pos()
        self.buffers["base_quat"] = self.robot.get_quat()
        self.buffers["base_euler"] = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.buffers["base_quat"]),
            rpy=True,
            degrees=False,
        )  # pyright: ignore
        inv_base_quat = inv_quat(self.buffers["base_quat"])
        self.buffers["base_lin_vel"] = transform_by_quat(
            self.robot.get_vel(), inv_base_quat
        )  # pyright: ignore
        self.buffers["base_ang_vel"] = transform_by_quat(
            self.robot.get_ang(), inv_base_quat
        )  # pyright: ignore
        self.buffers["projected_gravity"] = transform_by_quat(
            self.global_gravity, inv_base_quat
        )  # pyright: ignore
        self.buffers["dof_pos"] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.buffers["dof_vel"] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # visualize commanded and actual velocity
        self.scene.clear_debug_objects()

        if commands:
            cmd_vec = torch.zeros(3)
            cmd_vec[:2] = commands[0, :2]
            cmd_vec: torch.Tensor = transform_by_quat(
                cmd_vec, self.buffers["base_quat"][0, :]
            )  # type: ignore

            self.scene.draw_debug_arrow(
                self.buffers["base_pos"][0, :].cpu(),
                cmd_vec.cpu(),
                color=(0, 0, 1, 0.5),
            )

        self.scene.draw_debug_arrow(
            self.buffers["base_pos"][0, :].cpu(),
            self.buffers["base_lin_vel"][0, :].cpu(),
            color=(1, 0, 0, 0.5),
        )

        self.buffers["last_actions"].copy_(self.buffers["actions"])
        self.buffers["last_dof_vel"].copy_(self.buffers["dof_vel"])

    def reset(self, idx: torch.Tensor | None = None) -> None:
        # reset state
        self.robot.set_qpos(
            self.init_qpos, envs_idx=idx, zero_velocity=True, skip_forward=True
        )

        # reset buffers
        if idx is None:
            self.buffers["base_pos"].copy_(self.init_base_pos)
            self.buffers["base_quat"].copy_(self.init_base_quat)
            self.buffers["projected_gravity"].copy_(self.init_projected_gravity)
            self.buffers["dof_pos"].copy_(self.init_dof_pos)
            self.buffers["base_lin_vel"].zero_()
            self.buffers["base_ang_vel"].zero_()
            self.buffers["dof_vel"].zero_()
            self.buffers["actions"].zero_()
            self.buffers["last_actions"].zero_()
            self.buffers["last_dof_vel"].zero_()
            return
        else:
            torch.where(
                idx[:, None],
                self.init_base_pos,
                self.buffers["base_pos"],
                out=self.buffers["base_pos"],
            )
            torch.where(
                idx[:, None],
                self.init_base_quat,
                self.buffers["base_quat"],
                out=self.buffers["base_quat"],
            )
            torch.where(
                idx[:, None],
                self.init_projected_gravity,
                self.buffers["projected_gravity"],
                out=self.buffers["projected_gravity"],
            )
            torch.where(
                idx[:, None],
                self.init_dof_pos,
                self.buffers["dof_pos"],
                out=self.buffers["dof_pos"],
            )
            self.buffers["base_lin_vel"].masked_fill_(idx[:, None], 0.0)
            self.buffers["base_ang_vel"].masked_fill_(idx[:, None], 0.0)
            self.buffers["dof_vel"].masked_fill_(idx[:, None], 0.0)
            self.buffers["actions"].masked_fill_(idx[:, None], 0.0)
            self.buffers["last_actions"].masked_fill_(idx[:, None], 0.0)
            self.buffers["last_dof_vel"].masked_fill_(idx[:, None], 0.0)
