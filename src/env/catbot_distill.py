import genesis as gs
import torch
from genesis.engine.sensors.imu import IMUData, IMUSensor
from tensordict import TensorDict

from src.config import CommandConfig, DomainRandConfig, EnvConfig, ObsConfig, RewardConfig
from src.env.catbot import CatbotEnv


class CatbotDistillEnv(CatbotEnv):
    """CatbotEnv variant with separate teacher/student observation groups for distillation.

    Teacher (45-dim): full privileged obs matching phase7 PPO training exactly.
        ang_vel(3) + projected_gravity(3) + commands(3) + dof_pos(12) + dof_vel(12) + actions(12)

    Student (45-dim): same structure but substitutes IMU sensor readings for privileged state.
        imu_ang_vel(3) + imu_lin_acc(3) + commands(3) + dof_pos(12) + dof_vel(12) + actions(12)
    """

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
        teacher_dim = obs_cfg["num_obs"]["teacher"]
        student_dim = obs_cfg["num_obs"]["student"]

        # Parent expects a scalar num_obs; give it the teacher size so it
        # initialises its own buffers correctly.
        parent_obs_cfg = {**obs_cfg, "num_obs": teacher_dim}
        super().__init__(
            num_envs, env_cfg, parent_obs_cfg, reward_cfg, command_cfg,
            headless, debug, domain_rand_cfg, **kwargs,
        )

        # Replace the parent's single-group obs_dict with teacher/student groups.
        self.num_obs = obs_cfg["num_obs"]
        self.obs_dict = TensorDict(
            {
                "teacher": torch.empty(
                    (num_envs, teacher_dim), dtype=gs.tc_float, device=gs.device
                ),
                "student": torch.empty(
                    (num_envs, student_dim), dtype=gs.tc_float, device=gs.device
                ),
            },
            batch_size=(num_envs,),
        )

    def _add_sensors_before_build(self) -> None:
        """Attach a simulated IMU to the robot base for student observations.

        Noise parameters are read from domain_randomization.imu in the config.
        """
        imu_cfg = {}
        if self.dr_cfg.get("enabled", False):
            imu_cfg = self.dr_cfg.get("imu", {})

        acc_noise  = tuple(imu_cfg.get("acc_noise",         [0.05, 0.05, 0.05]))
        gyro_noise = tuple(imu_cfg.get("gyro_noise",        [0.02, 0.02, 0.02]))
        acc_rw     = tuple(imu_cfg.get("acc_random_walk",   [0.001, 0.001, 0.001]))
        gyro_rw    = tuple(imu_cfg.get("gyro_random_walk",  [0.001, 0.001, 0.001]))
        delay      = imu_cfg.get("delay",  0.02)
        jitter     = imu_cfg.get("jitter", 0.005)

        base_link = self.robot.get_link("chassisasm")
        self.imu: IMUSensor = self.scene.add_sensor(
            gs.sensors.IMU(
                entity_idx=self.robot.idx,
                link_idx_local=base_link.idx_local,
                pos_offset=(0.0, 0.0, 0.0),
                acc_noise=acc_noise,
                gyro_noise=gyro_noise,
                acc_random_walk=acc_rw,
                gyro_random_walk=gyro_rw,
                delay=delay,
                jitter=jitter,
                interpolate=True,
            )
        )

    def _update_observation(self):
        imu_data: IMUData = self.imu.read()

        noise_cfg = (
            self.dr_cfg.get("obs_noise", {})
            if self.dr_cfg.get("enabled", False)
            else {}
        )
        ang_vel = self.base_ang_vel
        dof_pos = self.dof_pos - self.default_dof_pos
        dof_vel = self.dof_vel

        if noise_cfg:
            if "ang_vel" in noise_cfg:
                ang_vel = ang_vel + torch.randn_like(ang_vel) * noise_cfg["ang_vel"]
            if "dof_pos" in noise_cfg:
                dof_pos = dof_pos + torch.randn_like(dof_pos) * noise_cfg["dof_pos"]
            if "dof_vel" in noise_cfg:
                dof_vel = dof_vel + torch.randn_like(dof_vel) * noise_cfg["dof_vel"]

        commands = self.commands * self.commands_scale
        ang_vel_scaled = ang_vel * self.obs_scales["ang_vel_z"]
        dof_pos_scaled = dof_pos * self.obs_scales["dof_pos"]
        dof_vel_scaled = dof_vel * self.obs_scales["dof_vel"]

        # Teacher: exact 45-dim obs the PPO actor was trained on.
        self.obs_dict["teacher"] = torch.cat(
            (
                ang_vel_scaled,         # 3  — privileged true ang_vel
                self.projected_gravity, # 3  — privileged true orientation
                commands,               # 3
                dof_pos_scaled,         # 12
                dof_vel_scaled,         # 12
                self.actions,           # 12
            ),
            dim=-1,
        )

        # Student: 45-dim — swaps privileged state for IMU sensor readings.
        self.obs_dict["student"] = torch.cat(
            (
                imu_data.ang_vel,       # 3  — noisy gyro
                imu_data.lin_acc,       # 3  — noisy accelerometer (encodes gravity direction)
                commands,               # 3
                dof_pos_scaled,         # 12
                dof_vel_scaled,         # 12
                self.actions,           # 12
            ),
            dim=-1,
        )
