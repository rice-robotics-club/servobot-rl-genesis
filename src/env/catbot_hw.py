import genesis as gs
import torch
from genesis.engine.sensors.imu import IMUData, IMUSensor

from src.env.catbot import CatbotEnv


class CatbotHwEnv(CatbotEnv):
    """CatbotEnv with hardware-observable observations for direct (non-distillation) training.

    Replaces the privileged sim state (true ang_vel, projected_gravity) with IMU sensor
    readings that are available on the physical robot:
        imu_ang_vel(3) + imu_lin_acc(3) + commands(3) + dof_pos(12) + dof_vel(12) + actions(12) = 45

    Compatible with OnPolicyRunner and obs_groups: {actor: [main], critic: [main]}.
    Train from scratch — no teacher needed.

    IMU noise is controlled by domain_randomization.imu in the config (see catbot_distill.yaml
    for reference values). When omitted or when DR is disabled, the IMU is noiseless.
    """

    def _add_sensors_before_build(self) -> None:
        imu_cfg = self.dr_cfg.get("imu", {})

        # Default to zero noise so unrandomized phases get a clean IMU signal.
        acc_noise  = tuple(imu_cfg.get("acc_noise",         [0.0, 0.0, 0.0]))
        gyro_noise = tuple(imu_cfg.get("gyro_noise",        [0.0, 0.0, 0.0]))
        acc_rw     = tuple(imu_cfg.get("acc_random_walk",   [0.0, 0.0, 0.0]))
        gyro_rw    = tuple(imu_cfg.get("gyro_random_walk",  [0.0, 0.0, 0.0]))
        delay      = imu_cfg.get("delay",  0.0)
        jitter     = imu_cfg.get("jitter", 0.0)

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
        dof_pos = self.dof_pos - self.default_dof_pos
        dof_vel = self.dof_vel

        # IMU noise is handled by Genesis; only apply encoder/velocity noise here.
        if noise_cfg:
            if "dof_pos" in noise_cfg:
                dof_pos = dof_pos + torch.randn_like(dof_pos) * noise_cfg["dof_pos"]
            if "dof_vel" in noise_cfg:
                dof_vel = dof_vel + torch.randn_like(dof_vel) * noise_cfg["dof_vel"]

        self.obs_dict["main"] = torch.cat(
            (
                imu_data.ang_vel,                          # 3  — noisy gyro
                imu_data.lin_acc,                          # 3  — noisy accel (encodes gravity)
                self.commands * self.commands_scale,       # 3
                dof_pos * self.obs_scales["dof_pos"],      # 12
                dof_vel * self.obs_scales["dof_vel"],      # 12
                self.actions,                              # 12
            ),
            dim=-1,
        )
