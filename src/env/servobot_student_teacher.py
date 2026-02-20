
import tensordict
import torch
import genesis as gs

from env.servobot import ServobotEnv


class ServobotStudentTeacherEnv(ServobotEnv):
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
        self.num_teacher_obs = obs_cfg["num_teacher_obs"]
        self.teacher_buf = torch.zeros(self.num_envs, self.num_teacher_obs, device=gs.device)
        self.obs_dict = tensordict.TensorDict(
            {"policy": self.policy_buf, "teacher": self.teacher_buf}, batch_size=[self.num_envs], device=gs.device
        )

    def update_observations(self):
        self.policy_buf = torch.concatenate(
            (
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos)
                * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ),
            dim=-1,
        )
        self.teacher_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel_z"],  # 3
                self.projected_gravity, # 3
            ),
            dim=-1,
        )
        
        self.obs_dict["policy"] = self.policy_buf
        self.obs_dict["teacher"] = self.teacher_buf