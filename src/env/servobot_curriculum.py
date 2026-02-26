
from env.servobot import ServobotEnv


class ServobotCurriculumEnv(ServobotEnv):
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        headless: bool = False,
        debug: bool = False,
        minecraft: bool = False,
    ):
        super().__init__(
            num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, headless, debug, minecraft
        )
        self.curriculum_enabled = env_cfg.get("curriculum_enabled", True)
        self.curriculum_steps = env_cfg.get("curriculum_steps", 1000)
        self.current_step = 0


''' 
Curriculum.yaml
 
config1:
    name: "Easy"
    env: 
        blah blah blah
    reward:
        blah blah blah
        value 
        
    commands:
        blah blah blah
'''