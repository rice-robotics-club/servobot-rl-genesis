import argparse
import os
import pickle
import torch
import pygame

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from env import ServobotEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="servobot")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("-j", "--joystick", action="store_true")
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = ServobotEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)



    obs, _ = env.reset()
    with torch.no_grad():
        if args.joystick:
            pygame.init()
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            print(f"Initialized joystick: {joystick.get_name()}")

        while True:
            if args.joystick:
                for _ in pygame.event.get():
                    pass

            command = (
                -joystick.get_axis(2),
                joystick.get_axis(3),
                -joystick.get_axis(0)
            ) if args.joystick else None

            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions, command=command)


if __name__ == "__main__":
    main()
