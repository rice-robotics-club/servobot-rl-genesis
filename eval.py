import argparse
import os
import pickle
import torch
import pygame

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from env import ServobotEnv
from src.controls import Controller


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint to load (default: logs/servobot/model_100.pt)")
    parser.add_argument("-t", "--teleop", type=str, default="none", choices=["keyboard", "xbox", "ps4"])
    args = parser.parse_args()

    
    gs.init()

    ckpt_dir = os.path.dirname(args.ckpt) if args.ckpt else "logs/servobot"
    
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{ckpt_dir}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}
    
    
    env = ServobotEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    if args.ckpt is None:
        resume_path = "logs/servobot/model_100.pt"
    else:
        resume_path = args.ckpt
    runner = OnPolicyRunner(env, train_cfg, ckpt_dir, device=gs.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)


    controller = Controller(type=args.teleop)
    controller.initialize()

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            
            command = controller.get_command()
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions, command=command)


if __name__ == "__main__":
    main()
