import argparse
import os
import pickle
import shutil
import yaml

from rsl_rl.runners import OnPolicyRunner
from src.kinematics import IK

import genesis as gs

from env import ServobotEnv


# def get_train_cfg(exp_name, max_iterations):
#     train_cfg_dict = {
#         "algorithm": {
#             "class_name": "PPO",
#             "clip_param": 0.2,
#             "desired_kl": 0.01,
#             "entropy_coef": 0.01,
#             "gamma": 0.99,
#             "lam": 0.95,
#             "learning_rate": 0.001,
#             "max_grad_norm": 1.0,
#             "num_learning_epochs": 5,
#             "num_mini_batches": 4,
#             "schedule": "adaptive",
#             "use_clipped_value_loss": True,
#             "value_loss_coef": 1.0,
#         },
#         "init_member_classes": {},
#         "policy": {
#             "activation": "elu",
#             "actor_hidden_dims": [512, 256, 128],
#             "critic_hidden_dims": [512, 256, 128],
#             "init_noise_std": 1.0,
#             "class_name": "ActorCritic",
#         },
#         "runner": {
#             "checkpoint": -1,
#             "experiment_name": exp_name,
#             "load_run": -1,
#             "log_interval": 1,
#             "max_iterations": max_iterations,
#             "record_interval": -1,
#             "resume": False,
#             "resume_path": None,
#             "run_name": "",
#         },
#         "runner_class_name": "OnPolicyRunner",
#         "num_steps_per_env": 24,
#         "save_interval": 100,
#         "empirical_normalization": None,
#         "seed": 1,
#     }
#
#     return train_cfg_dict


def get_cfgs():
    ik = IK()
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": ik.get_idle_cfg(),
        "joint_names": [
            "FL_Hip",
            "FL_TopLeg",
            "FL_BotLeg",
            "FR_Hip",
            "FR_TopLeg",
            "FR_BotLeg",
            "BL_Hip",
            "BL_TopLeg",
            "BL_BotLeg",
            "BR_Hip",
            "BR_TopLeg",
            "BR_BotLeg",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 20,  # degree
        "termination_if_pitch_greater_than": 20,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.18],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.18,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-0.5, 0.5],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-0.5, 0.5],
    }
    
    # Add symmetry configuration
    # Pairs: (left_idx, right_idx) where actions should be mirrored
    # FL <-> FR: (0,1,2) <-> (3,4,5)
    # BL <-> BR: (6,7,8) <-> (9,10,11)
    symmetry_cfg = {
        "symmetric_pairs": [
            [0, 3],   # FL_Hip <-> FR_Hip
            [1, 4],   # FL_TopLeg <-> FR_TopLeg
            [2, 5],   # FL_BotLeg <-> FR_BotLeg
            [6, 9],   # BL_Hip <-> BR_Hip
            [7, 10],  # BL_TopLeg <-> BR_TopLeg
            [8, 11],  # BL_BotLeg <-> BR_BotLeg
        ],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg, symmetry_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_cfg", type=str)
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=101)
    args = parser.parse_args()

    gs.init(logging_level="warning", )

    env_cfg, obs_cfg, reward_cfg, command_cfg, symmetry_cfg = get_cfgs()
    with open(args.train_cfg, 'r') as file:
        train_cfg = yaml.safe_load(file)
    log_dir = f"logs/{train_cfg["runner"]["experiment_name"]}"

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = ServobotEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
