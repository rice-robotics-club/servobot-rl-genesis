import argparse
import os
import pickle
import shutil
import yaml
from datetime import datetime

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
        "kv": 0.5,
        # termination
        "termination_if_roll_greater_than": 45,  # degree --- WAY HIGHER NOW! RUN MY BEAUTIFUL CREATURE, RUN
        "termination_if_pitch_greater_than": 45,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.18],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "domain_rand": {
            "kp_range": [15.0, 25.0],
            "kv_range": [0.3, 0.7],
            "friction_range": [0.5, 1.5],
            "payload_range": [[-0.05, -0.05, 0.0, 0.0], [0.05, 0.05, 0.1, 0.2]],  # x, y, z, mass(kg)
            "motor_strength_range": [0.8, 1.2],
        }
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
            "tracking_lin_vel": 1.75,
            "tracking_ang_vel": 0.75,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "energy": -0.0001,
            "survival": 0.3,
        },
    }
    command_cfg = {
        "num_commands": 3, #bigger range to teach faster gait at the cost of longer trainings
        "lin_vel_x_range": [-1.0, 1.0],
        "lin_vel_y_range": [-1.0, 1.0],
        "ang_vel_range": [-0.8, 0.8], 
    }
    
    # Add symmetry configuration
    # Pairs: (left_idx, right_idx) where actions should be mirrored
    # FL <-> FR: (0,1,2) <-> (3,4,5)
    # BL <-> BR: (6,7,8) <-> (9,10,11)
    # I have no idea how to properly give this information to RSL-RL so this is just sort of an unused variable right now
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
    parser.add_argument("-r", "--resume", type=str, default=None, 
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Custom directory name for saving (default: auto-generated with timestamp)")
    args = parser.parse_args()

    gs.init(logging_level="warning", )

    env_cfg, obs_cfg, reward_cfg, command_cfg, symmetry_cfg = get_cfgs()
    with open(args.train_cfg, 'r') as file:
        train_cfg = yaml.safe_load(file)
    
    exp_name = train_cfg["runner"]["experiment_name"]
    
    # Determine log directory
    if args.save_dir:
        # Use custom directory name
        log_dir = f"logs/{args.save_dir}"
    else:
        # Auto-generate with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if args.resume:
            # Extract original exp name and add "resumed" suffix
            original_dir = os.path.dirname(args.resume)
            original_name = os.path.basename(original_dir)
            log_dir = f"logs/{original_name}_resumed_{timestamp}"
        else:
            log_dir = f"logs/{exp_name}_{timestamp}"
    
    # Create new directory (never overwrite)
    if os.path.exists(log_dir):
        print(f"Warning: {log_dir} already exists. Adding timestamp suffix.")
        log_dir = f"{log_dir}_{datetime.now().strftime('%H%M%S')}"
    
    os.makedirs(log_dir, exist_ok=True)
    print(f"Saving to: {log_dir}")
    
    # Copy configs for reproducibility
    os.makedirs(f"{log_dir}/configs", exist_ok=True)
    shutil.copy(args.train_cfg, f"{log_dir}/configs/train.yaml")

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "resumed_from": args.resume,
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
    }
    with open(f"{log_dir}/metadata.yaml", 'w') as f:
        yaml.dump(metadata, f)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = ServobotEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
        show_viewer=True, num_viewer_envs=100
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    
    # Load checkpoint if resuming
    if args.resume:
        print(f"Loading checkpoint from: {args.resume}")
        runner.load(args.resume)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    print("="*60, "\n Training complete! \n Saved robot policy to:", log_dir, "\n", "="*60)

if __name__ == "__main__":
    main()
