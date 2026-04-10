import pickle
from pathlib import Path

import genesis as gs
import numpy as np
import torch

from src.config import Config
from src.utils import get_class


def main():
    gs.init(precision="32", logging_level="warning")

    # Load environment
    exp_dir = "savedlogs/catbot_leg_student"
    config = pickle.load(open(f"{exp_dir}/config.pkl", "rb"))

    env_class = get_class("src.env", config["env"]["class_name"])
    env = env_class(
        1,  # Single environment for debugging
        config["env"],
        config["obs"],
        config["reward"],
        config["commands"],
        headless=True,
    )

    print("=" * 80)
    print("YAW VELOCITY BIAS DIAGNOSTIC")
    print("=" * 80)

    # Test 1: Random actions, track correlations
    print("\nTest 1: Random Actions with Data Collection")
    print("-" * 80)

    num_episodes = 10
    all_yaw_vels = []
    all_commands = []
    all_actions = []
    all_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        episode_yaw_vels = []
        episode_commands = []
        episode_actions = []
        episode_rewards = []

        for step in range(50):
            # Random actions (like untrained agent)
            actions = torch.randn(1, env.num_actions) * 0.2

            obs, rewards, dones, extras = env.step(actions)

            episode_yaw_vels.append(env.yaw_vel[0, 0].item())
            episode_commands.append(env.commands[0, 0].item())
            episode_actions.append(actions[0, 0].item())  # Take first action
            episode_rewards.append(rewards[0].item())

            if dones[0]:
                break

        all_yaw_vels.extend(episode_yaw_vels)
        all_commands.extend(episode_commands)
        all_actions.extend(episode_actions)
        all_rewards.extend(episode_rewards)

    yaw_vels_array = np.array(all_yaw_vels)
    commands_array = np.array(all_commands)
    actions_array = np.array(all_actions)
    rewards_array = np.array(all_rewards)

    print(f"\nYaw Velocity Statistics:")
    print(f"  Mean:          {np.mean(yaw_vels_array):>10.6f}")
    print(f"  Std Dev:       {np.std(yaw_vels_array):>10.6f}")
    print(f"  Min:           {np.min(yaw_vels_array):>10.6f}")
    print(f"  Max:           {np.max(yaw_vels_array):>10.6f}")
    pos_count = np.sum(yaw_vels_array > 0)
    neg_count = np.sum(yaw_vels_array < 0)
    zero_count = np.sum(np.abs(yaw_vels_array) < 1e-6)
    total = len(yaw_vels_array)
    print(
        f"  Positive:      {pos_count:>6}/{total} ({100 * pos_count / total:>5.1f}%) [CLOCKWISE]"
    )
    print(
        f"  Negative:      {neg_count:>6}/{total} ({100 * neg_count / total:>5.1f}%) [COUNTER-CW]"
    )
    print(
        f"  Zero:          {zero_count:>6}/{total} ({100 * zero_count / total:>5.1f}%)"
    )

    print(f"\nCommand Statistics:")
    print(f"  Mean:          {np.mean(commands_array):>10.6f}")
    print(f"  Std Dev:       {np.std(commands_array):>10.6f}")
    print(f"  Min:           {np.min(commands_array):>10.6f}")
    print(f"  Max:           {np.max(commands_array):>10.6f}")

    # Correlation analysis
    print(f"\nCorrelation Analysis:")
    if len(all_commands) > 1:
        corr_yaw_cmd = np.corrcoef(yaw_vels_array, commands_array)[0, 1]
        print(f"  yaw_vel vs commands:  {corr_yaw_cmd:>10.6f}")
        if abs(corr_yaw_cmd) < 0.3:
            print(f"    ⚠️  LOW CORRELATION - yaw_vel not tracking commands properly!")

    corr_yaw_action = np.corrcoef(yaw_vels_array, actions_array)[0, 1]
    print(f"  yaw_vel vs actions:   {corr_yaw_action:>10.6f}")

    corr_cmd_action = np.corrcoef(commands_array, actions_array)[0, 1]
    print(f"  commands vs actions:  {corr_cmd_action:>10.6f}")

    # Test 2: Fixed positive command
    print("\n" + "=" * 80)
    print("Test 2: Fixed Positive Command")
    print("-" * 80)

    obs = env.reset()
    pos_yaws = []
    for _ in range(100):
        actions = torch.zeros(1, env.num_actions)
        obs, _, _, _ = env.step(actions, command=(1.0,))  # Full positive command
        pos_yaws.append(env.yaw_vel[0, 0].item())

    print(f"Command sent: +1.0 (max positive)")
    print(f"  Avg yaw_vel:  {np.mean(pos_yaws):>10.6f}")
    print(f"  Expected:     Should be POSITIVE (counter-clockwise)")
    if np.mean(pos_yaws) > 0:
        print(f"  ✓ CORRECT")
    else:
        print(f"  ✗ INVERTED - positive command gives negative yaw!")

    # Test 3: Fixed negative command
    print("\nTest 3: Fixed Negative Command")
    print("-" * 80)

    obs = env.reset()
    neg_yaws = []
    for _ in range(100):
        actions = torch.zeros(1, env.num_actions)
        obs, _, _, _ = env.step(actions, command=(-1.0,))  # Full negative command
        neg_yaws.append(env.yaw_vel[0, 0].item())

    print(f"Command sent: -1.0 (max negative)")
    print(f"  Avg yaw_vel:  {np.mean(neg_yaws):>10.6f}")
    print(f"  Expected:     Should be NEGATIVE (clockwise)")
    if np.mean(neg_yaws) < 0:
        print(f"  ✓ CORRECT")
    else:
        print(f"  ✗ INVERTED - negative command gives positive yaw!")

    # Test 4: Environment and reward configuration
    print("\n" + "=" * 80)
    print("Test 4: Environment Configuration")
    print("-" * 80)
    print(f"Joint names: {env.joint_names}")
    print(f"Num actions: {env.num_actions}")
    print(f"Command velocity range: {env.command_cfg['vel']}")
    print(f"Tracking sigma: {env.reward_cfg.get('tracking_sigma', 'N/A')}")
    print(f"\nYaw velocity shape: {env.yaw_vel.shape}")
    print(f"Commands shape: {env.commands.shape}")

    # Test 5: Check reward function computation
    print("\n" + "=" * 80)
    print("Test 5: Reward Function Analysis")
    print("-" * 80)

    # Simulate what the reward function does
    obs = env.reset()
    test_commands = np.linspace(-1, 1, 11)

    for cmd_val in test_commands:
        # Set up scenario
        for _ in range(10):  # Let it stabilize
            actions = torch.zeros(1, env.num_actions)
            obs, _, _, _ = env.step(actions, command=(cmd_val,))

        yaw = env.yaw_vel[0, 0].item()
        cmd = env.commands[0, 0].item()

        # Compute reward manually
        sigma = env.reward_cfg.get("tracking_sigma", 0.25)
        lin_vel_error = (cmd - yaw) ** 2
        reward = np.exp(-lin_vel_error / sigma)

        print(
            f"  Command: {cmd_val:>6.2f} → Yaw: {yaw:>8.4f} → Error: {lin_vel_error:>8.4f} → Reward: {reward:>8.4f}"
        )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Yaw velocity bias: {np.mean(yaw_vels_array):.6f}")
    if np.mean(yaw_vels_array) > 0.01:
        print("⚠️  STRONG CLOCKWISE BIAS DETECTED")
    elif np.mean(yaw_vels_array) < -0.01:
        print("⚠️  STRONG COUNTER-CLOCKWISE BIAS DETECTED")
    else:
        print("✓  No significant directional bias")

    print("=" * 80)


if __name__ == "__main__":
    main()
