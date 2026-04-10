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
        1,  # Single environment
        config["env"],
        config["obs"],
        config["reward"],
        config["commands"],
        headless=True,
    )

    print("=" * 80)
    print("SIMPLIFIED YAW VELOCITY SIGN TEST")
    print("=" * 80)

    # First, reset and take one step to initialize all buffers properly
    print("\n[1] Initial State (After Reset)")
    obs = env.reset()
    actions = torch.zeros(1, env.num_actions)
    obs, _, _, _ = env.step(actions)
    # Now yaw_vel should be properly initialized
    print(f"    Yaw velocity shape: {env.yaw_vel.shape}")
    print(f"    Yaw velocity: {env.yaw_vel.squeeze().item():.6f}")
    print(f"    Yaw joint index: {env.yaw_idx}")
    print(f"    Command: {env.commands[0, 0].item():.6f}")

    # Test positive command with zero actions
    print("\n[2] Positive Command +0.5 (Zero Actions)")
    obs = env.reset()
    yaw_vels_pos = []
    for i in range(20):
        actions = torch.zeros(1, env.num_actions)
        obs, rewards, dones, extras = env.step(actions, command=(0.5,))
        yaw_vel = env.yaw_vel.squeeze().item()
        if i < 5 or i >= 15:
            print(
                f"    Step {i:2d}: yaw_vel={yaw_vel:>8.5f}, cmd={env.commands[0, 0].item():>7.4f}"
            )
        elif i == 5:
            print(f"    ...")
        yaw_vels_pos.append(yaw_vel)

    mean_yaw_pos = np.mean(yaw_vels_pos)
    print(f"    Average yaw velocity: {mean_yaw_pos:.6f}")
    print(f"    Expected: POSITIVE (command is +0.5)")
    if mean_yaw_pos > 0.001:
        print(f"    ✓ CORRECT")
    else:
        print(f"    ✗ WRONG (expected positive, got {mean_yaw_pos:.6f})")

    # Test negative command with zero actions
    print("\n[3] Negative Command -0.5 (Zero Actions)")
    obs = env.reset()
    yaw_vels_neg = []
    for i in range(20):
        actions = torch.zeros(1, env.num_actions)
        obs, rewards, dones, extras = env.step(actions, command=(-0.5,))
        yaw_vel = env.yaw_vel.squeeze().item()
        if i < 5 or i >= 15:
            print(
                f"    Step {i:2d}: yaw_vel={yaw_vel:>8.5f}, cmd={env.commands[0, 0].item():>7.4f}"
            )
        elif i == 5:
            print(f"    ...")
        yaw_vels_neg.append(yaw_vel)

    mean_yaw_neg = np.mean(yaw_vels_neg)
    print(f"    Average yaw velocity: {mean_yaw_neg:.6f}")
    print(f"    Expected: NEGATIVE (command is -0.5)")
    if mean_yaw_neg < -0.001:
        print(f"    ✓ CORRECT")
    else:
        print(f"    ✗ WRONG (expected negative, got {mean_yaw_neg:.6f})")

    # Test with random actions to see if actions affect yaw
    print("\n[4] Effect of Actions on Yaw (Positive Command)")
    obs = env.reset()
    print("    Testing different action values...")
    for action_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        yaw_samples = []
        for _ in range(10):
            actions = torch.tensor([[action_val, action_val]])
            obs, _, _, _ = env.step(actions, command=(0.5,))
            yaw_samples.append(env.yaw_vel.squeeze().item())
        avg_yaw = np.mean(yaw_samples)
        print(
            f"    Actions=[{action_val:>4.1f}, {action_val:>4.1f}] → avg yaw={avg_yaw:>8.5f}"
        )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Positive command (+0.5) → yaw velocity: {mean_yaw_pos:>8.6f}")
    print(f"Negative command (-0.5) → yaw velocity: {mean_yaw_neg:>8.6f}")

    if mean_yaw_pos > 0.001 and mean_yaw_neg < -0.001:
        print("\n✓ YAW VELOCITY SIGN IS CORRECT")
    elif mean_yaw_pos < -0.001 and mean_yaw_neg > 0.001:
        print("\n✗ YAW VELOCITY SIGN IS INVERTED")
        print("  Need to invert the sign again or check joint direction")
    else:
        print("\n⚠️  YAW VELOCITY NOT RESPONDING TO COMMANDS")
        print("  Yaw joint may not be actuated or may be locked")
        print(
            f"  Both commands resulted in small velocities: {mean_yaw_pos:.6f} vs {mean_yaw_neg:.6f}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
