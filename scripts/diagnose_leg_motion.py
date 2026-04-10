import pickle
from pathlib import Path
import genesis as gs
import torch
import numpy as np
from src.config import Config
from src.utils import get_class

def main():
    gs.init(precision="32", logging_level="warning")

    # Load environment
    exp_dir = "savedlogs/catbot_leg_student"
    config = pickle.load(open(f"{exp_dir}/config.pkl", "rb"))

    env_class = get_class("src.env", config["env"]["class_name"])
    env = env_class(
        1,
        config["env"],
        config["obs"],
        config["reward"],
        config["commands"],
        headless=True,
    )

    print("=" * 100)
    print("LEG MOTION DIAGNOSTIC")
    print("=" * 100)
    print(f"\nJoint limits:")
    print(f"  'a': {env.dofs_limit[0, 0]:.4f} to {env.dofs_limit[0, 1]:.4f} rad")
    print(f"  'l': {env.dofs_limit[1, 0]:.4f} to {env.dofs_limit[1, 1]:.4f} rad")
    print(f"\nAction scale: {env.cfg['action_scale']}")
    print(f"Clip actions: {env.cfg['clip_actions']}")

    obs = env.reset()

    print("\n" + "=" * 100)
    print("STEP-BY-STEP MOTION DATA")
    print("=" * 100)
    print(f"\n{'Step':>5} | {'Cmd':>6} | {'Act_a':>7} | {'Act_l':>7} | {'Pos_a':>7} | {'Pos_l':>7} | {'Vel_a':>7} | {'Vel_l':>7} | {'Foot_z':>7}")
    print("-" * 100)

    steps_to_show = 100
    for step in range(steps_to_show):
        # Random actions for now - just to see what happens
        actions = torch.randn(1, 2) * 0.5

        obs, reward, done, extras = env.step(actions)

        cmd = env.commands[0, 0].item()
        act_a = actions[0, 0].item()
        act_l = actions[0, 1].item()
        pos_a = env.dof_pos[0, 0].item()
        pos_l = env.dof_pos[0, 1].item()
        vel_a = env.dof_vel[0, 0].item()
        vel_l = env.dof_vel[0, 1].item()
        foot_z = env.foot_pos[0, 0, 2].item() if env.foot_link_idx is not None else 0.0

        if step % 10 == 0:
            print(f"{step:>5} | {cmd:>6.3f} | {act_a:>7.3f} | {act_l:>7.3f} | {pos_a:>7.3f} | {pos_l:>7.3f} | {vel_a:>7.3f} | {vel_l:>7.3f} | {foot_z:>7.3f}")

    print("\n" + "=" * 100)
    print("STATISTICS")
    print("=" * 100)

    # Run another episode to collect stats
    obs = env.reset()
    pos_a_history = []
    pos_l_history = []
    vel_a_history = []
    vel_l_history = []
    foot_z_history = []

    for step in range(200):
        actions = torch.randn(1, 2) * 0.5
        obs, reward, done, extras = env.step(actions)

        pos_a_history.append(env.dof_pos[0, 0].item())
        pos_l_history.append(env.dof_pos[0, 1].item())
        vel_a_history.append(env.dof_vel[0, 0].item())
        vel_l_history.append(env.dof_vel[0, 1].item())
        foot_z_history.append(env.foot_pos[0, 0, 2].item() if env.foot_link_idx is not None else 0.0)

    pos_a = np.array(pos_a_history)
    pos_l = np.array(pos_l_history)
    vel_a = np.array(vel_a_history)
    vel_l = np.array(vel_l_history)
    foot_z = np.array(foot_z_history)

    print(f"\nJoint 'a' (angle joint):")
    print(f"  Position range: {pos_a.min():.4f} to {pos_a.max():.4f} rad")
    print(f"  Position std: {pos_a.std():.4f} rad")
    print(f"  Velocity range: {vel_a.min():.4f} to {vel_a.max():.4f} rad/s")
    print(f"  Velocity std: {vel_a.std():.4f} rad/s")
    print(f"  Motion type: {'OSCILLATING' if pos_a.std() > 0.1 else 'STUCK/BARELY MOVING'}")

    print(f"\nJoint 'l' (length joint):")
    print(f"  Position range: {pos_l.min():.4f} to {pos_l.max():.4f} rad")
    print(f"  Position std: {pos_l.std():.4f} rad")
    print(f"  Velocity range: {vel_l.min():.4f} to {vel_l.max():.4f} rad/s")
    print(f"  Velocity std: {vel_l.std():.4f} rad/s")
    print(f"  Motion type: {'OSCILLATING' if pos_l.std() > 0.1 else 'STUCK/BARELY MOVING'}")

    print(f"\nFoot motion:")
    print(f"  Height range: {foot_z.min():.4f} to {foot_z.max():.4f} m")
    print(f"  Height std: {foot_z.std():.4f} m")
    print(f"  Foot cycling: {'YES - leaving ground' if foot_z.max() - foot_z.min() > 0.01 else 'NO - always in contact'}")

    print("\n" + "=" * 100)

if __name__ == "__main__":
    main()

