"""
Test the stretching pain reward by sweeping t_l while holding t_a fixed,
and observing how pain responds as t_c approaches its limits.

Run with:
    python scripts/test_stretching_pain.py
"""

import math
from pathlib import Path

import genesis as gs
import torch

gs.init(precision="32", logging_level="warning", backend=gs.cpu)


def main():
    tc_min_deg = math.degrees(-1.22173)   # -70°
    tc_max_deg = math.degrees(1.047198)   #  60°
    threshold_deg = 5.0

    print("=" * 60)
    print(f"tc_min={tc_min_deg:.1f}°  tc_max={tc_max_deg:.1f}°  danger zone=±{threshold_deg}°")
    print()

    # --- Standalone pain curve ---
    threshold = threshold_deg * math.pi / 180.0
    print("Pain curve:")
    for deg in [10, 5, 4, 3, 2, 1, 0]:
        dist = deg * math.pi / 180.0
        pain = 0.0 if dist >= threshold else math.exp(((threshold - dist) / threshold) * 6) - 1
        print(f"  {deg:4.1f}° from limit → {pain:.1f}")
    print()

    # --- Spin up env ---
    config_path = Path("config/hw_obs/unrandomized/catbot_phase5_natural_gait.yaml")
    if not config_path.exists():
        candidates = [c for c in Path("config").rglob("catbot*.yaml") if "schema" not in str(c)]
        config_path = candidates[0]

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from src.env.catbot import CatbotEnv
    env = CatbotEnv(
        num_envs=1,
        env_cfg=config["env"],
        obs_cfg=config["obs"],
        reward_cfg=config["reward"],
        command_cfg=config["commands"],
        headless=True,
        domain_rand_cfg=None,
    )
    env.reset()

    a_indices = [i for i, n in enumerate(env.joint_names) if n.split("_")[-1] == "a" or n == "a"]
    l_indices = [i for i, n in enumerate(env.joint_names) if n.split("_")[-1] == "l" or n == "l"]

    def set_and_measure(dof_a_rad, dof_l_rad):
        dof_pos = env.robot.get_dofs_position(env.motors_dof_idx).clone()
        for i in a_indices:
            dof_pos[:, i] = dof_a_rad
        for i in l_indices:
            dof_pos[:, i] = dof_l_rad
        env.robot.set_dofs_position(dof_pos, env.motors_dof_idx)
        env.scene.step()
        env.dof_pos = env.robot.get_dofs_position(env.motors_dof_idx)
        pain = env._reward_stretching_pain()
        return pain.item()

    # Hold t_a at mid-range (dof=pi/2 → t_a_deg=0), sweep t_l
    dof_a = math.pi / 2
    print(f"Sweeping t_l (dof_pos) with t_a fixed at {math.degrees(dof_a):.1f}° (dof_pos={dof_a:.3f} rad)")
    print(f"  {'dof_l_rad':>10}  {'t_l_deg_reward':>15}  {'pain':>8}")
    print("  " + "-" * 40)

    prev_pain = None
    for dof_l_deg in range(-180, 181, 10):
        dof_l = dof_l_deg * math.pi / 180.0
        pain = set_and_measure(dof_a, dof_l)
        t_l_deg_reward = dof_l_deg - 90  # what reward sees: dof_pos*(180/pi)-90
        flag = ""
        if prev_pain is not None and pain > 0 and prev_pain == 0:
            flag = " ← enters danger zone"
        if prev_pain is not None and pain == 0 and prev_pain > 0:
            flag = " ← exits danger zone"
        if pain > 0:
            print(f"  {dof_l:>10.3f}  {t_l_deg_reward:>15.1f}  {pain:>8.2f}{flag}")
        else:
            print(f"  {dof_l:>10.3f}  {t_l_deg_reward:>15.1f}  {pain:>8.2f}{flag}")
        prev_pain = pain

    print()
    print("Expected: pain=0 across most of the sweep, spiking near the extremes.")


if __name__ == "__main__":
    main()
