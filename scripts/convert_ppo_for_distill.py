"""Convert a phase7-style ActorCritic checkpoint into a format the
DistillationRunner can load as the teacher.

Usage:
    python scripts/convert_ppo_for_distill.py \
        savedlogs/catbot_phase7/model_22600.pt \
        savedlogs/catbot_phase7/teacher_22600.pt
"""

import argparse
import torch


def convert(src_path: str, dst_path: str) -> None:
    src = torch.load(src_path, map_location="cpu")
    model_sd = src["model_state_dict"]

    teacher_sd = {}
    for k, v in model_sd.items():
        if k.startswith("actor."):
            # actor.0.weight -> mlp.0.weight
            teacher_sd["mlp." + k[len("actor."):]] = v
        elif k.startswith("actor_obs_normalizer."):
            # actor_obs_normalizer._mean -> obs_normalizer._mean
            teacher_sd["obs_normalizer." + k[len("actor_obs_normalizer."):]] = v
        elif k == "std":
            # old rsl_rl std param -> distribution.std_param
            teacher_sd["distribution.std_param"] = v
        # drop critic.*, etc.

    dst = {
        "actor_state_dict": teacher_sd,
        "iter": src.get("iter", 0),
        "infos": None,
    }
    torch.save(dst, dst_path)
    print(f"Saved teacher checkpoint to {dst_path}")
    print(f"Keys: {list(teacher_sd.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="Source PPO checkpoint (.pt)")
    parser.add_argument("dst", help="Output teacher checkpoint (.pt)")
    args = parser.parse_args()
    convert(args.src, args.dst)
