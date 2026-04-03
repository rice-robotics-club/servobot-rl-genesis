"""
Migrate old rsl_rl checkpoints (ActorCritic schema) to new rsl_rl 5.0 schema (MLPModel).

Old state dict keys:
  actor.N.weight/bias         -> mlp.N.weight/bias          (in actor model)
  actor_obs_normalizer.*      -> obs_normalizer.*            (in actor model)
  critic.N.weight/bias        -> mlp.N.weight/bias          (in critic model)
  critic_obs_normalizer.*     -> obs_normalizer.*            (in critic model)
  std                         -> distribution.std_param      (in actor model)

Old config.pkl runner schema:
  policy.class_name = ActorCritic
  policy.actor_hidden_dims / critic_hidden_dims
  obs_groups: {"policy": [...], "critic": [...]}

New runner schema:
  actor.class_name = MLPModel
  critic.class_name = MLPModel
  obs_groups: {"actor": [...], "critic": [...]}
"""

import argparse
import pickle
import shutil
from pathlib import Path

import torch


def migrate_state_dict(old: dict) -> tuple[dict, dict]:
    """Split old combined model_state_dict into separate actor/critic dicts."""
    actor = {}
    critic = {}
    for k, v in old.items():
        if k == "std":
            actor["distribution.std_param"] = v
        elif k.startswith("actor_obs_normalizer."):
            actor["obs_normalizer." + k[len("actor_obs_normalizer."):]] = v
        elif k.startswith("critic_obs_normalizer."):
            critic["obs_normalizer." + k[len("critic_obs_normalizer."):]] = v
        elif k.startswith("actor."):
            actor["mlp." + k[len("actor."):]] = v
        elif k.startswith("critic."):
            critic["mlp." + k[len("critic."):]] = v
    return actor, critic


def migrate_runner_cfg(runner_cfg: dict) -> dict:
    cfg = dict(runner_cfg)
    if "policy" in cfg and "actor" not in cfg:
        old = cfg.pop("policy")
        hidden_dims = old.get("actor_hidden_dims", [512, 256, 128])
        activation = old.get("activation", "elu")
        cfg["actor"] = {
            "class_name": "MLPModel",
            "activation": activation,
            "obs_normalization": old.get("actor_obs_normalization", True),
            "hidden_dims": hidden_dims,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": old.get("init_noise_std", 1.0),
                "std_type": "scalar",
            },
        }
        cfg["critic"] = {
            "class_name": "MLPModel",
            "activation": activation,
            "obs_normalization": old.get("critic_obs_normalization", True),
            "hidden_dims": old.get("critic_hidden_dims", hidden_dims),
        }
    if "obs_groups" in cfg and "policy" in cfg["obs_groups"]:
        cfg["obs_groups"]["actor"] = cfg["obs_groups"].pop("policy")
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Migrate old rsl_rl checkpoint to new schema")
    parser.add_argument("exp_dir", type=str, help="Experiment directory (e.g. logs/catbot_phase6_...)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without writing")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        raise SystemExit(f"Directory not found: {exp_dir}")

    # --- Migrate config.pkl ---
    config_path = exp_dir / "config.pkl"
    if not config_path.exists():
        raise SystemExit(f"No config.pkl found in {exp_dir}")

    config = pickle.load(open(config_path, "rb"))
    old_runner_keys = list(config["runner"].keys())
    config["runner"] = migrate_runner_cfg(config["runner"])
    new_runner_keys = list(config["runner"].keys())

    print(f"config.pkl runner keys: {old_runner_keys} -> {new_runner_keys}")

    if not args.dry_run:
        shutil.copy(config_path, config_path.with_suffix(".pkl.bak"))
        pickle.dump(config, open(config_path, "wb"))
        print(f"  Wrote {config_path}  (backup: config.pkl.bak)")

    # --- Migrate all model_*.pt checkpoints ---
    checkpoints = sorted(exp_dir.glob("model_*.pt"))
    if not checkpoints:
        print("No model_*.pt files found.")
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        old_keys = list(ckpt["model_state_dict"].keys())
        actor_sd, critic_sd = migrate_state_dict(ckpt["model_state_dict"])
        del ckpt["model_state_dict"]
        ckpt["actor_state_dict"] = actor_sd
        ckpt["critic_state_dict"] = critic_sd
        print(f"\n{ckpt_path.name}:")
        for k in old_keys:
            if k == "std":
                print(f"  model_state_dict[{k}]  ->  actor_state_dict[distribution.std_param]")
            elif k.startswith("actor"):
                suffix = k[len("actor_obs_normalizer."):] if k.startswith("actor_obs_normalizer.") else "mlp." + k[len("actor."):]
                print(f"  model_state_dict[{k}]  ->  actor_state_dict[{suffix}]")
            elif k.startswith("critic"):
                suffix = k[len("critic_obs_normalizer."):] if k.startswith("critic_obs_normalizer.") else "mlp." + k[len("critic."):]
                print(f"  model_state_dict[{k}]  ->  critic_state_dict[{suffix}]")
        if not args.dry_run:
            shutil.copy(ckpt_path, ckpt_path.with_suffix(".pt.bak"))
            torch.save(ckpt, ckpt_path)
            print(f"  Saved  (backup: {ckpt_path.name}.bak)")


if __name__ == "__main__":
    main()
