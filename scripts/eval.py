import argparse
import pickle
from pathlib import Path

import genesis as gs
import torch
from rsl_rl.runners import OnPolicyRunner

from src.utils import get_class, get_latest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default=None,
        help="Experiment directory to load (default: logs/[latest])",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Model iteration file to load (default: (model_[max].pt))",
    )
    parser.add_argument(
        "-i", "--input", type=str, default=None, choices=["keyboard", "gamepad"]
    )
    parser.add_argument(
        "--minecraft",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    gs.init(
        logging_level="warning",
    )

    from src.env import GenesisEnv

    if args.experiment:
        exp_dir = get_latest(args.experiment, mode="exact")
    else:
        exp_dir = get_latest("logs")
    if not exp_dir:
        raise ValueError("No experiment directory found")

    if args.model:
        model_path = Path(exp_dir, args.model)
    else:
        model_path = get_latest(Path(exp_dir, "model_"))
    if not model_path:
        raise ValueError("No model file found")

    config = pickle.load(open(f"{exp_dir}/config.pkl", "rb"))

    env_class: type[GenesisEnv] | None = get_class(
        "src.env", config["env"]["class_name"]
    )
    if not env_class:
        return
    env = env_class(
        1, config["env"], config["obs"], config["reward"], config["commands"],
        minecraft=args.minecraft,
    )

    runner = OnPolicyRunner(
        env, config["runner"], str(model_path.parent), device=str(gs.device)
    )
    runner.load(str(model_path), map_location=str(gs.device))
    policy = runner.get_inference_policy(device=str(gs.device))

    input = None

    if args.input == "gamepad":
        from src.input import Gamepad
        print("Gamepad input initialized.")
        input = Gamepad()
    elif args.input == "keyboard":
        from src.input import Keyboard
        print("Keyboard input initialized.")
        input = Keyboard()



    obs = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, _, _ = env.step(
                actions, command=input.command if input else None
            )


if __name__ == "__main__":
    main()
