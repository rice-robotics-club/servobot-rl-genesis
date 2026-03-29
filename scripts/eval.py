import argparse
import pickle
import time
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
        "-i", "--input", type=str, default=None, choices=["keyboard", "gamepad", "fullspeedahead", "ninjamoves"], help="Input method to use (default: none)"
    )
    parser.add_argument(
        "--minecraft",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--slowmo",
        type=float,
        default=1.0,
        help="Playback speed multiplier (e.g. 0.5 = half speed, 0.1 = 10x slower)",
    )
    parser.add_argument(
        "--nre",
        type=int,
        default=1,
        dest="num_rendered_envs",
        help="Number of environments to render in the viewer (default: 1)",
    )
    args = parser.parse_args()

    gs.init(
        precision="32",
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
        args.num_rendered_envs,
        config["env"],
        config["obs"],
        config["reward"],
        config["commands"],
        num_rendered_envs=args.num_rendered_envs,
        **({"minecraft": True} if args.minecraft else {}),
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
    elif args.input == "fullspeedahead":
        from src.input import FullSpeedAhead

        print("Full speed ahead input initialized.")
        input = FullSpeedAhead()
    elif args.input == "ninjamoves":
        from src.input import NinjaMoves

        print("Ninja moves input initialized.")
        input = NinjaMoves()

    obs = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, _, _ = env.step(
                actions, input.command if input is not None else None
            )
            if args.slowmo != 1.0:
                extra = env.dt * (1.0 / args.slowmo - 1.0)
                end = time.monotonic() + extra
                while time.monotonic() < end:
                    if env.scene.viewer:
                        env.scene.viewer.update()
                    else:
                        time.sleep(env.dt)


if __name__ == "__main__":
    main()
