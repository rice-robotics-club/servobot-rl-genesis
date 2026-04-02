import argparse
import pickle
from pathlib import Path

from rsl_rl.runners import OnPolicyRunner

from src.config import Config
from src.env import NullEnv
from src.utils import get_latest


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
    args = parser.parse_args()

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

    config: Config = pickle.load(open(f"{exp_dir}/config.pkl", "rb"))

    print(f"config: {config}")

    runner = OnPolicyRunner(
        NullEnv(1, config["env"], config["obs"]),
        config["runner"],  # type: ignore
        str(model_path.parent),
    )
    runner.load(str(model_path))
    runner.export_policy_to_onnx(str(model_path.parent), filename="policy.onnx")


if __name__ == "__main__":
    main()
