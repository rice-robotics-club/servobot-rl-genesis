import argparse
import pickle
from pathlib import Path
import os
import torch

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

    # Export policy to ONNX
    onnx_model = runner.alg.get_policy().as_onnx(verbose=False).double()
    onnx_model.to("cpu")
    onnx_model.eval()

    save_path = os.path.join(exp_dir, "polic.onnx")

    torch.onnx.export(
        onnx_model,
        onnx_model.get_dummy_inputs(),  # type: ignore
        save_path,
        export_params=True,
        opset_version=18,
        verbose=False,
        input_names=onnx_model.input_names,  # type: ignore
        output_names=onnx_model.output_names,  # type: ignore
    )


if __name__ == "__main__":
    main()
