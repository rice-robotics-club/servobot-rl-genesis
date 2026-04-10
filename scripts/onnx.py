import argparse
import pickle
from pathlib import Path

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from src.config import Config, EnvConfig, ObsConfig
from src.utils import get_latest


class NullEnv(VecEnv):
    """
    An environment for using RSL_RL without training.
    """

    def __init__(self, num_envs, env_cfg: EnvConfig, obs_cfg: ObsConfig):
        self.device = "cpu"
        self.cfg = env_cfg
        self.obs = obs_cfg
        self.num_envs = num_envs
        self.num_actions = len(env_cfg["joints"])
        self.max_episode_length = 1
        self.observations = TensorDict(
            {
                group: torch.empty((self.num_envs, num), dtype=torch.float64)
                for group, num in obs_cfg["num_obs"].items()
            },
            batch_size=(self.num_envs,),
        )
        self.rewards = torch.empty(
            (self.num_envs,), dtype=torch.float64, device=self.device
        )
        self.reset = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)

    def get_observations(self) -> TensorDict:
        return self.observations

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        return self.observations, self.rewards, self.reset, {}


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
