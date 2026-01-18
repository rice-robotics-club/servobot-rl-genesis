import argparse
import os
import pickle
import shutil
import time
from datetime import datetime

import genesis as gs
import yaml
from rsl_rl.runners import OnPolicyRunner

from src.config import load_config
from src.utils import get_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="config/servobot.yaml",
        help="Config file path to use (default: config/servobot.yaml)",
    )
    parser.add_argument(
        "-n",
        "--num_envs",
        type=int,
        default=4096,
        help="Number of environments to use (default: 4096)",
    )
    parser.add_argument(
        "-m",
        "--max_iterations",
        type=int,
        default=10000,
        help="Maximum number of iterations to run (default: 10000)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        default=None,
        help="Custom directory name for saving logs (default: auto-generated with timestamp)",
    )
    parser.add_argument("--headless", action="store_true", help="trains without GUI")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Loads config and views initial configuration",
    )
    parser.add_argument(
        "--no_print",
        action="store_true",
        help="Disables RSL_RL training info",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Prints environment debug information",
    )
    # parser.add_argument(
    #     "--randomize", action="store_true", help="Enable domain randomization"
    # )
    args = parser.parse_args()

    gs.init(
        logging_level="warning",
    )

    from src.env import GenesisEnv

    config = load_config(args.config_path)

    exp_name = config["runner"]["experiment_name"]

    # Determine log directory
    if args.save_dir:
        # Use custom directory name
        log_dir = f"logs/{args.save_dir}"
    else:
        # Auto-generate with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if args.resume:
            # Extract original exp name and add "resumed" suffix
            original_dir = os.path.dirname(args.resume)
            original_name = os.path.basename(original_dir)
            log_dir = f"logs/{original_name}_resumed_{timestamp}"
        else:
            log_dir = f"logs/{exp_name}_{timestamp}"

    # Create new directory (never overwrite)
    if os.path.exists(log_dir):
        print(f"Warning: {log_dir} already exists. Adding timestamp suffix.")
        log_dir = f"{log_dir}_{datetime.now().strftime('%H%M%S')}"

    os.makedirs(log_dir, exist_ok=True)
    print(f"Saving to: {log_dir}")

    # Copy configs for reproducibility
    shutil.copy(args.config_path, f"{log_dir}/config.yaml")

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "resumed_from": args.resume,
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
    }
    with open(f"{log_dir}/metadata.yaml", "w") as f:
        yaml.dump(metadata, f)

    with open(f"{log_dir}/config.pkl", "wb") as f:
        pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)

    # # load model via rsl-rl OR initialize fresh one
    # if load_existing_log:
    #     model = model.load(load_existing_log)
    # else:
    #     model = model.initialize(configs)

    env_class: type[GenesisEnv] | None = get_class(
        "src.env", config["env"]["class_name"]
    )
    if not env_class:
        return
    env = env_class(
        args.num_envs,
        config["env"],
        config["obs"],
        config["reward"],
        config["commands"],
        args.headless,
        args.debug,
    )

    runner_class: type[OnPolicyRunner] | None = get_class(
        "rsl_rl.runners", config["runner"]["class_name"]
    )
    if not runner_class:
        return
    runner = runner_class(env, config["runner"], log_dir, device=gs.device)  # pyright: ignore

    if args.no_print:
        runner.logger.disable_logs = True

    if not args.once:
        runner.learn(
            num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
        )
    else:
        env.reset()
        env.step(env.actions)
        while True:
            time.sleep(0.1)


if __name__ == "__main__":
    main()
