import os
import pickle
import shutil
import time
from datetime import datetime

import genesis as gs
import hydra
import yaml
from rsl_rl.runners import OnPolicyRunner
from servo import Config

from src.utils import get_class


@hydra.main(config_path="config", config_name="servobot")
def main(cfg: Config):

    exp_name = cfg.runner.experiment_name

    # Determine log directory
    if cfg.log_dir:
        # Use custom directory name
        log_dir = f"logs/{cfg.log_dir}"
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
