import argparse
import pickle
from pathlib import Path
from datetime import datetime

import genesis as gs
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from tqdm import tqdm

from src.utils import get_class, get_latest


class Test():
    def __init__(self, linear_vel_range, angular_vel_range, command_interval, name, timesteps):
        self.linear_vel_range = linear_vel_range
        self.angular_vel_range = angular_vel_range
        self.command_interval = command_interval
        self.name = name
        self.timesteps = timesteps
    
    def run(self, env, policy):
        """Run test phase and return detailed results."""
        obs = env.reset()

        # Initialize tracking variables
        lin_vel_errors = []
        ang_vel_errors = []
        survived = True
        step_count = 0
        command_step_counter = 0
        current_command = None

        with torch.no_grad():
            pbar = tqdm(range(self.timesteps), desc="Simulating", leave=False, unit="step", colour='cyan')
            for step in pbar:
                # Generate random command at specified intervals
                if step % self.command_interval == 0:
                    current_command = [
                        np.random.uniform(*self.linear_vel_range),  # lin_vel_x
                        0.0,  # lin_vel_y (keeping 0 for forward motion only)
                        np.random.uniform(*self.angular_vel_range),  # ang_vel_z
                    ]
                    command_step_counter = 0

                # Run policy step
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions, command=current_command)

                # Track velocity errors if we have a command
                if current_command is not None:
                    base_lin_vel = env.base_lin_vel[0].cpu().numpy()
                    base_ang_vel = env.base_ang_vel[0].cpu().numpy()

                    lin_vel_error = np.sqrt(
                        (base_lin_vel[0] - current_command[0])**2 +
                        (base_lin_vel[1] - current_command[1])**2
                    )
                    ang_vel_error = abs(base_ang_vel[2] - current_command[2])

                    lin_vel_errors.append(lin_vel_error)
                    ang_vel_errors.append(ang_vel_error)

                # Update progress bar with current errors
                if lin_vel_errors:
                    pbar.set_postfix({
                        'lin_err': f'{lin_vel_errors[-1]:.3f}',
                        'ang_err': f'{ang_vel_errors[-1]:.3f}'
                    })

                # Check for failure
                if dones[0]:
                    survived = False
                    step_count = step + 1
                    pbar.close()
                    break

                step_count = step + 1
                command_step_counter += 1

        # Calculate results
        results = {
            'survived': survived,
            'steps_completed': step_count,
            'avg_lin_vel_error': np.mean(lin_vel_errors) if lin_vel_errors else float('inf'),
            'avg_ang_vel_error': np.mean(ang_vel_errors) if ang_vel_errors else float('inf'),
            'max_lin_vel_error': np.max(lin_vel_errors) if lin_vel_errors else float('inf'),
            'max_ang_vel_error': np.max(ang_vel_errors) if ang_vel_errors else float('inf'),
        }

        return results

test_phases = {
    "slow_commands": {
        'linear_vel_range': (0.2, 0.5),
        'angular_vel_range': (-0.1, 0.1),
        'command_interval': 5,
        'name': 'Phase 1: Slow Commands (0.2-0.5 m/s, -0.1 to 0.1 rad/s)',
        'timesteps': 100
    },
    "medium_commands": {
        'linear_vel_range': (0.5, 0.8),
        'angular_vel_range': (-0.2, 0.2),
        'command_interval': 3,
        'name': 'Phase 2: Medium Commands (0.5-0.8 m/s, -0.2 to 0.2 rad/s)',
        'timesteps': 100
    },
    "fast_commands": {
        'linear_vel_range': (0.8, 1.0),
        'angular_vel_range': (-0.3, 0.3),
        'command_interval': 1,
        'name': 'Phase 3: Fast Commands (0.8-1.0 m/s, -0.3 to 0.3 rad/s)',
        'timesteps': 100
    },
    "rapid_commands": {
        'linear_vel_range': (0.2, 1.0),
        'angular_vel_range': (-0.5, 0.5),
        'command_interval': 1,
        'name': 'Phase 4: Rapid Commands (0.2-1.0 m/s, -0.5 to 0.5 rad/s)',
        'timesteps': 100
    },
    "full_range": {
        'linear_vel_range': (0.2, 1.0),
        'angular_vel_range': (-0.5, 0.5),
        'command_interval': 2,
        'name': 'Phase 5: Full Range Commands (0.2-1.0 m/s, -0.5 to 0.5 rad/s)',
        'timesteps': 100
    }
}


def main():
    # number of runs per test phase is its own argument. Tests don't print each time they run, but aggregate results at the end.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--runs_per_phase",
        type=int,
        default=5,
        help="Number of runs to perform per test phase (default: 5)",
    )
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
        help="Model iteration file to load (default: model_[max].pt)",
    )
    args = parser.parse_args()

    # Initialize Genesis in headless mode
    gs.init(
        precision="32",
        logging_level="warning",
        performance_mode=True,
    )

    from src.env import GenesisEnv

    # Load experiment and model
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

    print(f"Loading experiment: {exp_dir}")
    print(f"Loading model: {model_path}")

    # Load config
    config = pickle.load(open(f"{exp_dir}/config.pkl", "rb"))

    # Create environment (single env for testing, headless)
    env_class: type[GenesisEnv] | None = get_class(
        "src.env", config["env"]["class_name"]
    )
    if not env_class:
        raise ValueError(f"Environment class not found: {config['env']['class_name']}")

    env = env_class(
        1,  # Single environment for testing
        config["env"],
        config["obs"],
        config["reward"],
        config["commands"],
        headless=True,
    )

    # Load policy
    runner = OnPolicyRunner(
        env, config["runner"], str(model_path.parent), device=str(gs.device)
    )
    runner.load(str(model_path), map_location=str(gs.device))
    policy = runner.get_inference_policy(device=str(gs.device))

    print(f"\nRunning {args.runs_per_phase} test(s) per phase...\n")

    # Run test phases
    all_results = {}
    for phase_name, phase_config in tqdm(test_phases.items(), desc="Test Phases", unit="phase", colour='green'):
        print(f"\n{'='*60}")
        print(f"{phase_config['name']}")
        print(f"{'='*60}")

        phase_results = []
        for _ in tqdm(range(args.runs_per_phase), desc=f"Running {phase_name}", leave=False, unit="run"):
            test = Test(**phase_config)
            results = test.run(env, policy)
            phase_results.append(results)

        # Aggregate results for this phase
        survived_count = sum(1 for r in phase_results if r['survived'])
        avg_lin_vel_errors = [r['avg_lin_vel_error'] for r in phase_results if r['survived']]
        avg_ang_vel_errors = [r['avg_ang_vel_error'] for r in phase_results if r['survived']]

        all_results[phase_name] = {
            'survival_rate': survived_count / args.runs_per_phase,
            'avg_lin_vel_error': np.mean(avg_lin_vel_errors) if avg_lin_vel_errors else float('inf'),
            'avg_ang_vel_error': np.mean(avg_ang_vel_errors) if avg_ang_vel_errors else float('inf'),
            'std_lin_vel_error': np.std(avg_lin_vel_errors) if avg_lin_vel_errors else 0,
            'std_ang_vel_error': np.std(avg_ang_vel_errors) if avg_ang_vel_errors else 0,
        }

        # Print results for this phase
        print(f"Survival Rate: {all_results[phase_name]['survival_rate']*100:.1f}% ({survived_count}/{args.runs_per_phase})")
        if avg_lin_vel_errors:
            print(f"Avg Linear Velocity Error: {all_results[phase_name]['avg_lin_vel_error']:.4f} ± {all_results[phase_name]['std_lin_vel_error']:.4f} m/s")
            print(f"Avg Angular Velocity Error: {all_results[phase_name]['avg_ang_vel_error']:.4f} ± {all_results[phase_name]['std_ang_vel_error']:.4f} rad/s")
        else:
            print("No successful runs to calculate velocity errors")
        print()

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for phase_name, results in all_results.items():
        print(f"{phase_name:20s} - Survival: {results['survival_rate']*100:5.1f}% | "
              f"Lin Vel Error: {results['avg_lin_vel_error']:6.4f} m/s | "
              f"Ang Vel Error: {results['avg_ang_vel_error']:6.4f} rad/s")

    # Save results to file
    tests_dir = Path("tests")
    tests_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = model_path.name
    output_file = tests_dir / f"test_results_{timestamp}.txt"

    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TEST RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Experiment: {exp_dir}\n")
        f.write(f"Runs per phase: {args.runs_per_phase}\n\n")

        # Write detailed phase results
        for phase_name, phase_config in test_phases.items():
            f.write("="*60 + "\n")
            f.write(f"{phase_config['name']}\n")
            f.write("="*60 + "\n")

            results = all_results[phase_name]
            survived_count = int(results['survival_rate'] * args.runs_per_phase)

            f.write(f"Survival Rate: {results['survival_rate']*100:.1f}% ({survived_count}/{args.runs_per_phase})\n")
            if results['avg_lin_vel_error'] != float('inf'):
                f.write(f"Avg Linear Velocity Error: {results['avg_lin_vel_error']:.4f} ± {results['std_lin_vel_error']:.4f} m/s\n")
                f.write(f"Avg Angular Velocity Error: {results['avg_ang_vel_error']:.4f} ± {results['std_ang_vel_error']:.4f} rad/s\n")
            else:
                f.write("No successful runs to calculate velocity errors\n")
            f.write("\n")

        # Write summary
        f.write("="*60 + "\n")
        f.write("SUMMARY\n")
        f.write("="*60 + "\n")
        for phase_name, results in all_results.items():
            f.write(f"{phase_name:20s} - Survival: {results['survival_rate']*100:5.1f}% | "
                  f"Lin Vel Error: {results['avg_lin_vel_error']:6.4f} m/s | "
                  f"Ang Vel Error: {results['avg_ang_vel_error']:6.4f} rad/s\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()