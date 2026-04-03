import argparse
import pickle
import time
from pathlib import Path

import genesis as gs
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from src.config import Config
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
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        choices=["follow", "cycle", "orbit"],
        help="Camera mode: 'follow' tracks from behind, 'cycle' rotates through angles every 4s, 'orbit' smoothly circles the robot",
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

    config: Config = pickle.load(open(f"{exp_dir}/config.pkl", "rb"))

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
    actor = runner.alg.actor
    actor.to("cpu")
    actor.eval()

    num_obs = config["obs"]["num_obs"]
    if isinstance(num_obs, dict):
        num_obs = num_obs.get("main", next(iter(num_obs.values())))

    # Wrap actor so ONNX sees a plain flat tensor instead of TensorDict
    obs_group = list(runner.cfg["obs_groups"]["actor"])[0]

    import tensordict as td_lib

    class ActorWrapper(torch.nn.Module):
        def __init__(self, model, group):
            super().__init__()
            self.model = model
            self.group = group

        def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
            obs_td = td_lib.TensorDict({self.group: obs_flat.unsqueeze(0)}, batch_size=[1])
            return self.model(obs_td).squeeze(0)

    onnx_model = ActorWrapper(actor, obs_group)
    onnx_model.eval()

    # Trace and save the model
    torch.onnx.export(
        onnx_model,
        torch.zeros(num_obs).to("cpu"),
        "./policy.onnx",
        export_params=True,
        opset_version=18,
        verbose=False
    )

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

    # Camera offsets for cycle mode: (x, y, z) relative to robot
    CYCLE_OFFSETS = [
        (2.5,  0.0,  1.0),   # rear
        (0.0,  2.5,  1.0),   # right side
        (-2.5, 0.0,  1.0),   # front
        (0.0, -2.5,  1.0),   # left side
        (1.5,  1.5,  3.0),   # high angle
    ]
    cycle_index = 0
    cycle_last_switch = time.monotonic()

    obs = env.reset()

    ORBIT_RADIUS = 2.5
    ORBIT_HEIGHT = 1.2
    ORBIT_PERIOD = 30.0  # seconds per full revolution
    orbit_start = time.monotonic()

    if args.camera in ("follow", "cycle") and env.scene.viewer:
        env.scene.viewer.follow_entity(env.robot)

    with torch.no_grad():
        while True:
            if args.camera == "orbit" and env.scene.viewer:
                angle = (time.monotonic() - orbit_start) / ORBIT_PERIOD * 2.0 * np.pi
                robot_pos = env.robot.get_pos().cpu().numpy()[0]
                cam_pos = robot_pos + np.array([
                    ORBIT_RADIUS * np.cos(angle),
                    ORBIT_RADIUS * np.sin(angle),
                    ORBIT_HEIGHT,
                ])
                env.scene.viewer._camera_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                env.scene.viewer.set_camera_pose(pos=cam_pos, lookat=robot_pos)

            if args.camera == "cycle" and env.scene.viewer:
                now = time.monotonic()
                if now - cycle_last_switch >= 4.0:
                    cycle_index = (cycle_index + 1) % len(CYCLE_OFFSETS)
                    cycle_last_switch = now
                    # Rebuild follow with new offset by temporarily patching init pos
                    env.scene.viewer._camera_init_pos = np.array(CYCLE_OFFSETS[cycle_index], dtype=np.float32)
                    env.scene.viewer._camera_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    env.scene.viewer.follow_entity(env.robot)

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
