#!/usr/bin/env python
"""Convenience launcher for Kinova Gen3 simulation, teleoperation and rollout."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KINOVA_IP_DEFAULT = "192.168.1.10"


def _add_teleop_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--teleop", choices=("quest3", "so101"), default="quest3")
    parser.add_argument("--teleop-port", help="SO-ARM101 serial port; required with --teleop so101")
    parser.add_argument("--quest-position-scale", type=float, default=1.2)


def _add_robot_args(parser: argparse.ArgumentParser, *, hardware: bool = False) -> None:
    parser.add_argument("--robot-ip", default=KINOVA_IP_DEFAULT)
    parser.add_argument(
        "--camera",
        choices=("none", "wrist"),
        default="none",
        help="Use the Kinova wrist camera as an observation",
    )
    parser.add_argument("--display-data", action="store_true", help="Show observations in Rerun")
    parser.add_argument("--dry-run", action="store_true", help="Print the LeRobot command without starting it")
    if hardware:
        parser.add_argument(
            "--max-joint-speed-deg-s",
            type=float,
            default=5.0,
            help="Kinova joint speed cap for initial real-robot testing",
        )
    else:
        parser.add_argument("--headless", action="store_true", help="Run MuJoCo without its viewer window")


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-id", help="Dataset repo id; required when --push-to-hub is used")
    parser.add_argument("--root", type=Path, help="Local dataset directory")
    parser.add_argument("--task", default="Kinova Gen3 teleoperation task")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-seconds", type=float, default=30.0)
    parser.add_argument("--reset-seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--push-to-hub", action="store_true", help="Upload the finished dataset")


def _add_target_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target", choices=("sim", "real"), default="sim")
    _add_robot_args(parser)
    parser.add_argument("--max-joint-speed-deg-s", type=float, default=5.0)


def _teleop_flags(args: argparse.Namespace) -> list[str]:
    if args.teleop == "quest3":
        return ["--teleop.type=kinova_gen3_quest3", f"--teleop.position_scale={args.quest_position_scale}"]
    if not args.teleop_port:
        raise ValueError("--teleop-port is required when --teleop=so101")
    return [
        "--teleop.type=kinova_gen3_so101_leader",
        f"--teleop.port={args.teleop_port}",
    ]


def _robot_flags(args: argparse.Namespace, target: str) -> list[str]:
    flags = [f"--robot.ip={args.robot_ip}"]
    if target == "sim":
        flags.extend(["--robot.type=kinova_gen3_sim", f"--robot.show_viewer={str(not args.headless).lower()}"])
        if args.camera == "wrist":
            flags.append("--robot.render_cameras=[wrist]")
    else:
        flags.extend(
            [
                "--robot.type=kinova_gen3",
                f"--robot.max_joint_speed_deg_s={args.max_joint_speed_deg_s}",
            ]
        )
        if args.camera == "wrist":
            flags.append(
                "--robot.cameras={wrist: {type: kinova_gen3_vision, width: 640, height: 480, fps: 30}}"
            )
    if args.display_data:
        flags.append("--display_data=true")
    return flags


def _run(module: str, flags: list[str], *, dry_run: bool = False) -> int:
    command = [sys.executable, "-m", module, *flags]
    print(f"Launching: {shlex.join(command)}", flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def _run_teleoperation(args: argparse.Namespace, target: str) -> int:
    flags = _robot_flags(args, target) + _teleop_flags(args) + [f"--fps={args.fps}"]
    return _run("lerobot.scripts.lerobot_teleoperate", flags, dry_run=args.dry_run)


def _run_record(args: argparse.Namespace) -> int:
    if args.push_to_hub and not args.repo_id:
        raise ValueError("--repo-id must be provided when --push-to-hub is enabled")
    repo_id = args.repo_id or f"local/kinova_gen3_{args.target}"
    root = args.root or (REPO_ROOT / "outputs" / "datasets" / f"kinova_gen3_{args.target}")
    flags = _robot_flags(args, args.target) + _teleop_flags(args)
    flags.extend(
        [
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={root}",
            f"--dataset.single_task={args.task}",
            f"--dataset.num_episodes={args.episodes}",
            f"--dataset.episode_time_s={args.episode_seconds}",
            f"--dataset.reset_time_s={args.reset_seconds}",
            f"--dataset.fps={args.fps}",
            f"--dataset.push_to_hub={str(args.push_to_hub).lower()}",
        ]
    )
    return _run("lerobot.scripts.lerobot_record", flags, dry_run=args.dry_run)


def _run_rollout(args: argparse.Namespace) -> int:
    if not args.policy:
        raise ValueError("--policy is required for rollout")
    flags = _robot_flags(args, args.target)
    flags.extend(
        [
            "--strategy.type=base",
            f"--policy.path={args.policy}",
            f"--task={args.task}",
            f"--duration={args.duration}",
            f"--fps={args.fps}",
        ]
    )
    if args.inference != "sync":
        flags.append(f"--inference.type={args.inference}")
    if args.device:
        flags.append(f"--device={args.device}")
    return _run("lerobot.scripts.lerobot_rollout", flags, dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quick launcher for Kinova Gen3 MuJoCo simulation and hardware workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sim = subparsers.add_parser("sim", help="teleoperate the MuJoCo Gen3 + Robotiq model")
    _add_robot_args(sim)
    _add_teleop_args(sim)
    sim.add_argument("--fps", type=int, default=30)
    sim.set_defaults(run=lambda args: _run_teleoperation(args, "sim"))

    real = subparsers.add_parser("real", help="teleoperate the physical Kinova Gen3")
    _add_robot_args(real, hardware=True)
    _add_teleop_args(real)
    real.add_argument("--fps", type=int, default=30)
    real.set_defaults(run=lambda args: _run_teleoperation(args, "real"))

    record = subparsers.add_parser("record", help="collect a LeRobot dataset")
    _add_target_args(record)
    _add_teleop_args(record)
    _add_dataset_args(record)
    record.set_defaults(run=_run_record)

    rollout = subparsers.add_parser("rollout", help="run a trained policy")
    _add_target_args(rollout)
    rollout.add_argument("--policy", required=True, help="Local checkpoint path or Hub model id")
    rollout.add_argument("--task", default="Run the trained Kinova task")
    rollout.add_argument("--duration", type=float, default=30.0)
    rollout.add_argument("--fps", type=int, default=30)
    rollout.add_argument("--inference", choices=("sync", "rtc", "remote"), default="sync")
    rollout.add_argument("--device", help="Inference device, e.g. cuda or cpu")
    rollout.set_defaults(run=_run_rollout)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.run(args)
    except ValueError as exc:
        build_parser().error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
