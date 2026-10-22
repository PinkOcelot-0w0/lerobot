from dataclasses import dataclass, field
from pathlib import Path

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("kinova_gen3_quest3")
@dataclass
class KinovaGen3Quest3Config(TeleoperatorConfig):
    """Quest3/XRoboToolkit incremental end-effector teleoperation for Kinova Gen3."""

    model_xml: Path | None = None
    gripper_xml: Path | None = None
    clutch_trigger: str = "right_grip"
    gripper_trigger: str = "right_trigger"
    clutch_threshold: float = 0.9
    position_scale: float = 1.2
    orientation_scale: float = 1.0
    world_from_headset: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    control_orientation: bool = True
    ik_iterations: int = 18
    ik_damping: float = 0.04
    orientation_weight: float = 0.35
    max_controller_delta_m: float = 0.08
    # Per-sample IK setpoint slew limit. At 60 Hz this allows up to 360 deg/s;
    # the real robot's own velocity cap remains a separate safety limit.
    max_joint_step_deg: float = 6.0
    max_controller_rotation_rad: float = 0.35
    position_deadband_m: float = 0.001
    rotation_deadband_rad: float = 0.01
