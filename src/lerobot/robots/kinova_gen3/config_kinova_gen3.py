from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("kinova_gen3")
@dataclass
class KinovaGen3Config(RobotConfig):
    """Configuration for a Kinova Gen3 controlled through Kortex TCP API."""

    ip: str = "192.168.1.10"
    port: int = 10000
    kinova_user: str = "admin"
    kinova_pwd: str = "admin"
    session_inactivity_timeout: int = 10_000
    connection_inactivity_timeout: int = 2_000

    # The high-level position tracker sends joint velocities at this rate.
    control_rate_hz: float = 50.0
    position_gain: float = 1.5
    max_joint_speed_deg_s: float = 20.0
    joint_deadband_deg: float = 0.5
    gripper_rate_hz: float = 20.0
    gripper_gain: float = 0.15
    gripper_deadband: float = 0.01
    gripper_max_speed: float = 0.75
    no_op: bool = False
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
