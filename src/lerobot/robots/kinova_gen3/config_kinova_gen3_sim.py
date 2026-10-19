from dataclasses import dataclass, field
from pathlib import Path

from ..config import RobotConfig


@RobotConfig.register_subclass("kinova_gen3_sim")
@dataclass
class KinovaGen3SimConfig(RobotConfig):
    """MuJoCo simulation settings for a Kinova Gen3 with Robotiq 2F-85."""

    model_xml: Path | None = None
    gripper_xml: Path | None = None
    show_viewer: bool = True
    simulation_timestep: float = 0.002
    viewer_sync_hz: float = 60.0
    render_cameras: list[str] = field(default_factory=list)
    render_width: int = 640
    render_height: int = 480
