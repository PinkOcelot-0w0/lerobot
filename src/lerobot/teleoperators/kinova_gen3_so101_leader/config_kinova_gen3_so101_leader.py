from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("kinova_gen3_so101_leader")
@dataclass
class KinovaGen3So101LeaderConfig(TeleoperatorConfig):
    port: str
    use_degrees: bool = True
    num_read_retries: int = 2
    shoulder_lift_offset: float = 20.0
    elbow_flex_offset: float = 90.0
    wrist_flex_offset: float = -20.0
    gripper_offset: float = -100.0
