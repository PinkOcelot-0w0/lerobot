from typing import Any

from ..so_leader import SOLeader
from .config_kinova_gen3_so101_leader import KinovaGen3So101LeaderConfig


class KinovaGen3So101Leader(SOLeader):
    """SO-ARM101 leader mapped to the seven joints of a Kinova Gen3."""

    config_class = KinovaGen3So101LeaderConfig
    name = "kinova_gen3_so101_leader"

    def __init__(self, config: KinovaGen3So101LeaderConfig):
        super().__init__(config)
        self.config = config

    @property
    def action_features(self) -> dict[str, type]:
        return {**{f"joint_id_{i}.pos": float for i in range(7)}, "gripper.pos": float}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    def get_action(self) -> dict[str, Any]:
        if not self.config.use_degrees:
            raise ValueError("kinova_gen3_so101_leader requires use_degrees=True")
        values = self.bus.sync_read("Present_Position", num_retry=self.config.num_read_retries)
        leader = {f"{motor}.pos": float(value) for motor, value in values.items()}
        def value(name: str, offset: float = 0.0, sign: float = 1.0) -> float:
            return sign * (leader[name] + offset)
        return {
            "joint_id_0.pos": value("shoulder_pan.pos"),
            "joint_id_1.pos": value("shoulder_lift.pos", self.config.shoulder_lift_offset),
            "joint_id_2.pos": 180.0,
            "joint_id_3.pos": value("elbow_flex.pos", self.config.elbow_flex_offset, -1.0),
            "joint_id_4.pos": 0.0,
            "joint_id_5.pos": value("wrist_flex.pos", self.config.wrist_flex_offset, -1.0),
            "joint_id_6.pos": value("wrist_roll.pos"),
            "gripper.pos": value("gripper.pos", self.config.gripper_offset, -0.01),
        }
