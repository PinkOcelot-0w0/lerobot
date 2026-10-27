"""LeRobot ``Teleoperator`` adapters with optional, lazy-loaded VR dependencies."""

__all__ = [
    "BiQuestTeleoperator",
    "BiQuestTeleoperatorConfig",
    "SingleArmQuestTeleoperator",
    "SingleArmQuestTeleoperatorConfig",
    "KinovaGen3VRTeleoperator",
    "KinovaGen3VRTeleoperatorConfig",
]


def __getattr__(name: str):
    """Load adapters on demand so unrelated LeRobot CLIs need no VR extras."""
    if name in {"BiQuestTeleoperator", "BiQuestTeleoperatorConfig"}:
        from .bi_quest_teleop import BiQuestTeleoperator, BiQuestTeleoperatorConfig

        return {
            "BiQuestTeleoperator": BiQuestTeleoperator,
            "BiQuestTeleoperatorConfig": BiQuestTeleoperatorConfig,
        }[name]
    if name in {"SingleArmQuestTeleoperator", "SingleArmQuestTeleoperatorConfig"}:
        from .single_arm_quest_teleop import (
            SingleArmQuestTeleoperator,
            SingleArmQuestTeleoperatorConfig,
        )

        return {
            "SingleArmQuestTeleoperator": SingleArmQuestTeleoperator,
            "SingleArmQuestTeleoperatorConfig": SingleArmQuestTeleoperatorConfig,
        }[name]
    if name in {"KinovaGen3VRTeleoperator", "KinovaGen3VRTeleoperatorConfig"}:
        from .kinova_gen3_teleop import KinovaGen3VRTeleoperator, KinovaGen3VRTeleoperatorConfig

        return {
            "KinovaGen3VRTeleoperator": KinovaGen3VRTeleoperator,
            "KinovaGen3VRTeleoperatorConfig": KinovaGen3VRTeleoperatorConfig,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
