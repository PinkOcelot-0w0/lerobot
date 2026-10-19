from pathlib import Path
from typing import Any

from lerobot.utils.import_utils import _mujoco_available, require_package


def load_kinova_gen3_mujoco_model(
    arm_xml: Path | None = None,
    gripper_xml: Path | None = None,
) -> Any:
    """Load the packaged Gen3 scene and attach the Robotiq gripper at pinch_site."""
    if not _mujoco_available:
        require_package("mujoco", "kinova_gen3_sim")
    import mujoco

    models_dir = Path(__file__).resolve().parent / "models"
    arm_xml = (arm_xml or models_dir / "kinova_gen3" / "scene.xml").expanduser().resolve()
    gripper_xml = (gripper_xml or models_dir / "robotiq_2f85" / "2f85.xml").expanduser().resolve()
    if not arm_xml.is_file():
        raise FileNotFoundError(f"Kinova MuJoCo model not found: {arm_xml}")
    if not gripper_xml.is_file():
        raise FileNotFoundError(f"Robotiq 2F-85 MuJoCo model not found: {gripper_xml}")

    arm_spec = mujoco.MjSpec.from_file(str(arm_xml))
    gripper_spec = mujoco.MjSpec.from_file(str(gripper_xml))
    pinch_site = arm_spec.site("pinch_site")
    if pinch_site is None:
        raise ValueError(f"The Kinova model has no 'pinch_site' for attaching the gripper: {arm_xml}")
    arm_spec.attach(gripper_spec, site=pinch_site, prefix="gripper_")
    return arm_spec.compile()
