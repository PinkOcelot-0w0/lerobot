"""MuJoCo model helpers for the Kinova Gen3 VR teleoperation adapter."""

from __future__ import annotations

from pathlib import Path

import mujoco

from lerobot.robots.kinova_gen3.mujoco_model import load_kinova_gen3_mujoco_model

KINOVA_JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 8))
KINOVA_EE_SITE = "pinch_site"


def build_kinova_gen3_model(
    model_xml: str | Path | None = None,
    gripper_xml: str | Path | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the repository's Kinova Gen3 scene with the Robotiq gripper attached.

    The default scene includes the table and free cube used for simulation. The
    IK solver addresses the seven named arm joints directly, so extra free
    objects in the scene do not change its joint-vector interface.
    """
    arm_path = Path(model_xml).expanduser() if model_xml is not None else None
    gripper_path = Path(gripper_xml).expanduser() if gripper_xml is not None else None
    model = load_kinova_gen3_mujoco_model(arm_path, gripper_path)

    missing = [
        name
        for name in (*KINOVA_JOINT_NAMES, KINOVA_EE_SITE)
        if mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE if name == KINOVA_EE_SITE else mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        < 0
    ]
    if missing:
        raise ValueError(f"Kinova Gen3 MuJoCo model is missing required joints/sites: {missing}")

    return model, mujoco.MjData(model)
