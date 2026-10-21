import logging
import threading
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.import_utils import _xrobotoolkit_sdk_available, require_package

from ..teleoperator import Teleoperator
from .config_kinova_gen3_quest3 import KinovaGen3Quest3Config

logger = logging.getLogger(__name__)

_JOINT_NAMES = tuple(f"joint_{i}" for i in range(1, 8))
_GRIPPER_JOINT_NAME = "gripper_right_driver_joint"
_GRIPPER_JOINT_RANGE = 0.8


class KinovaGen3Quest3(Teleoperator):
    """Quest3 controller input using XRoboToolkit's clutch-based delta-pose scheme."""

    config_class = KinovaGen3Quest3Config
    name = "kinova_gen3_quest3"
    requires_feedback = True

    def __init__(self, config: KinovaGen3Quest3Config):
        super().__init__(config)
        self.config = config
        self._xr = None
        self._connected = False
        self._state_lock = threading.RLock()
        self._mujoco = None
        self._model = None
        self._data = None
        self._site_id = -1
        self._joint_qpos_addresses: list[int] = []
        self._joint_dof_addresses: list[int] = []
        self._joint_ids: list[int] = []
        self._joint_actuator_ids: list[int] = []
        self._measured_joint_deg = np.zeros(7, dtype=np.float64)
        self._measured_gripper = 0.0
        self._target_controller_pos: np.ndarray | None = None
        self._target_controller_rotation: np.ndarray | None = None
        self._target_ee_pos: np.ndarray | None = None
        self._target_ee_rotation: np.ndarray | None = None
        self._previous_controller_pos: np.ndarray | None = None
        self._previous_controller_rotation: np.ndarray | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {**{f"joint_id_{i}.pos": float for i in range(7)}, "gripper.pos": float}

    @property
    def feedback_features(self) -> dict[str, type]:
        return self.action_features

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        if not _xrobotoolkit_sdk_available:
            require_package("xrobotoolkit-sdk", "quest3", import_name="xrobotoolkit_sdk")
        if not self.config.control_orientation and self.config.orientation_scale != 0:
            logger.info("Quest3 orientation input is disabled; position-only IK will be used")
        if not 0.0 < self.config.clutch_threshold <= 1.0:
            raise ValueError("clutch_threshold must be in (0, 1]")
        if self.config.position_scale <= 0 or self.config.ik_iterations <= 0:
            raise ValueError("position_scale and ik_iterations must be greater than zero")
        if self.config.ik_damping <= 0 or self.config.max_joint_step_deg <= 0:
            raise ValueError("ik_damping and max_joint_step_deg must be greater than zero")
        if self.config.max_controller_delta_m <= 0 or self.config.max_controller_rotation_rad <= 0:
            raise ValueError("Controller delta limits must be greater than zero")
        transform = np.asarray(self.config.world_from_headset, dtype=np.float64)
        if (
            transform.shape != (3, 3)
            or not np.allclose(transform @ transform.T, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(transform), 1.0, atol=1e-5)
        ):
            raise ValueError("world_from_headset must be an orthonormal 3x3 rotation matrix")
        if self.config.orientation_scale < 0 or self.config.orientation_weight < 0:
            raise ValueError("orientation_scale and orientation_weight must be non-negative")

        import mujoco
        import xrobotoolkit_sdk as xrt

        from lerobot.robots.kinova_gen3.mujoco_model import load_kinova_gen3_mujoco_model

        self._mujoco = mujoco
        self._model = load_kinova_gen3_mujoco_model(self.config.model_xml, self.config.gripper_xml)
        self._data = mujoco.MjData(self._model)
        home_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id >= 0:
            mujoco.mj_resetDataKeyframe(self._model, self._data, home_id)
        else:
            mujoco.mj_resetData(self._model, self._data)

        self._joint_qpos_addresses = []
        self._joint_dof_addresses = []
        self._joint_ids = []
        self._joint_actuator_ids = []
        for name in _JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            actuator_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if joint_id < 0 or actuator_id < 0:
                raise ValueError(f"Kinova MuJoCo model is missing joint/actuator {name!r}")
            self._joint_ids.append(joint_id)
            self._joint_qpos_addresses.append(int(self._model.jnt_qposadr[joint_id]))
            self._joint_dof_addresses.append(int(self._model.jnt_dofadr[joint_id]))
            self._joint_actuator_ids.append(actuator_id)
        gripper_joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, _GRIPPER_JOINT_NAME)
        self._site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
        if gripper_joint_id < 0 or self._site_id < 0:
            raise ValueError("Kinova model needs Robotiq gripper and 'pinch_site'")
        self._gripper_qpos_address = int(self._model.jnt_qposadr[gripper_joint_id])
        mujoco.mj_forward(self._model, self._data)
        self._measured_joint_deg = np.rad2deg(
            self._data.qpos[self._joint_qpos_addresses].copy()
        )
        self._measured_gripper = float(
            np.clip(self._data.qpos[self._gripper_qpos_address] / _GRIPPER_JOINT_RANGE, 0.0, 1.0)
        )

        try:
            xrt.init()
        except Exception as exc:
            raise RuntimeError(
                "Could not initialize XRoboToolkit. Install xrobotoolkit-sdk, start XRoboToolkit "
                "PC Service, and connect the headset before starting teleoperation."
            ) from exc
        self._xr = xrt
        self._connected = True
        logger.info("%s connected to XRoboToolkit; hold the right grip to engage the arm", self)

    def disconnect(self) -> None:
        if self._xr is not None:
            try:
                self._xr.close()
            except Exception:
                logger.exception("Failed to close XRoboToolkit SDK")
            finally:
                self._xr = None
        self._connected = False

    def configure(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

    def calibrate(self) -> None:
        return

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        with self._state_lock:
            for index in range(7):
                key = f"joint_id_{index}.pos"
                if key in feedback:
                    self._measured_joint_deg[index] = float(feedback[key])
            if "gripper.pos" in feedback:
                self._measured_gripper = float(np.clip(feedback["gripper.pos"], 0.0, 1.0))
            self._set_model_joint_state(self._measured_joint_deg)
            self._mujoco.mj_forward(self._model, self._data)

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        clutch = float(self._xr.get_right_grip())
        gripper_trigger = float(self._xr.get_right_trigger())
        if not np.isfinite(gripper_trigger):
            gripper_trigger = 1.0 - self._measured_gripper
        measured = self._measured_joint_deg.copy()

        if clutch >= self.config.clutch_threshold:
            pose = self._read_controller_pose()
            if pose is not None:
                controller_pos, controller_rotation = pose
                if self._target_ee_pos is None or self._previous_controller_pos is None:
                    self._set_end_effector_reference()
                    self._target_controller_pos = controller_pos
                    self._target_controller_rotation = controller_rotation
                    self._previous_controller_pos = controller_pos
                    self._previous_controller_rotation = controller_rotation
                    measured = self._measured_joint_deg.copy()
                else:
                    self._apply_controller_delta(controller_pos, controller_rotation)
                    target_joints = self._solve_ik()
                    delta = target_joints - measured
                    delta = np.clip(delta, -self.config.max_joint_step_deg, self.config.max_joint_step_deg)
                    measured = measured + delta
        else:
            self._clear_clutch_reference()

        action = {f"joint_id_{i}.pos": float(measured[i]) for i in range(7)}
        action["gripper.pos"] = float(np.clip(1.0 - gripper_trigger, 0.0, 1.0))
        return action

    def _read_controller_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        raw_pose = np.asarray(self._xr.get_right_controller_pose(), dtype=np.float64).reshape(-1)
        if raw_pose.size != 7 or not np.all(np.isfinite(raw_pose)):
            logger.warning("Ignoring invalid Quest controller pose")
            return None
        if np.linalg.norm(raw_pose[3:7]) < 1e-8:
            logger.warning("Ignoring Quest controller pose with a zero-length quaternion")
            return None
        position = self._world_from_headset() @ raw_pose[:3]
        rotation = self._quaternion_to_matrix(raw_pose[3:7])
        world_from_headset = self._world_from_headset()
        rotation = world_from_headset @ rotation @ world_from_headset.T
        return position, rotation

    def _set_end_effector_reference(self) -> None:
        self._set_model_joint_state(self._measured_joint_deg)
        self._mujoco.mj_forward(self._model, self._data)
        self._target_ee_pos = self._data.site_xpos[self._site_id].copy()
        self._target_ee_rotation = self._data.site_xmat[self._site_id].reshape(3, 3).copy()

    def _apply_controller_delta(self, controller_pos: np.ndarray, controller_rotation: np.ndarray) -> None:
        assert self._previous_controller_pos is not None
        assert self._previous_controller_rotation is not None
        assert self._target_ee_pos is not None
        assert self._target_ee_rotation is not None

        delta_position = (controller_pos - self._previous_controller_pos) * self.config.position_scale
        distance = float(np.linalg.norm(delta_position))
        if distance > self.config.max_controller_delta_m:
            delta_position *= self.config.max_controller_delta_m / distance
        self._target_ee_pos += delta_position

        if self.config.control_orientation:
            delta_world_rotation = controller_rotation @ self._previous_controller_rotation.T
            delta_rotation = self._rotation_log(delta_world_rotation) * self.config.orientation_scale
            angle = float(np.linalg.norm(delta_rotation))
            if angle > self.config.max_controller_rotation_rad:
                delta_rotation *= self.config.max_controller_rotation_rad / angle
            delta_rotation_matrix = self._rotation_exp(delta_rotation)
            self._target_ee_rotation = delta_rotation_matrix @ self._target_ee_rotation

        self._previous_controller_pos = controller_pos
        self._previous_controller_rotation = controller_rotation

    def _solve_ik(self) -> np.ndarray:
        assert self._target_ee_pos is not None
        assert self._target_ee_rotation is not None
        mujoco = self._mujoco
        qpos_addresses = self._joint_qpos_addresses
        dof_addresses = self._joint_dof_addresses
        measured_rad = np.deg2rad(self._measured_joint_deg)
        jacobian_position = np.zeros((3, self._model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self._model.nv), dtype=np.float64)
        weighted_jacobian = np.zeros((6, 7), dtype=np.float64)
        error = np.zeros(6, dtype=np.float64)
        weights = np.array([1.0, 1.0, 1.0, *([self.config.orientation_weight] * 3)])

        self._set_model_joint_state(self._measured_joint_deg)
        for _ in range(self.config.ik_iterations):
            mujoco.mj_forward(self._model, self._data)
            current_position = self._data.site_xpos[self._site_id].copy()
            current_rotation = self._data.site_xmat[self._site_id].reshape(3, 3).copy()
            position_error = self._target_ee_pos - current_position
            rotation_error = self._rotation_log(self._target_ee_rotation @ current_rotation.T)
            if (
                np.linalg.norm(position_error) <= self.config.position_deadband_m
                and np.linalg.norm(rotation_error) <= self.config.rotation_deadband_rad
            ):
                break

            mujoco.mj_jacSite(self._model, self._data, jacobian_position, jacobian_rotation, self._site_id)
            error[:3] = position_error
            error[3:] = rotation_error
            for index, dof_address in enumerate(dof_addresses):
                weighted_jacobian[:3, index] = jacobian_position[:, dof_address]
                weighted_jacobian[3:, index] = jacobian_rotation[:, dof_address]
            weighted_jacobian *= weights[:, None]
            weighted_error = error * weights
            system = weighted_jacobian @ weighted_jacobian.T + self.config.ik_damping**2 * np.eye(6)
            try:
                delta = weighted_jacobian.T @ np.linalg.solve(system, weighted_error)
            except np.linalg.LinAlgError:
                logger.warning("Quest3 IK became singular; holding the measured Kinova pose")
                return self._measured_joint_deg.copy()
            delta = np.clip(delta, -0.15, 0.15)
            measured_rad += delta
            for index, joint_id in enumerate(self._joint_ids):
                if self._model.jnt_limited[joint_id]:
                    lower, upper = self._model.jnt_range[joint_id]
                    measured_rad[index] = np.clip(measured_rad[index], lower, upper)
            self._data.qpos[qpos_addresses] = measured_rad

        return np.rad2deg(measured_rad)

    def _set_model_joint_state(self, joint_positions_deg: np.ndarray) -> None:
        self._data.qpos[self._joint_qpos_addresses] = np.deg2rad(joint_positions_deg)

    def _clear_clutch_reference(self) -> None:
        self._target_controller_pos = None
        self._target_controller_rotation = None
        self._target_ee_pos = None
        self._target_ee_rotation = None
        self._previous_controller_pos = None
        self._previous_controller_rotation = None

    def _world_from_headset(self) -> np.ndarray:
        return np.asarray(self.config.world_from_headset, dtype=np.float64)

    @staticmethod
    def _quaternion_to_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
        x, y, z, w = quaternion_xyzw
        norm = float(np.linalg.norm(quaternion_xyzw))
        if norm < 1e-8:
            raise ValueError("Quest controller returned a zero-length quaternion")
        x, y, z, w = quaternion_xyzw / norm
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rotation_log(rotation: np.ndarray) -> np.ndarray:
        cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
        angle = float(np.arccos(cosine))
        skew = np.array(
            [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]
        )
        if angle < 1e-7:
            return 0.5 * skew
        sine = float(np.sin(angle))
        if abs(sine) < 1e-7:
            return angle * np.array([1.0, 0.0, 0.0])
        return angle / (2.0 * sine) * skew

    @staticmethod
    def _rotation_exp(rotation_vector: np.ndarray) -> np.ndarray:
        angle = float(np.linalg.norm(rotation_vector))
        if angle < 1e-8:
            return np.eye(3)
        axis = rotation_vector / angle
        x, y, z = axis
        skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
