"""Quest WebXR teleoperator for the Kinova Gen3 MuJoCo simulator."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

if TYPE_CHECKING:
    from lerobot.vr_teleop_kit.core.pose_mapping import ClutchPoseMapper
    from lerobot.vr_teleop_kit.ik.kinova_ik import KinovaGen3IKSolver

logger = logging.getLogger(__name__)

_DEFAULT_R_CALIB = [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
_GRIP_BUTTON_INDEX = 1
_TRIGGER_BUTTON_INDEX = 0
_PRECISION_BUTTON_INDEX = 4
_HOME_BUTTON_INDEX = 3
_HANDOFF_BUTTON_INDEX = 5
_STALE_FRAME_S = 0.2
_MAX_JOINT_STEP_RAD = 0.5
_SIM_GRIPPER_FORCE_SCALE_NM = 2.0


def _yaw_from_quat_xyzw(quaternion: list[float]) -> float:
    x, y, z, w = (float(value) for value in quaternion)
    return float(np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


@TeleoperatorConfig.register_subclass("kinova_gen3_vr")
@dataclass
class KinovaGen3VRTeleoperatorConfig(TeleoperatorConfig):
    """Settings for the WebXR Quest teleoperator used by Kinova simulation."""

    ws_url: str = "ws://127.0.0.1:8443/ws"
    model_xml: Path | None = None
    gripper_xml: Path | None = None
    r_calib: list[list[float]] = field(default_factory=lambda: [row[:] for row in _DEFAULT_R_CALIB])
    scale_translation: float = 1.2
    scale_rotation: float = 1.0
    precision_factor: float = 0.5
    pose_filter_alpha: float = 0.8
    pos_reach_limit: float = 0.25
    rot_reach_limit: float = 0.6
    max_joint_step_rad: float = 0.01
    home_ramp_duration_s: float = 2.0
    damping: float = 0.08
    posture_gain: float = 0.02
    rest_qpos: list[float] | None = None
    force_haptic_enabled: bool = True
    force_haptic_threshold_nm_left: float = 0.35
    force_haptic_threshold_nm_right: float = 0.35
    force_haptic_max_nm: float = 1.0
    connect_timeout_s: float = 5.0


class KinovaGen3VRTeleoperator(Teleoperator):
    """One-controller WebXR teleoperator that emits Gen3 joint targets in degrees."""

    config_class = KinovaGen3VRTeleoperatorConfig
    name = "kinova_gen3_vr"
    requires_feedback = True

    def __init__(self, config: KinovaGen3VRTeleoperatorConfig):
        """Create a simulator-only WebXR teleoperator."""
        super().__init__(config)
        self.config = config
        self._connected = False
        self._lock = threading.RLock()
        self._latest_frame: dict[str, Any] | None = None
        self._last_frame_time = 0.0
        self._ws_thread: threading.Thread | None = None
        self._ws_loop: asyncio.AbstractEventLoop | None = None
        self._ws_stop: threading.Event | None = None
        self._ws: Any = None
        self._ws_connected = threading.Event()
        self._mapper: ClutchPoseMapper | None = None
        self._solver: KinovaGen3IKSolver | None = None
        self._joint_target_rad = np.zeros(7, dtype=np.float64)
        self._gripper_target = 0.0
        self._gripper_measured = 0.0
        self._gripper_force_proxy_nm = 0.0
        self._force_haptic = 0.0
        self._haptic_calibration: dict[str, float] | None = None
        self._home_ramp: dict[str, Any] | None = None
        self._last_home_button = False
        self._buttons = {
            "left": {"grip": False, "handoff": False},
            "right": {"grip": False, "handoff": False},
        }
        self._max_dq_pos = 0.01
        self._max_dq_rot = 0.01
        self._camera_last_sent = 0.0
        self._camera_registered = False
        self._camera_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._camera_stop = threading.Event()
        self._camera_thread: threading.Thread | None = None
        self._pos_filter: np.ndarray | None = None
        self._quat_filter: np.ndarray | None = None
        self._last_grip = False
        self._last_precision = False
        self._engaged = False
        self._haptic = 0.0
        self._loop_hz: float | None = None
        self._last_action_time: float | None = None

    @property
    def action_features(self) -> dict[str, type]:
        """Describe the seven joint targets and normalized gripper action."""
        return {**{f"joint_id_{index}.pos": float for index in range(7)}, "gripper.pos": float}

    @property
    def feedback_features(self) -> dict[str, type]:
        """Use the simulated robot's measured joints as IK feedback."""
        return self.action_features

    @property
    def is_connected(self) -> bool:
        """Return whether the teleoperator lifecycle is active, including reconnects."""
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        """This teleoperator has no persistent motor calibration."""
        return True

    def connect(self, calibrate: bool = True) -> None:
        """Initialize the Kinova IK model and connect to the WebXR relay."""
        del calibrate
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        if not 0.0 < self.config.pose_filter_alpha <= 1.0:
            raise ValueError("pose_filter_alpha must be in (0, 1]")
        if self.config.scale_translation <= 0.0 or self.config.scale_rotation < 0.0:
            raise ValueError("VR translation scale must be positive and rotation scale non-negative")
        if self.config.max_joint_step_rad <= 0.0 or self.config.home_ramp_duration_s <= 0.0:
            raise ValueError("max_joint_step_rad must be greater than zero")
        finite_settings = (
            self.config.scale_translation,
            self.config.scale_rotation,
            self.config.precision_factor,
            self.config.pose_filter_alpha,
            self.config.pos_reach_limit,
            self.config.rot_reach_limit,
            self.config.max_joint_step_rad,
            self.config.home_ramp_duration_s,
            self.config.damping,
            self.config.posture_gain,
            self.config.force_haptic_threshold_nm_left,
            self.config.force_haptic_threshold_nm_right,
            self.config.force_haptic_max_nm,
        )
        if not np.all(np.isfinite(finite_settings)):
            raise ValueError("VR teleoperator settings must all be finite")
        if not 0.0 <= self.config.precision_factor <= 1.0:
            raise ValueError("precision_factor must be in [0, 1]")
        if self.config.pos_reach_limit < 0.0 or self.config.rot_reach_limit < 0.0:
            raise ValueError("reach limits must be non-negative")
        if self.config.damping <= 0.0 or self.config.posture_gain < 0.0:
            raise ValueError("IK damping must be positive and posture gain non-negative")
        if self.config.force_haptic_max_nm <= 0.0:
            raise ValueError("force_haptic_max_nm must be greater than zero")
        rotation = np.asarray(self.config.r_calib, dtype=np.float64)
        if (
            rotation.shape != (3, 3)
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
        ):
            raise ValueError("r_calib must be an orthonormal 3x3 rotation matrix")

        # Keep MuJoCo and WebSocket packages optional for non-VR LeRobot CLIs.
        from lerobot.vr_teleop_kit.core.pose_mapping import ClutchPoseMapper
        from lerobot.vr_teleop_kit.ik.kinova_ik import KinovaGen3IKSolver

        self._solver = KinovaGen3IKSolver(
            self.config.model_xml,
            self.config.gripper_xml,
            damping=self.config.damping,
            posture_gain=self.config.posture_gain,
            rest_qpos=self.config.rest_qpos,
            max_joint_step_rad=self.config.max_joint_step_rad,
        )
        self._max_dq_pos = self.config.max_joint_step_rad
        self._max_dq_rot = self.config.max_joint_step_rad
        self._mapper = ClutchPoseMapper(
            R=rotation,
            scale=self.config.scale_translation,
            scale_rotation=self.config.scale_rotation,
            pos_reach_limit=self.config.pos_reach_limit,
            rot_reach_limit=self.config.rot_reach_limit,
        )

        try:
            import websockets  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The Kinova simulation VR teleoperator requires the VR extra. "
                "Install it with `pip install -e '.[vr-teleop]'`."
            ) from exc

        self._ws_stop = threading.Event()
        self._ws_thread = threading.Thread(target=self._ws_thread_main, name="kinova-vr-ws", daemon=True)
        self._ws_thread.start()
        if not self._ws_connected.wait(timeout=self.config.connect_timeout_s):
            self.disconnect()
            raise RuntimeError(f"Timed out connecting to the VR relay at {self.config.ws_url}")
        self._connected = True
        self._camera_stop.clear()
        self._camera_thread = threading.Thread(
            target=self._camera_publish_loop,
            name="kinova-vr-camera",
            daemon=True,
        )
        self._camera_thread.start()
        logger.info("%s connected to WebXR relay %s", self, self.config.ws_url)

    def disconnect(self) -> None:
        """Stop the relay connection and release the current clutch."""
        self._camera_stop.set()
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=1.0)
            self._camera_thread = None
        if self._ws_stop is not None:
            self._ws_stop.set()
        if self._ws_loop is not None and self._ws is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(self._ws.close(), self._ws_loop)
                future.result(timeout=1.0)
            except Exception:
                logger.debug("WebSocket close did not complete cleanly", exc_info=True)
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=2.0)
        if self._mapper is not None:
            self._mapper.disengage()
        self._connected = False
        self._ws_connected.clear()
        self._ws = None
        self._ws_loop = None
        self._ws_thread = None
        logger.info("%s disconnected", self)

    def calibrate(self) -> None:
        """No calibration is required for the WebXR controller."""
        return

    def configure(self) -> None:
        """Validate that the relay connection is active."""
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Synchronize the IK seed with the current simulated robot state."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        with self._lock:
            joint_values = [float(feedback[f"joint_id_{index}.pos"]) for index in range(7)]
            if not np.all(np.isfinite(joint_values)):
                raise ValueError("Kinova simulation feedback contains non-finite joint positions")
            self._joint_target_rad = np.deg2rad(np.asarray(joint_values, dtype=np.float64))
            if "gripper.pos" in feedback:
                self._gripper_measured = float(np.clip(float(feedback["gripper.pos"]), 0.0, 1.0))
            self._gripper_force_proxy_nm = (
                max(0.0, self._gripper_target - self._gripper_measured) * _SIM_GRIPPER_FORCE_SCALE_NM
            )
            if self._haptic_calibration is not None:
                self._haptic_calibration["peak_nm"] = max(
                    self._haptic_calibration["peak_nm"], self._gripper_force_proxy_nm
                )
            wrist_image = feedback.get("wrist")

        if wrist_image is not None:
            self._publish_sim_camera(wrist_image)

    def get_action(self) -> dict[str, float]:
        """Convert the latest right-hand WebXR pose to Gen3 joint targets."""
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        now = time.perf_counter()
        if self._last_action_time is not None:
            rate = 1.0 / max(now - self._last_action_time, 1e-3)
            self._loop_hz = rate if self._loop_hz is None else 0.1 * rate + 0.9 * self._loop_hz
        self._last_action_time = now

        with self._lock:
            frame = self._latest_frame
            frame_age = time.time() - self._last_frame_time
            settings = {
                "scale_translation": self.config.scale_translation,
                "scale_rotation": self.config.scale_rotation,
                "precision_factor": self.config.precision_factor,
                "pose_filter_alpha": self.config.pose_filter_alpha,
                "pos_reach_limit": self.config.pos_reach_limit,
                "rot_reach_limit": self.config.rot_reach_limit,
            }

        if frame is not None and frame_age <= _STALE_FRAME_S:
            self._update_from_frame(frame, settings)
        elif self._engaged:
            logger.warning("Quest pose stream is stale; releasing Kinova VR clutch")
            self._disengage()

        self._update_haptic_calibration()
        with self._lock:
            result = {
                f"joint_id_{index}.pos": float(np.rad2deg(value))
                for index, value in enumerate(self._joint_target_rad)
            }
            result["gripper.pos"] = float(self._gripper_target)
            if self.config.force_haptic_enabled:
                threshold = self.config.force_haptic_threshold_nm_right
                span = max(0.05, self.config.force_haptic_max_nm - threshold)
                self._force_haptic = float(
                    np.clip((self._gripper_force_proxy_nm - threshold) / span, 0.0, 1.0)
                )
            else:
                self._force_haptic = 0.0
        self._publish_ik_state()
        return result

    def is_engaged(self) -> bool:
        """Return the live right-grip clutch level for HIL orchestrators."""
        with self._lock:
            return self._buttons["right"]["grip"]

    def is_handoff_pressed(self) -> bool:
        """Return the Quest B/Y handoff level for HIL orchestrators."""
        with self._lock:
            return self._buttons["right"]["handoff"] or self._buttons["left"]["handoff"]

    def is_pause_pressed(self) -> bool:
        """Expose the right-hand B button as a policy pause signal."""
        with self._lock:
            return self._buttons["right"]["handoff"]

    def is_reverse_pressed(self) -> bool:
        """Expose the left-hand Y button as a reverse signal."""
        with self._lock:
            return self._buttons["left"]["handoff"]

    def seed_qpos_from_obs(self, observation: dict[str, float]) -> None:
        """Seed the Kinova IK from an observation before a human handoff."""
        joints = [float(observation[f"joint_id_{index}.pos"]) for index in range(7)]
        if not np.all(np.isfinite(joints)):
            raise ValueError("handoff observation contains non-finite joint positions")
        with self._lock:
            self._joint_target_rad = np.deg2rad(np.asarray(joints, dtype=np.float64))
            if "gripper.pos" in observation:
                self._gripper_target = float(np.clip(observation["gripper.pos"], 0.0, 1.0))
            self._engaged = False
            self._last_grip = False
            self._pos_filter = None
            self._quat_filter = None
            if self._mapper is not None:
                self._mapper.disengage()

    def publish_state(self) -> None:
        """Publish the latest Kinova qpos and haptic state to the Quest UI."""
        self._publish_ik_state()

    def _update_from_frame(self, frame: dict[str, Any], settings: dict[str, float]) -> None:
        controller = (frame.get("controllers") or {}).get("right")
        if controller is None:
            if self._engaged:
                self._disengage()
            return
        position = np.asarray(controller.get("position", []), dtype=np.float64).reshape(-1)
        orientation_xyzw = np.asarray(controller.get("orientation", []), dtype=np.float64).reshape(-1)
        if (
            position.size != 3
            or orientation_xyzw.size != 4
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(orientation_xyzw))
            or np.linalg.norm(orientation_xyzw) < 1e-8
        ):
            return
        orientation = np.array(
            [orientation_xyzw[3], orientation_xyzw[0], orientation_xyzw[1], orientation_xyzw[2]],
            dtype=np.float64,
        )
        orientation /= np.linalg.norm(orientation)

        alpha = settings["pose_filter_alpha"]
        if self._pos_filter is None or self._quat_filter is None:
            self._pos_filter = position.copy()
            self._quat_filter = orientation.copy()
        else:
            self._pos_filter = (1.0 - alpha) * self._pos_filter + alpha * position
            aligned = orientation if np.dot(self._quat_filter, orientation) >= 0.0 else -orientation
            self._quat_filter = (1.0 - alpha) * self._quat_filter + alpha * aligned
            self._quat_filter /= np.linalg.norm(self._quat_filter)

        buttons = controller.get("buttons") or []
        grip = bool(buttons[_GRIP_BUTTON_INDEX].get("p")) if len(buttons) > _GRIP_BUTTON_INDEX else False
        trigger = float(buttons[_TRIGGER_BUTTON_INDEX].get("v", 0.0)) if buttons else 0.0
        precision = (
            bool(buttons[_PRECISION_BUTTON_INDEX].get("p"))
            if len(buttons) > _PRECISION_BUTTON_INDEX
            else False
        )
        home_button = (
            bool(buttons[_HOME_BUTTON_INDEX].get("p")) if len(buttons) > _HOME_BUTTON_INDEX else False
        )
        trigger = float(np.clip(trigger, 0.0, 1.0))
        yaw = None
        viewer_orientation = (frame.get("viewer") or {}).get("orientation")
        if viewer_orientation is not None and len(viewer_orientation) == 4:
            yaw = _yaw_from_quat_xyzw(viewer_orientation)

        if grip and not self._last_grip:
            self._anchor(self._pos_filter, self._quat_filter, yaw)
        elif not grip and self._last_grip:
            self._disengage()
        self._last_grip = grip

        if precision != self._last_precision and self._engaged:
            self._anchor(self._pos_filter, self._quat_filter, yaw)
        self._last_precision = precision

        if home_button and not self._last_home_button and self._home_ramp is None:
            assert self._solver is not None
            self._home_ramp = {
                "start_rad": self._joint_target_rad.copy(),
                "target_rad": self._solver.rest_qpos.copy(),
                "start_time": time.perf_counter(),
            }
        self._last_home_button = home_button

        if self._home_ramp is not None:
            ramp = self._home_ramp
            fraction = min(
                1.0,
                (time.perf_counter() - ramp["start_time"]) / max(0.1, self.config.home_ramp_duration_s),
            )
            self._joint_target_rad = ramp["start_rad"] + fraction * (ramp["target_rad"] - ramp["start_rad"])
            if fraction >= 1.0:
                self._home_ramp = None
                if self._engaged:
                    self._anchor(self._pos_filter, self._quat_filter, yaw)
            self._haptic *= 0.6
            return

        if self._engaged:
            assert self._mapper is not None and self._solver is not None
            factor = settings["precision_factor"] if precision else 1.0
            self._mapper.scale = settings["scale_translation"] * factor
            self._mapper.scale_rotation = settings["scale_rotation"] * factor
            self._mapper.pos_reach_limit = settings["pos_reach_limit"] or None
            self._mapper.rot_reach_limit = settings["rot_reach_limit"] or None
            self._solver.max_joint_step_rad = np.array(
                [self._max_dq_pos] * 3 + [self._max_dq_rot] * 4,
                dtype=np.float64,
            )
            self._gripper_target = trigger

            current_pos, current_quat = self._solver.fk(self._joint_target_rad)
            target = self._mapper.target(self._pos_filter, self._quat_filter, current_pos, current_quat)
            if target is not None:
                self._joint_target_rad = self._solver.solve(target[0], target[1], self._joint_target_rad)

            pressure = self._solver.last_joint_limit_pressure
            position_error = self._solver.last_position_error_m
            rotation_error = self._solver.last_rotation_error_rad
            singularity = self._solver.last_singularity_proximity
            wrist_singularity = self._solver.last_wrist_singularity_proximity
            raw_haptic = max(
                np.clip(pressure / 0.25, 0.0, 1.0),
                np.clip(position_error / 0.10, 0.0, 1.0),
                np.clip(rotation_error / 0.8, 0.0, 1.0),
                np.clip((singularity - 0.8) / 0.2, 0.0, 1.0),
                np.clip((wrist_singularity - 0.5) / 0.5, 0.0, 1.0),
            )
            self._haptic = 0.6 * self._haptic + 0.4 * float(raw_haptic)
        else:
            self._haptic *= 0.6

    def _anchor(self, controller_pos: np.ndarray, controller_quat: np.ndarray, yaw: float | None) -> None:
        assert self._mapper is not None and self._solver is not None
        if yaw is not None:
            calibrated = np.asarray(self.config.r_calib, dtype=np.float64) @ _rotation_y(-yaw)
            self._mapper.set_R(calibrated)
        ee_pos, ee_quat = self._solver.fk(self._joint_target_rad)
        wrist_pivot = self._solver.wrist_pivot_xpos(self._joint_target_rad)
        self._mapper.engage(
            controller_pos,
            controller_quat,
            ee_pos,
            ee_quat,
            pivot_armbase=wrist_pivot,
        )
        self._engaged = True

    def _disengage(self) -> None:
        if self._mapper is not None:
            self._mapper.disengage()
        self._engaged = False
        self._last_grip = False
        self._pos_filter = None
        self._quat_filter = None

    def _publish_ik_state(self) -> None:
        if self._ws is None or self._ws_loop is None:
            return
        payload = {
            "type": "ik_state",
            "qpos": [float(value) for value in self._joint_target_rad] + [float(self._gripper_target)],
            "right_qpos": [float(value) for value in self._joint_target_rad] + [float(self._gripper_target)],
            "engaged": self._engaged,
            "right_engaged": self._engaged,
            "right_haptic": float(self._haptic),
            "right_force_haptic": float(self._force_haptic),
            "loop_hz": float(self._loop_hz or 0.0),
        }
        try:
            asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(payload)), self._ws_loop)
        except RuntimeError:
            logger.debug("Could not publish VR IK state because the relay loop has stopped")

    def _apply_web_settings(self, values: dict[str, Any]) -> None:
        numeric_fields = {
            "scale_translation": (0.05, 3.0),
            "scale_rotation": (0.0, 3.0),
            "precision_factor": (0.05, 1.0),
            "pose_filter_alpha": (0.05, 1.0),
            "pos_reach_limit": (0.0, 1.0),
            "rot_reach_limit": (0.0, 3.14),
            "force_haptic_threshold_nm_left": (0.0, 2.0),
            "force_haptic_threshold_nm_right": (0.0, 2.0),
        }
        with self._lock:
            for name, (lower, upper) in numeric_fields.items():
                if name not in values:
                    continue
                try:
                    value = float(values[name])
                except (TypeError, ValueError):
                    logger.warning("Ignoring invalid VR setting %s=%r", name, values[name])
                    continue
                if np.isfinite(value):
                    setattr(self.config, name, float(np.clip(value, lower, upper)))
            if "max_dq_per_joint_scalar_pos" in values:
                try:
                    value = float(values["max_dq_per_joint_scalar_pos"])
                except (TypeError, ValueError):
                    value = self._max_dq_pos
                if np.isfinite(value):
                    self._max_dq_pos = float(np.clip(value, 0.001, _MAX_JOINT_STEP_RAD))
            if "max_dq_per_joint_scalar_rot" in values:
                try:
                    value = float(values["max_dq_per_joint_scalar_rot"])
                except (TypeError, ValueError):
                    value = self._max_dq_rot
                if np.isfinite(value):
                    self._max_dq_rot = float(np.clip(value, 0.001, _MAX_JOINT_STEP_RAD))
            if "force_haptic_enabled" in values:
                self.config.force_haptic_enabled = bool(values["force_haptic_enabled"])

    def _update_haptic_calibration(self) -> None:
        result: dict[str, float] | None = None
        with self._lock:
            calibration = self._haptic_calibration
            if (
                calibration is None
                or time.monotonic() - calibration["start_time"] < calibration["duration_s"]
            ):
                return
            peak = calibration["peak_nm"]
            threshold = min(1.0, peak + 0.05)
            self.config.force_haptic_threshold_nm_right = threshold
            self.config.force_haptic_max_nm = max(self.config.force_haptic_max_nm, threshold + 0.25)
            result = {
                "type": "haptic_calibrate_result",
                "left_peak_nm": 0.0,
                "right_peak_nm": peak,
                "left_threshold_nm": self.config.force_haptic_threshold_nm_left,
                "right_threshold_nm": threshold,
                "margin_nm": 0.05,
                "max_nm": self.config.force_haptic_max_nm,
                "suspicious": [],
            }
            self._haptic_calibration = None
        if result is not None:
            self._send_ws_payload(result)

    def _publish_sim_camera(self, image: np.ndarray) -> None:
        frame = np.asarray(image)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            logger.debug("Skipping invalid simulated wrist camera frame: %s %s", frame.shape, frame.dtype)
            return
        try:
            self._camera_queue.put_nowait(frame.copy())
        except queue.Full:
            try:
                self._camera_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._camera_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

    def _camera_publish_loop(self) -> None:
        while not self._camera_stop.is_set():
            try:
                frame = self._camera_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            wait_s = 1.0 / 15.0 - (time.monotonic() - self._camera_last_sent)
            if wait_s > 0.0 and self._camera_stop.wait(wait_s):
                break
            if self._ws is None or self._ws_loop is None:
                continue
            self._camera_last_sent = time.monotonic()
            self._encode_and_send_camera_frame(frame)

    def _encode_and_send_camera_frame(self, frame: np.ndarray) -> None:
        try:
            import cv2

            success, encoded = cv2.imencode(".jpg", frame[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not success:
                return
            payload = {
                "type": "sim_camera_frame",
                "camera_id": "sim_wrist",
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "jpeg": base64.b64encode(encoded).decode("ascii"),
            }
            if not self._camera_registered:
                self._send_ws_payload(
                    {
                        "type": "sim_camera_register",
                        "camera_id": "sim_wrist",
                        "label": "Kinova simulated wrist",
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                        "fps": 15,
                    }
                )
                self._camera_registered = True
            self._send_ws_payload(payload)
        except Exception:
            logger.debug("Could not encode simulated wrist camera frame", exc_info=True)

    def _send_ws_payload(self, payload: dict[str, Any]) -> None:
        websocket, loop = self._ws, self._ws_loop
        if websocket is None or loop is None:
            return
        try:
            future = asyncio.run_coroutine_threadsafe(websocket.send(json.dumps(payload)), loop)
            future.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        except RuntimeError:
            logger.debug("Could not queue VR relay message because its loop has stopped")

    def _ws_thread_main(self) -> None:
        self._ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._ws_loop)
        try:
            self._ws_loop.run_until_complete(self._ws_runner())
        finally:
            self._ws_loop.close()
            self._ws_loop = None

    async def _ws_runner(self) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError('Install the VR dependencies with `pip install -e ".[vr-teleop]"`.') from exc

        backoff = 0.25
        while self._ws_stop is not None and not self._ws_stop.is_set():
            try:
                ssl_context = self._ssl_context_for(self.config.ws_url)
                async with websockets.connect(self.config.ws_url, ssl=ssl_context) as websocket:
                    self._ws = websocket
                    self._ws_connected.set()
                    backoff = 0.25
                    logger.info("Connected to VR relay %s", self.config.ws_url)
                    await websocket.send(json.dumps({"type": "request_settings"}))
                    async for raw in websocket:
                        if self._ws_stop.is_set():
                            break
                        try:
                            message = json.loads(raw)
                        except (TypeError, json.JSONDecodeError):
                            continue
                        if message.get("type") == "xr_frame":
                            with self._lock:
                                self._latest_frame = message
                                self._last_frame_time = time.time()
                                controllers = message.get("controllers") or {}
                                for hand in ("left", "right"):
                                    buttons = (controllers.get(hand) or {}).get("buttons") or []
                                    self._buttons[hand] = {
                                        "grip": bool(
                                            buttons[_GRIP_BUTTON_INDEX].get("p")
                                            if len(buttons) > _GRIP_BUTTON_INDEX
                                            else False
                                        ),
                                        "handoff": bool(
                                            buttons[_HANDOFF_BUTTON_INDEX].get("p")
                                            if len(buttons) > _HANDOFF_BUTTON_INDEX
                                            else False
                                        ),
                                    }
                        elif message.get("type") == "config_update":
                            self._apply_web_settings(message.get("config") or {})
                        elif message.get("type") == "haptic_calibrate":
                            with self._lock:
                                duration = float(np.clip(message.get("duration_s", 5.0), 0.5, 10.0))
                                self._haptic_calibration = {
                                    "start_time": time.monotonic(),
                                    "duration_s": duration,
                                    "peak_nm": self._gripper_force_proxy_nm,
                                }
                        elif message.get("type") == "camera_list":
                            cameras = message.get("cameras") or []
                            self._camera_registered = any(
                                camera.get("id") == "sim_wrist" for camera in cameras
                            )
            except Exception as exc:
                if not self._ws_stop.is_set():
                    logger.warning(
                        "VR relay connection error (%s); retrying in %.2f s", type(exc).__name__, backoff
                    )
            finally:
                self._ws = None
                self._ws_connected.clear()
                self._camera_registered = False
                with self._lock:
                    self._latest_frame = None
                    self._last_frame_time = 0.0
            if self._ws_stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 5.0)

    @staticmethod
    def _ssl_context_for(url: str):
        """Allow the relay's self-signed TLS certificate for localhost WSS."""
        if not url.startswith("wss://") or ("//localhost" not in url and "//127.0.0.1" not in url):
            return None
        import ssl

        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
