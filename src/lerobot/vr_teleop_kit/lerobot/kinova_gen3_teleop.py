"""Quest WebXR teleoperator for the Kinova Gen3 MuJoCo simulator."""

from __future__ import annotations

import asyncio
import json
import logging
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
_STALE_FRAME_S = 0.2


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
    damping: float = 0.08
    posture_gain: float = 0.02
    rest_qpos: list[float] | None = None
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
        """Return whether the relay WebSocket is active."""
        return self._connected and self._ws_connected.is_set()

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
        if self.config.max_joint_step_rad <= 0.0:
            raise ValueError("max_joint_step_rad must be greater than zero")
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
        logger.info("%s connected to WebXR relay %s", self, self.config.ws_url)

    def disconnect(self) -> None:
        """Stop the relay connection and release the current clutch."""
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
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Synchronize the IK seed with the current simulated robot state."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        joint_values = [float(feedback[f"joint_id_{index}.pos"]) for index in range(7)]
        if not np.all(np.isfinite(joint_values)):
            raise ValueError("Kinova simulation feedback contains non-finite joint positions")
        self._joint_target_rad = np.deg2rad(np.asarray(joint_values, dtype=np.float64))
        if "gripper.pos" in feedback:
            self._gripper_target = float(np.clip(float(feedback["gripper.pos"]), 0.0, 1.0))

    def get_action(self) -> dict[str, float]:
        """Convert the latest right-hand WebXR pose to Gen3 joint targets."""
        if not self.is_connected:
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

        result = {
            f"joint_id_{index}.pos": float(np.rad2deg(value))
            for index, value in enumerate(self._joint_target_rad)
        }
        result["gripper.pos"] = float(self._gripper_target)
        self._publish_ik_state()
        return result

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

        if self._engaged:
            assert self._mapper is not None and self._solver is not None
            factor = settings["precision_factor"] if precision else 1.0
            self._mapper.scale = settings["scale_translation"] * factor
            self._mapper.scale_rotation = settings["scale_rotation"] * factor
            self._mapper.pos_reach_limit = settings["pos_reach_limit"] or None
            self._mapper.rot_reach_limit = settings["rot_reach_limit"] or None
            self._gripper_target = trigger

            current_pos, current_quat = self._solver.fk(self._joint_target_rad)
            target = self._mapper.target(self._pos_filter, self._quat_filter, current_pos, current_quat)
            if target is not None:
                self._joint_target_rad = self._solver.solve(target[0], target[1], self._joint_target_rad)

            pressure = self._solver.last_joint_limit_pressure
            position_error = self._solver.last_position_error_m
            rotation_error = self._solver.last_rotation_error_rad
            raw_haptic = max(
                np.clip(pressure / 0.25, 0.0, 1.0),
                np.clip(position_error / 0.10, 0.0, 1.0),
                np.clip(rotation_error / 0.8, 0.0, 1.0),
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
            "engaged": self._engaged,
            "right_engaged": self._engaged,
            "right_haptic": float(self._haptic),
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
        }
        with self._lock:
            for name, (lower, upper) in numeric_fields.items():
                if name not in values:
                    continue
                value = float(values[name])
                if np.isfinite(value):
                    setattr(self.config, name, float(np.clip(value, lower, upper)))

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
                async with websockets.connect(self.config.ws_url) as websocket:
                    self._ws = websocket
                    self._ws_connected.set()
                    backoff = 0.25
                    logger.info("Connected to VR relay %s", self.config.ws_url)
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
                        elif message.get("type") == "config_update":
                            self._apply_web_settings(message.get("config") or {})
            except Exception as exc:
                if not self._ws_stop.is_set():
                    logger.warning(
                        "VR relay connection error (%s); retrying in %.2f s", type(exc).__name__, backoff
                    )
            finally:
                self._ws = None
                self._ws_connected.clear()
                with self._lock:
                    self._latest_frame = None
                    self._last_frame_time = 0.0
            if self._ws_stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 5.0)
