import contextlib
import copy
import logging
import math
import os
import threading
import time
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_kinova_gen3 import KinovaGen3Config

logger = logging.getLogger(__name__)

_MAX_JOINT_SPEED_DEG_S = 3.0
_MAX_GRIPPER_SPEED = 0.25
_MAX_MEASUREMENT_AGE_S = 0.5


def _kortex():
    # Kortex 2.8.0.post5 ships protobuf stubs generated for the legacy runtime.
    # Keep the compatibility mode local to the Kortex import so the rest of the
    # current LeRobot environment can continue using its newer protobuf runtime.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    try:
        from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
        from kortex_api.autogen.messages import Base_pb2, Session_pb2
        from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
        from kortex_api.SessionManager import SessionManager
        from kortex_api.TCPTransport import TCPTransport
    except ImportError as exc:
        raise ImportError(
            "Kinova Gen3 control requires the Kinova Kortex Python API. "
            "Install the official Kortex API wheel (2.5 or 2.8) in this environment."
        ) from exc

    return (
        RouterClient,
        RouterClientSendOptions,
        SessionManager,
        TCPTransport,
        BaseClient,
        Base_pb2,
        Session_pb2,
    )


class KinovaGen3(Robot):
    """Kinova Gen3 high-level joint position teleop endpoint.

    The robot receives desired positions from LeRobot and internally streams
    proportional joint velocities through Kortex. Optional camera streams are
    exposed as live observations for teleoperation visualization.
    """

    config_class = KinovaGen3Config
    name = "kinova_gen3"

    def __init__(self, config: KinovaGen3Config):
        super().__init__(config)
        self.config = config
        camera_configs = {}
        default_stream = "rtsp://192.168.1.10/color"
        for name, camera_config in config.cameras.items():
            if camera_config.type == "kinova_gen3_vision" and camera_config.index_or_path == default_stream:
                camera_config = copy.copy(camera_config)
                camera_config.index_or_path = f"rtsp://{config.ip}/color"
            camera_configs[name] = camera_config
        self.cameras = make_cameras_from_configs(camera_configs)
        self.transport = None
        self.router = None
        self.base = None
        self.session_manager = None
        self._connected = False
        self._measure_thread: threading.Thread | None = None
        self._tracker_thread: threading.Thread | None = None
        self._measure_stop = threading.Event()
        self._tracker_stop = threading.Event()
        self._control_fault = threading.Event()
        self._state_lock = threading.Lock()
        self._target_lock = threading.Lock()
        self._joint_positions = [0.0] * 7
        self._last_measurement_time = 0.0
        self._target_positions = [0.0] * 7
        self._gripper_position = 0.0
        self._gripper_target = 0.0
        self._last_gripper_speed = 0.0
        self._last_gripper_time = 0.0

    @property
    def observation_features(self) -> dict[str, type]:
        features = {**{f"joint_id_{i}.pos": float for i in range(7)}, "gripper.pos": float}
        features.update({name: (camera.height, camera.width, 3) for name, camera in self.cameras.items()})
        return features

    @property
    def action_features(self) -> dict[str, type]:
        # Images belong to observations only; actions are joint/gripper targets.
        return {**{f"joint_id_{i}.pos": float for i in range(7)}, "gripper.pos": float}

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
        self._validate_motion_limits()
        router_client, _, session_manager_cls, tcp_transport, base_client, base_pb2, session_pb2 = _kortex()
        self.transport = tcp_transport()
        self.router = router_client(self.transport, router_client.basicErrorCallback)
        self.transport.connect(self.config.ip, self.config.port)
        if self.config.kinova_user:
            info = session_pb2.CreateSessionInfo(
                username=self.config.kinova_user,
                password=self.config.kinova_pwd,
                session_inactivity_timeout=self.config.session_inactivity_timeout,
                connection_inactivity_timeout=self.config.connection_inactivity_timeout,
            )
            self.session_manager = session_manager_cls(self.router)
            self.session_manager.CreateSession(info)
        self.base = base_client(self.router)
        request = base_pb2.GripperRequest(mode=base_pb2.GRIPPER_POSITION)
        joints = self.base.GetMeasuredJointAngles().joint_angles
        measured = [float(j.value) for j in sorted(joints, key=lambda j: j.joint_identifier)]
        if len(measured) != 7:
            raise RuntimeError(f"Expected a 7-DoF Gen3, got {len(measured)} joints")
        gripper = float(self.base.GetMeasuredGripperMovement(request).finger[0].value)
        with self._state_lock:
            self._joint_positions = measured
            self._gripper_position = gripper
            self._last_measurement_time = time.monotonic()
        with self._target_lock:
            self._target_positions = measured.copy()
            self._gripper_target = gripper
        mode = base_pb2.ServoingModeInformation(servoing_mode=base_pb2.SINGLE_LEVEL_SERVOING)
        self.base.SetServoingMode(mode)
        if self.cameras:
            from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
            from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient

            device_manager = DeviceManagerClient(self.router)
            vision_config = VisionConfigClient(self.router)
            try:
                for camera in self.cameras.values():
                    if camera.config.type == "kinova_gen3_vision":
                        camera.connect(vision_config, device_manager)
                    else:
                        camera.connect()
            except BaseException:
                for camera in self.cameras.values():
                    if camera.is_connected:
                        with contextlib.suppress(Exception):
                            camera.disconnect()
                if self.session_manager is not None:
                    with contextlib.suppress(Exception):
                        _, router_options_cls, _, _, _, _, _ = _kortex()
                        self.session_manager.CloseSession(router_options_cls())
                with contextlib.suppress(Exception):
                    self.transport.disconnect()
                self.base = None
                self.transport = None
                self.session_manager = None
                raise
        self._connected = True
        self._control_fault.clear()
        self._start_measure_thread()
        self._start_tracker_thread()
        logger.info("%s connected", self)

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        self._tracker_stop.set()
        self._measure_stop.set()
        for camera in self.cameras.values():
            if camera.is_connected:
                with contextlib.suppress(Exception):
                    camera.disconnect()
        for thread in (self._tracker_thread, self._measure_thread):
            if thread:
                thread.join(timeout=2.0)
        with contextlib.suppress(Exception):
            self._send_joint_speeds([0.0] * 7)
        with contextlib.suppress(Exception):
            self._send_gripper_speed(0.0)
        if self.session_manager is not None:
            with contextlib.suppress(Exception):
                _, router_options_cls, _, _, _, _, _ = _kortex()
                options = router_options_cls()
                options.timeout_ms = 1000
                self.session_manager.CloseSession(options)
        if self.transport is not None:
            with contextlib.suppress(Exception):
                self.transport.disconnect()
        self._connected = False
        self._tracker_thread = None
        self._measure_thread = None
        self.base = None

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        with self._state_lock:
            joints = self._joint_positions.copy()
            gripper = self._gripper_position
        observation = {**{f"joint_id_{i}.pos": joints[i] for i in range(7)}, "gripper.pos": gripper}
        for name, camera in self.cameras.items():
            observation[name] = camera.async_read(timeout_ms=500)
        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        joint_targets = {i: float(action[key]) for i in range(7) if (key := f"joint_id_{i}.pos") in action}
        if not all(math.isfinite(target) for target in joint_targets.values()):
            raise ValueError("Kinova joint targets must be finite")
        gripper_target = float(action["gripper.pos"]) if "gripper.pos" in action else None
        if gripper_target is not None and not math.isfinite(gripper_target):
            raise ValueError("gripper.pos must be finite")
        with self._target_lock:
            for index, target in joint_targets.items():
                self._target_positions[index] = target
            if gripper_target is not None:
                self._gripper_target = max(0.0, min(1.0, gripper_target))
        return action

    def _validate_motion_limits(self) -> None:
        config = self.config
        limits = (
            config.control_rate_hz,
            config.position_gain,
            config.max_joint_speed_deg_s,
            config.joint_deadband_deg,
            config.gripper_rate_hz,
            config.gripper_gain,
            config.gripper_deadband,
            config.gripper_max_speed,
        )
        if not all(math.isfinite(value) for value in limits):
            raise ValueError("Kinova motion settings must all be finite numbers")
        if not 0.0 < config.max_joint_speed_deg_s <= _MAX_JOINT_SPEED_DEG_S:
            raise ValueError(
                f"max_joint_speed_deg_s must be in (0, {_MAX_JOINT_SPEED_DEG_S}] for this integration"
            )
        if not 0.0 < config.gripper_max_speed <= _MAX_GRIPPER_SPEED:
            raise ValueError(f"gripper_max_speed must be in (0, {_MAX_GRIPPER_SPEED}]")
        if not 1.0 <= config.control_rate_hz <= 100.0:
            raise ValueError("control_rate_hz must be between 1 and 100 Hz")
        if not 1.0 <= config.gripper_rate_hz <= 50.0:
            raise ValueError("gripper_rate_hz must be between 1 and 50 Hz")
        if not 0.0 <= config.position_gain <= 2.0 or not 0.0 <= config.gripper_gain <= 0.5:
            raise ValueError("Kinova position and gripper gains exceed the allowed software limits")
        if config.joint_deadband_deg < 0.0 or config.gripper_deadband < 0.0:
            raise ValueError("Kinova deadbands must be non-negative")

    def _start_measure_thread(self) -> None:
        self._measure_stop.clear()
        self._measure_thread = threading.Thread(target=self._measure_loop, daemon=True, name="kinova-measure")
        self._measure_thread.start()

    def _start_tracker_thread(self) -> None:
        self._tracker_stop.clear()
        self._tracker_thread = threading.Thread(target=self._tracker_loop, daemon=True, name="kinova-tracker")
        self._tracker_thread.start()

    def _measure_loop(self) -> None:
        _, _, _, _, _, base_pb2, _ = _kortex()
        request = base_pb2.GripperRequest(mode=base_pb2.GRIPPER_POSITION)
        period = 1.0 / max(1.0, self.config.control_rate_hz)
        while not self._measure_stop.is_set():
            try:
                joints = self.base.GetMeasuredJointAngles().joint_angles
                values = [float(j.value) for j in sorted(joints, key=lambda j: j.joint_identifier)]
                gripper = float(self.base.GetMeasuredGripperMovement(request).finger[0].value)
                if (
                    len(values) != 7
                    or not all(math.isfinite(value) for value in values)
                    or not math.isfinite(gripper)
                ):
                    raise RuntimeError("Kinova returned invalid joint or gripper measurements")
                with self._state_lock:
                    self._joint_positions = values
                    self._gripper_position = gripper
                    self._last_measurement_time = time.monotonic()
            except Exception:
                logger.exception("Kinova state read failed; stopping motion commands")
                self._control_fault.set()
                return
            time.sleep(period)

    def _tracker_loop(self) -> None:
        period = 1.0 / max(1.0, self.config.control_rate_hz)
        while not self._tracker_stop.is_set():
            if self._control_fault.is_set():
                self._send_stop_commands()
                return
            with self._target_lock:
                targets = self._target_positions.copy()
                gripper_target = self._gripper_target
            with self._state_lock:
                current = self._joint_positions.copy()
                gripper = self._gripper_position
                measurement_age = time.monotonic() - self._last_measurement_time
            if measurement_age > _MAX_MEASUREMENT_AGE_S:
                logger.error("Kinova measurements are stale; stopping motion commands")
                self._control_fault.set()
                self._send_stop_commands()
                return
            speeds = []
            speed_limit = min(self.config.max_joint_speed_deg_s, _MAX_JOINT_SPEED_DEG_S)
            for target, measured in zip(targets, current, strict=True):
                error = (target - measured + 180.0) % 360.0 - 180.0
                speed = (
                    0.0 if abs(error) <= self.config.joint_deadband_deg else self.config.position_gain * error
                )
                speeds.append(max(-speed_limit, min(speed_limit, speed)))
            if self.config.no_op:
                speeds = [0.0] * 7
            try:
                self._send_joint_speeds(speeds)
            except Exception:
                logger.exception("Kinova joint command failed; stopping motion commands")
                self._control_fault.set()
                self._send_stop_commands()
                return
            error = gripper_target - gripper
            grip_speed = (
                0.0 if abs(error) <= self.config.gripper_deadband else self.config.gripper_gain * error
            )
            gripper_speed_limit = min(self.config.gripper_max_speed, _MAX_GRIPPER_SPEED)
            grip_speed = max(-gripper_speed_limit, min(gripper_speed_limit, grip_speed))
            now = time.perf_counter()
            if abs(
                grip_speed - self._last_gripper_speed
            ) > 1e-3 or now - self._last_gripper_time >= 1.0 / max(1.0, self.config.gripper_rate_hz):
                try:
                    self._send_gripper_speed(-grip_speed)
                except Exception:
                    logger.exception("Kinova gripper command failed; stopping motion commands")
                    self._control_fault.set()
                    self._send_stop_commands()
                    return
            time.sleep(period)

    def _send_stop_commands(self) -> None:
        if self.base is not None:
            with contextlib.suppress(Exception):
                self.base.Stop()
        with contextlib.suppress(Exception):
            self._send_joint_speeds([0.0] * 7)
        with contextlib.suppress(Exception):
            self._send_gripper_speed(0.0)

    def _send_joint_speeds(self, speeds: list[float]) -> None:
        if len(speeds) != 7 or not all(math.isfinite(speed) for speed in speeds):
            raise ValueError("Kinova joint speed command must contain seven finite values")
        _, _, _, _, _, base_pb2, _ = _kortex()
        command = base_pb2.JointSpeeds()
        for i, speed in enumerate(speeds):
            bounded_speed = max(-_MAX_JOINT_SPEED_DEG_S, min(_MAX_JOINT_SPEED_DEG_S, float(speed)))
            command.joint_speeds.add(joint_identifier=i, value=bounded_speed, duration=0)
        self.base.SendJointSpeedsCommand(command)

    def _send_gripper_speed(self, speed: float) -> None:
        if not math.isfinite(speed):
            raise ValueError("Kinova gripper speed must be finite")
        speed = max(-_MAX_GRIPPER_SPEED, min(_MAX_GRIPPER_SPEED, float(speed)))
        _, _, _, _, _, base_pb2, _ = _kortex()
        command = base_pb2.GripperCommand(mode=base_pb2.GRIPPER_SPEED)
        command.gripper.finger.add(finger_identifier=0, value=float(speed))
        self.base.SendGripperCommand(command)
        self._last_gripper_speed = speed
        self._last_gripper_time = time.perf_counter()
