import contextlib
import copy
import logging
import os
import threading
import time
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_kinova_gen3 import KinovaGen3Config

logger = logging.getLogger(__name__)


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

    return RouterClient, RouterClientSendOptions, SessionManager, TCPTransport, BaseClient, Base_pb2, Session_pb2


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
        self._state_lock = threading.Lock()
        self._target_lock = threading.Lock()
        self._joint_positions = [0.0] * 7
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
        with self._target_lock:
            for i in range(7):
                key = f"joint_id_{i}.pos"
                if key in action:
                    self._target_positions[i] = float(action[key])
            if "gripper.pos" in action:
                self._gripper_target = max(0.0, min(1.0, float(action["gripper.pos"])))
        return action

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
                with self._state_lock:
                    self._joint_positions = values
                    self._gripper_position = gripper
            except Exception:
                logger.exception("Kinova state read failed")
                return
            time.sleep(period)

    def _tracker_loop(self) -> None:
        period = 1.0 / max(1.0, self.config.control_rate_hz)
        while not self._tracker_stop.is_set():
            with self._target_lock:
                targets = self._target_positions.copy()
                gripper_target = self._gripper_target
            with self._state_lock:
                current = self._joint_positions.copy()
                gripper = self._gripper_position
            speeds = []
            for target, measured in zip(targets, current, strict=True):
                error = (target - measured + 180.0) % 360.0 - 180.0
                speed = 0.0 if abs(error) <= self.config.joint_deadband_deg else self.config.position_gain * error
                speeds.append(max(-self.config.max_joint_speed_deg_s, min(self.config.max_joint_speed_deg_s, speed)))
            if self.config.no_op:
                speeds = [0.0] * 7
            with contextlib.suppress(Exception):
                self._send_joint_speeds(speeds)
            error = gripper_target - gripper
            grip_speed = 0.0 if abs(error) <= self.config.gripper_deadband else self.config.gripper_gain * error
            grip_speed = max(-self.config.gripper_max_speed, min(self.config.gripper_max_speed, grip_speed))
            now = time.perf_counter()
            if abs(grip_speed - self._last_gripper_speed) > 1e-3 or now - self._last_gripper_time >= 1.0 / max(1.0, self.config.gripper_rate_hz):
                with contextlib.suppress(Exception):
                    self._send_gripper_speed(-grip_speed)
            time.sleep(period)

    def _send_joint_speeds(self, speeds: list[float]) -> None:
        _, _, _, _, _, base_pb2, _ = _kortex()
        command = base_pb2.JointSpeeds()
        for i, speed in enumerate(speeds):
            command.joint_speeds.add(joint_identifier=i, value=float(speed), duration=0)
        self.base.SendJointSpeedsCommand(command)

    def _send_gripper_speed(self, speed: float) -> None:
        _, _, _, _, _, base_pb2, _ = _kortex()
        command = base_pb2.GripperCommand(mode=base_pb2.GRIPPER_SPEED)
        command.gripper.finger.add(finger_identifier=0, value=float(speed))
        self.base.SendGripperCommand(command)
        self._last_gripper_speed = speed
        self._last_gripper_time = time.perf_counter()
