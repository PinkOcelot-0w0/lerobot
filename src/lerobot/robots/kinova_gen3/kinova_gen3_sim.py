import contextlib
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_kinova_gen3_sim import KinovaGen3SimConfig
from .mujoco_model import load_kinova_gen3_mujoco_model

logger = logging.getLogger(__name__)

_JOINT_NAMES = tuple(f"joint_{i}" for i in range(1, 8))
_GRIPPER_JOINT_NAME = "gripper_right_driver_joint"
_GRIPPER_ACTUATOR_NAME = "gripper_fingers_actuator"
_GRIPPER_JOINT_RANGE = 0.8
_GRIPPER_ACTUATOR_RANGE = 255.0


class KinovaGen3Sim(Robot):
    """MuJoCo-backed Kinova Gen3 Robot using the same action units as hardware."""

    config_class = KinovaGen3SimConfig
    name = "kinova_gen3_sim"
    supports_position_hold = True

    def __init__(self, config: KinovaGen3SimConfig):
        super().__init__(config)
        self.config = config
        self.model = None
        self.data = None
        self.viewer = None
        self.renderers: dict[str, Any] = {}
        self._connected = False
        self._stop_event = threading.Event()
        self._simulation_thread: threading.Thread | None = None
        self._state_lock = threading.RLock()
        self._joint_qpos_addresses: list[int] = []
        self._joint_actuator_ids: list[int] = []
        self._gripper_qpos_address: int | None = None
        self._gripper_actuator_id: int | None = None

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        features: dict[str, type | tuple[int, int, int]] = {
            **{f"joint_id_{i}.pos": float for i in range(7)},
            "gripper.pos": float,
        }
        features.update(
            dict.fromkeys(
                self.config.render_cameras,
                (self.config.render_height, self.config.render_width, 3),
            )
        )
        return features

    @property
    def action_features(self) -> dict[str, type]:
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
        import mujoco

        if self.config.simulation_timestep <= 0:
            raise ValueError("simulation_timestep must be greater than zero")
        if self.config.viewer_sync_hz <= 0:
            raise ValueError("viewer_sync_hz must be greater than zero")

        self.model = load_kinova_gen3_mujoco_model(self.config.model_xml, self.config.gripper_xml)
        self.model.opt.timestep = self.config.simulation_timestep
        self.data = mujoco.MjData(self.model)
        self._joint_qpos_addresses = []
        self._joint_actuator_ids = []
        for name in _JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if joint_id < 0 or actuator_id < 0:
                raise ValueError(f"Kinova model must contain joint and position actuator '{name}'")
            self._joint_qpos_addresses.append(int(self.model.jnt_qposadr[joint_id]))
            self._joint_actuator_ids.append(actuator_id)

        gripper_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, _GRIPPER_JOINT_NAME
        )
        self._gripper_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, _GRIPPER_ACTUATOR_NAME
        )
        if gripper_joint_id < 0 or self._gripper_actuator_id < 0:
            raise ValueError("Combined model is missing the Robotiq 2F-85 driver joint or actuator")
        self._gripper_qpos_address = int(self.model.jnt_qposadr[gripper_joint_id])

        camera_ids = []
        for name in self.config.render_cameras:
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if camera_id < 0:
                model_name = self.config.model_xml or "packaged Gen3 scene"
                raise ValueError(f"No MuJoCo camera named {name!r} in {model_name}")
            camera_ids.append((name, camera_id))

        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        if self.config.render_cameras:
            try:
                for name, _ in camera_ids:
                    self.renderers[name] = mujoco.Renderer(
                        self.model,
                        height=self.config.render_height,
                        width=self.config.render_width,
                    )
            except Exception as exc:
                self._close_renderers()
                raise RuntimeError(
                    "Could not initialize MuJoCo offscreen rendering. Configure a working OpenGL "
                    "backend, or omit render_cameras to run the viewer without camera observations."
                ) from exc

        if self.config.show_viewer:
            try:
                import mujoco.viewer

                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception as exc:
                self._close_renderers()
                raise RuntimeError(
                    "Could not open the MuJoCo viewer. Use --robot.show_viewer=false for headless "
                    "execution. On macOS, launch through MuJoCo's mjpython command."
                ) from exc

        self._connected = True
        self._stop_event.clear()
        self._simulation_thread = threading.Thread(
            target=self._simulation_loop,
            name="kinova-gen3-mujoco",
            daemon=True,
        )
        self._simulation_thread.start()
        logger.info("%s connected using model %s", self, self.config.model_xml or "packaged Gen3 scene")

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        self._stop_event.set()
        if self._simulation_thread is not None:
            self._simulation_thread.join(timeout=3.0)
            if self._simulation_thread.is_alive():
                logger.warning("MuJoCo simulation thread did not stop within 3 seconds")
        if self.viewer is not None:
            with contextlib.suppress(Exception):
                self.viewer.close()
            self.viewer = None
        self._close_renderers()
        self._simulation_thread = None
        self._connected = False
        logger.info("%s disconnected", self)

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        with self._locked_state():
            observation: dict[str, Any] = {
                f"joint_id_{i}.pos": float(np.rad2deg(self.data.qpos[address]))
                for i, address in enumerate(self._joint_qpos_addresses)
            }
            assert self._gripper_qpos_address is not None
            observation["gripper.pos"] = float(
                np.clip(self.data.qpos[self._gripper_qpos_address] / _GRIPPER_JOINT_RANGE, 0.0, 1.0)
            )
            for name, renderer in self.renderers.items():
                renderer.update_scene(self.data, camera=name)
                observation[name] = renderer.render().copy()
        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        with self._locked_state():
            for i in range(7):
                key = f"joint_id_{i}.pos"
                if key not in action:
                    continue
                actuator_id = self._joint_actuator_ids[i]
                target = float(np.deg2rad(float(action[key])))
                if self.model.actuator_ctrllimited[actuator_id]:
                    lower, upper = self.model.actuator_ctrlrange[actuator_id]
                    target = float(np.clip(target, lower, upper))
                self.data.ctrl[actuator_id] = target

            if "gripper.pos" in action:
                assert self._gripper_actuator_id is not None
                normalized = float(np.clip(float(action["gripper.pos"]), 0.0, 1.0))
                self.data.ctrl[self._gripper_actuator_id] = normalized * _GRIPPER_ACTUATOR_RANGE
        return action

    @contextmanager
    def _locked_state(self) -> Iterator[None]:
        viewer = self.viewer
        if viewer is not None:
            with viewer.lock():
                yield
        else:
            with self._state_lock:
                yield

    def _simulation_loop(self) -> None:
        import mujoco

        timestep = float(self.model.opt.timestep)
        viewer_period = 1.0 / self.config.viewer_sync_hz
        next_step = time.perf_counter()
        next_viewer_sync = next_step
        while not self._stop_event.is_set():
            viewer = self.viewer
            if viewer is not None and not viewer.is_running():
                self.viewer = None
                viewer = None

            with self._locked_state():
                mujoco.mj_step(self.model, self.data)
            now = time.perf_counter()
            if viewer is not None and now >= next_viewer_sync:
                with contextlib.suppress(Exception):
                    viewer.sync()
                next_viewer_sync = now + viewer_period

            next_step += timestep
            delay = next_step - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -timestep * 5:
                next_step = time.perf_counter()

    def _close_renderers(self) -> None:
        for renderer in self.renderers.values():
            with contextlib.suppress(Exception):
                renderer.close()
        self.renderers.clear()
