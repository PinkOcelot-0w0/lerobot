import os
import time
from typing import Any

# Configure FFmpeg before OpenCV opens its RTSP backend. TCP favors a stable
# stream; the small buffer and low-delay flags reduce latency where supported.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|buffer_size;102400|fflags;nobuffer|flags;low_delay",
)

from lerobot.utils.errors import DeviceAlreadyConnectedError

from ..opencv.camera_opencv import OpenCVCamera
from .configuration_kinova_gen3_vision import KinovaGen3VisionCameraConfig


class KinovaGen3VisionCamera(OpenCVCamera):
    """Kinova Gen3 integrated color camera, configured through Kortex and read over RTSP."""

    def __init__(self, config: KinovaGen3VisionCameraConfig):
        super().__init__(config)
        self.config: KinovaGen3VisionCameraConfig = config

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        # The integrated camera is discovered through the robot's Kortex API,
        # not as a local host camera.
        return []

    def connect(self, vision_config: Any, device_manager: Any, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected")

        # Kortex API 2.8 generated protobuf files require its legacy runtime;
        # the robot connection sets this compatibility mode before reaching here.
        from kortex_api.autogen.messages import DeviceConfig_pb2, VisionConfig_pb2

        devices = device_manager.ReadAllDevices().device_handle
        vision_devices = [device for device in devices if device.device_type == DeviceConfig_pb2.VISION]
        if len(vision_devices) != 1:
            raise RuntimeError(f"Expected one Kinova vision module, found {len(vision_devices)}")
        vision_device_id = vision_devices[0].device_identifier

        resolution_map = {
            (640, 480): VisionConfig_pb2.RESOLUTION_640x480,
            (320, 240): VisionConfig_pb2.RESOLUTION_320x240,
            (424, 240): VisionConfig_pb2.RESOLUTION_424x240,
            (480, 270): VisionConfig_pb2.RESOLUTION_480x270,
            (1280, 720): VisionConfig_pb2.RESOLUTION_1280x720,
            (1920, 1080): VisionConfig_pb2.RESOLUTION_1920x1080,
        }
        framerate_map = {
            6: VisionConfig_pb2.FRAMERATE_6_FPS,
            15: VisionConfig_pb2.FRAMERATE_15_FPS,
            30: VisionConfig_pb2.FRAMERATE_30_FPS,
        }
        try:
            resolution = resolution_map[(self.config.width, self.config.height)]
        except KeyError:
            raise ValueError(
                "Kinova vision supports 320x240, 424x240, 480x270, 640x480, "
                "1280x720, or 1920x1080"
            ) from None
        try:
            frame_rate = framerate_map[self.config.fps]
        except KeyError:
            raise ValueError("Kinova vision supports 6, 15, or 30 FPS") from None

        settings = VisionConfig_pb2.SensorSettings(
            sensor=VisionConfig_pb2.SENSOR_COLOR,
            resolution=resolution,
            frame_rate=frame_rate,
            bit_rate=VisionConfig_pb2.BITRATE_10_MBPS,
        )
        vision_config.SetSensorSettings(settings, vision_device_id)
        focus = VisionConfig_pb2.SensorFocusAction(
            sensor=VisionConfig_pb2.SENSOR_COLOR,
            focus_action=VisionConfig_pb2.FOCUSACTION_PAUSE_CONTINUOUS_FOCUS,
        )
        vision_config.DoSensorFocusAction(focus, vision_device_id)
        time.sleep(0.5)

        # RTSP sources frequently report unreliable FPS and reject VideoCapture
        # property setters. Trust the profile set through Kortex and validate
        # actual frame dimensions in OpenCVCamera's image postprocessing.
        self.capture_width = self.config.width
        self.capture_height = self.config.height
        self.fps = self.config.fps
        super().connect(warmup=warmup)

    def _configure_capture_settings(self) -> None:
        if self.videocapture is None or not self.videocapture.isOpened():
            raise ConnectionError(f"Could not open Kinova RTSP stream {self.index_or_path}")
        self.width = self.config.width
        self.height = self.config.height
        self.capture_width = self.config.width
        self.capture_height = self.config.height
        self.fps = self.config.fps

    def _validate_width_and_height(self) -> None:
        # Width/height are selected on the camera via Kortex before RTSP opens.
        return

    def _validate_fps(self) -> None:
        # The RTSP stream's reported FPS is not reliable on some Gen3 firmware.
        return

