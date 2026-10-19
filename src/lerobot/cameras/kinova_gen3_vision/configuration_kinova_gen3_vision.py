from dataclasses import dataclass
from pathlib import Path

from ..configs import CameraConfig, ColorMode, Cv2Backends, Cv2Rotation


@CameraConfig.register_subclass("kinova_gen3_vision")
@dataclass
class KinovaGen3VisionCameraConfig(CameraConfig):
    index_or_path: str | Path = "rtsp://192.168.1.10/color"
    fps: int = 30
    width: int = 640
    height: int = 480
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 2
    fourcc: str | None = None
    backend: Cv2Backends = Cv2Backends.ANY
