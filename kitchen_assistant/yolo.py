"""Ultralytics YOLO runtime adapter kept outside workflow logic."""

from dataclasses import dataclass, field
from pathlib import Path

from .state import Detection, Frame


@dataclass(slots=True)
class UltralyticsYoloDetector:
    weights: Path
    device: int | str = 0
    image_size: int = 640
    confidence: float = 0.25
    roi: tuple[int, int, int, int] | None = None
    _model: object | None = field(default=None, init=False, repr=False)

    def _load(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise RuntimeError("Install the yolo optional dependencies") from error
            self._model = YOLO(str(self.weights), task="detect")
        return self._model

    def detect(self, frame: Frame) -> tuple[Detection, ...]:
        try:
            import cv2
            import numpy as np
        except ImportError as error:
            raise RuntimeError("OpenCV and NumPy are required for YOLO inference") from error
        image = cv2.imdecode(np.frombuffer(frame.image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode frame {frame.id}")
        offset_x = offset_y = 0
        if self.roi:
            x1, y1, x2, y2 = self.roi
            if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
                raise ValueError("YOLO ROI is outside the frame")
            image = image[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        result = self._load().predict(
            image,
            imgsz=self.image_size,
            device=self.device,
            conf=self.confidence,
            verbose=False,
            rect=False,
        )[0]
        names = result.names
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            name = names[class_id] if not isinstance(names, dict) else names[class_id]
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    entity=name,
                    confidence=float(box.conf[0].item()),
                    bbox=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                )
            )
        return tuple(detections)

