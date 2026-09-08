"""Camera, web-stream, and deterministic video frame readers."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Condition, Thread
from time import monotonic, sleep
from typing import Protocol

from .state import Frame


class FrameReader(Protocol):
    live: bool

    def read(self) -> Frame | None: ...

    def close(self) -> None: ...


class _OpenCvReader:
    live = True

    def __init__(self, source: int | str, *, label: str, jpeg_quality: int = 90) -> None:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is required for camera, web, and video input") from error
        self._cv2 = cv2
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Cannot open {label}: {source}")
        self._label = label
        self._jpeg_quality = jpeg_quality
        self._index = 0

    def _next_image(self) -> tuple[bytes, int] | None:
        ok, image = self._capture.read()
        if not ok:
            return None
        self._index += 1
        ok, encoded = self._cv2.imencode(
            ".jpg",
            image,
            [self._cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        if not ok:
            raise RuntimeError("Could not encode input frame")
        return encoded.tobytes(), self._index

    def read(self) -> Frame | None:
        result = self._next_image()
        if result is None:
            return None
        image, index = result
        return Frame(f"{self._label}:{index}", datetime.now(timezone.utc), image)

    def close(self) -> None:
        self._capture.release()


class CameraReader(_OpenCvReader):
    def __init__(self, camera: int = 0, *, jpeg_quality: int = 90) -> None:
        super().__init__(camera, label=f"camera-{camera}", jpeg_quality=jpeg_quality)


class WebStreamReader(_OpenCvReader):
    def __init__(self, url: str, *, jpeg_quality: int = 90) -> None:
        if not url:
            raise ValueError("Web stream URL cannot be empty")
        super().__init__(url, label="web-stream", jpeg_quality=jpeg_quality)


class VideoReader(_OpenCvReader):
    live = False

    def __init__(self, path: Path, *, fps: float = 1.0, jpeg_quality: int = 90) -> None:
        if fps <= 0:
            raise ValueError("Video analysis FPS must be positive")
        super().__init__(str(path), label=path.name, jpeg_quality=jpeg_quality)
        source_fps = self._capture.get(self._cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            self.close()
            raise RuntimeError(f"Video has no valid FPS: {path}")
        self._source_fps = source_fps
        self._sample_period = 1 / fps
        self._next_seconds = 0.0
        self._timeline_start = datetime.now(timezone.utc)

    def read(self) -> Frame | None:
        while True:
            result = self._next_image()
            if result is None:
                return None
            image, index = result
            source_seconds = (index - 1) / self._source_fps
            if source_seconds + 1e-9 < self._next_seconds:
                continue
            self._next_seconds += self._sample_period
            return Frame(
                f"{self._label}:{index}",
                self._timeline_start + timedelta(seconds=source_seconds),
                image,
                source_seconds,
            )


@dataclass(slots=True)
class FrameScheduler:
    """Keep capture live while allowing only one in-flight inference."""

    reader: FrameReader
    analysis_fps: float = 1.0
    _condition: Condition = field(default_factory=Condition, init=False)
    _latest: Frame | None = field(default=None, init=False)
    _finished: bool = field(default=False, init=False)
    _error: BaseException | None = field(default=None, init=False)
    _thread: Thread | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.analysis_fps <= 0:
            raise ValueError("Analysis FPS must be positive")

    def __iter__(self):
        if not self.reader.live:
            while frame := self.reader.read():
                yield frame
            return
        self._thread = Thread(target=self._capture_loop, name="frame-capture", daemon=True)
        self._thread.start()
        last_id = None
        deadline = monotonic()
        while True:
            remaining = deadline - monotonic()
            if remaining > 0:
                sleep(remaining)
            with self._condition:
                if self._latest is None and not self._finished:
                    self._condition.wait(timeout=1 / self.analysis_fps)
                if self._error:
                    raise RuntimeError("Frame capture failed") from self._error
                frame = self._latest
                finished = self._finished
            if frame is not None and frame.id != last_id:
                last_id = frame.id
                yield frame
            if finished:
                break
            deadline = monotonic() + 1 / self.analysis_fps

    def _capture_loop(self) -> None:
        try:
            while frame := self.reader.read():
                with self._condition:
                    self._latest = frame
                    self._condition.notify_all()
        except BaseException as error:
            self._error = error
        finally:
            with self._condition:
                self._finished = True
                self._condition.notify_all()

    def close(self) -> None:
        self.reader.close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

