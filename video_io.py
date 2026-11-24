"""Decode sequential or indexed RGB frames as uint8 tensors on the requested device."""

from collections.abc import Generator, Iterable
from itertools import count
import warnings

import cv2
import torch

Frame = tuple[int, torch.Tensor]


class VideoDecodeError(RuntimeError):
    """A video or one of its requested frames could not be decoded."""


class VideoReader:
    """Read one sequential pass, or request frames by index outside that pass.

    Use a context manager when you may stop reading before EOF.
    """

    def __init__(
        self,
        video_path: str,
        device: str = "cuda:0",
    ):
        self.video_path = str(video_path)
        self.device = device
        self._closed = False
        self._decoder = None

        self._init_backend()
        self._frames = self._decode_decord() if self._decoder is not None else self._decode_cv2()

    def _init_backend(self):
        try:
            from decord import VideoReader as DecordReader, cpu

            decoder = DecordReader(self.video_path, ctx=cpu(0))
            if hasattr(decoder, "get_width"):
                self.width, self.height = decoder.get_width(), decoder.get_height()
            else:
                self.height, self.width = decoder[0].shape[:2]
            self.fps = decoder.get_avg_fps()
            if len(decoder) == 0:
                raise VideoDecodeError("视频没有可用帧")
            self.frame_count = len(decoder)
            self._decoder = decoder
            self._backend = "Decord"
            return
        except ImportError:
            pass
        except Exception as exc:
            warnings.warn(f"Decord不可用: {exc}", RuntimeWarning)

        cap = cv2.VideoCapture(self.video_path)
        try:
            if not cap.isOpened():
                raise VideoDecodeError(f"无法打开视频：{self.video_path}")
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.frame_count = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            if self.width <= 0 or self.height <= 0:
                raise VideoDecodeError(f"无法读取视频分辨率：{self.video_path}")
        finally:
            cap.release()

        self._backend = "CV2"
        warnings.warn("使用CV2解码，性能较低且随机访问慢", RuntimeWarning)

    def _decode_decord(self) -> Generator[Frame, None, None]:
        decoder = self._decoder
        for index in range(self.frame_count):
            frame = decoder[index].asnumpy()
            yield index, torch.from_numpy(frame).permute(2, 0, 1)

    def _decode_cv2(self, frame_ids: list[int] | None = None) -> Generator[Frame, None, None]:
        requested = None if frame_ids is None else set(frame_ids)
        if requested and min(requested) < 0:
            raise VideoDecodeError("视频帧序号必须为非负整数")
        cap = cv2.VideoCapture(self.video_path)
        try:
            # Direct seeking can miss a decodable final frame near the end of a GOP.
            for index in count():
                if requested is not None and not requested:
                    return
                if not cap.grab():
                    if requested:
                        raise VideoDecodeError(f"无法读取视频第 {min(requested)} 帧")
                    if index == 0:
                        raise VideoDecodeError("没有读到视频画面")
                    return
                if requested is not None:
                    if index not in requested:
                        continue
                    requested.remove(index)
                ok, frame = cap.retrieve()
                if not ok:
                    raise VideoDecodeError(f"无法读取视频第 {index} 帧")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield index, torch.from_numpy(rgb).permute(2, 0, 1)
        finally:
            cap.release()

    def __iter__(self):
        return self

    def __next__(self) -> Frame:
        try:
            index, frame = next(self._frames)
        except StopIteration:
            raise
        except Exception as exc:
            raise VideoDecodeError(f"{self.backend} 无法继续读取视频：{self.video_path}") from exc
        return index, frame.to(self.device)

    def close(self):
        self._closed = True
        self._frames.close()
        self._decoder = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def backend(self) -> str:
        return self._backend

    def get_batch(self, frame_ids: Iterable[int]) -> torch.Tensor:
        """Read the requested frames in order as a tensor of shape [N, C, H, W]."""
        if self._closed:
            raise ValueError("视频读取器已关闭")
        frame_ids = list(frame_ids)
        try:
            if self.backend == "Decord":
                decoded = self._decoder.get_batch(frame_ids).asnumpy()
                frames = torch.from_numpy(decoded).permute(0, 3, 1, 2)
            else:
                decoded = dict(self._decode_cv2(frame_ids))
                frames = torch.stack([decoded[index] for index in frame_ids])
        except Exception as exc:
            raise VideoDecodeError(f"{self.backend} 无法读取指定视频帧：{self.video_path}") from exc
        return frames.to(self.device)

    def get_frame(self, frame_id: int) -> torch.Tensor:
        return self.get_batch([frame_id])[0]
