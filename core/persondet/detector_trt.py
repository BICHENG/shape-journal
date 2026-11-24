"""YOLO26 detection and its fixed-shape TensorRT cache."""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from config import TRT_CACHE_ROOT, YOLO_MODEL_DIR
from core.motionpack.io import _file_sha256
from core.trt_cache import cached_trt

DEFAULT_THRESHOLD = 0.3


def load_yolo(checkpoint: Path, example: torch.Tensor):
    from ultralytics import YOLO

    model = YOLO(str(checkpoint.resolve(strict=True))).model.eval().to(example.device)
    model.model[-1].export = True
    model.model[-1].format = "onnx"
    # Cache anchors for the fixed input shape before capturing the graph.
    with torch.no_grad():
        model(example)
    return model


def detector_pixels(images: torch.Tensor, input_size: Tuple[int, int]) -> torch.Tensor:
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    if images.shape[2:4] != input_size:
        images = torch.nn.functional.interpolate(images, size=input_size, mode="bilinear", align_corners=False)
    return images


def detect_persons(infer, images: torch.Tensor, input_size: Tuple[int, int], threshold: float) -> np.ndarray:
    """Map YOLO26 end-to-end detections from the network image to the original frame."""
    h, w = images.shape[2:4]
    target_h, target_w = input_size

    with torch.inference_mode():
        boxes = infer(detector_pixels(images, input_size))[0].cpu().numpy()
    boxes = boxes[(boxes[:, 4] >= threshold) & (boxes[:, 5] == 0)]
    if not len(boxes):
        return np.empty((0, 6))
    if (h, w) != input_size:
        boxes[:, [0, 2]] *= w / target_w
        boxes[:, [1, 3]] *= h / target_h
    return boxes


class YOLO26TRTDetector:
    def __init__(
        self,
        model_name: str = "yolo26m.pt",
        device: str = 'cuda',
        input_size: Tuple[int, int] = (640, 640),
        use_fp16: bool = True,
    ):
        self.input_size = input_size
        h, w = input_size
        checkpoint = YOLO_MODEL_DIR / model_name
        if not checkpoint.is_file():
            checkpoint = Path(model_name)
        fingerprint = _file_sha256(checkpoint)[:12]
        precision = torch.float16 if use_fp16 else torch.float32
        tag = "fp16" if use_fp16 else "fp32"
        path = TRT_CACHE_ROOT / f"{checkpoint.stem}_{h}x{w}_{tag}_{fingerprint}.pt2"
        example = torch.zeros(1, 3, h, w, device=device)
        self._trt_model = cached_trt(path, lambda: load_yolo(checkpoint, example), example, precision=precision)

    def detect(self, images: torch.Tensor, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
        return detect_persons(self._trt_model, images, self.input_size, threshold)
