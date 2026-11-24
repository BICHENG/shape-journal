"""NLF neural inference with the production split weights."""

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import torch

import config as cfg
from config import NLF_METADATA_SNAPSHOT_PATH, NLF_WEIGHT_FILES as MAIN_RUNTIME_WEIGHT_FILES
from core.motionpack.io import _file_sha256
from core.trt_cache import cached_trt
from nlf_pt_ops import NLF
from nlf_pt_model import EfficientNetV2L, GPSField, LocalizerHead

def _json_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _weight_records(directory: Path) -> list[dict]:
    return [
        {"path": f"weights/main/{name}", "sha256": _file_sha256(directory / name)}
        for name in MAIN_RUNTIME_WEIGHT_FILES
    ]


def _canonical_record(path: Path, array: np.ndarray) -> dict:
    return {
        "path": "output/canonical_smpl.npy",
        "file_sha256": _file_sha256(path),
        "array_sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    }


def runtime_artifacts_metadata(weights_dir: Path, smpl_canonical_path: Path) -> dict:
    artifacts = {
        "weights": _weight_records(Path(weights_dir)),
        "smpl_canonical": _canonical_record(smpl_canonical_path, np.load(smpl_canonical_path, allow_pickle=False)),
    }
    return {**artifacts, "digest": _json_sha256(artifacts)}


def _validated_runtime_artifacts(snapshot: dict) -> dict:
    try:
        runtime = snapshot["runtime_artifacts"]
        content = {
            "weights": runtime["weights"],
            "smpl_canonical": runtime["smpl_canonical"],
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("NLF runtime artifact metadata is incomplete") from exc
    if set(runtime) != {"weights", "smpl_canonical", "digest"}:
        raise ValueError("NLF runtime artifact metadata has unexpected fields")
    if runtime["digest"] != _json_sha256(content):
        raise ValueError("NLF runtime artifact metadata digest mismatch")
    return runtime


def verify_main_runtime_weights(weights_dir: Path, snapshot: dict = None) -> tuple:
    if snapshot is None:
        snapshot = json.loads(NLF_METADATA_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    runtime = _validated_runtime_artifacts(snapshot)
    if _weight_records(Path(weights_dir)) != runtime["weights"]:
        raise ValueError("NLF main runtime weights do not match the production artifacts")
    return snapshot, runtime["digest"]


def verify_smpl_runtime_artifact(path: Path, canonical: np.ndarray, snapshot: dict) -> tuple:
    runtime = _validated_runtime_artifacts(snapshot)
    expected = runtime["smpl_canonical"]
    query = snapshot.get("smpl_query_1048", {})
    if tuple(canonical.shape) != (query.get("point_count"), 3):
        raise ValueError("NLF SMPL canonical shape does not match the production 1048 layout")
    if canonical.dtype.name != query.get("dtype"):
        raise ValueError("NLF SMPL canonical dtype does not match the production layout")
    actual = _canonical_record(path, canonical)
    if actual != expected or actual["array_sha256"] != query.get("canonical_digest"):
        raise ValueError("NLF SMPL canonical does not match the production artifact")
    try:
        return (
            snapshot["producer_model"]["sha256"],
            query["layout_id"],
            runtime["digest"],
        )
    except KeyError as exc:
        raise ValueError("NLF production identity metadata is incomplete") from exc


class NLFSMPLInference(torch.nn.Module):
    """Predict 1048 body points with the split weights."""

    def __init__(self, device="cuda", *, use_trt=True, trt_batch_size=4, smpl_data_dir=None):
        super().__init__()
        self.device = device
        self.use_fp16 = torch.device(device).type == "cuda"
        self.trt_batch_size = trt_batch_size
        weights = cfg.NLF_WEIGHTS_ROOT
        canonical_path = Path(smpl_data_dir) / cfg.NLF_CANONICAL_PATH.name if smpl_data_dir else cfg.NLF_CANONICAL_PATH
        snapshot, _ = verify_main_runtime_weights(weights)
        canonical = np.load(canonical_path, allow_pickle=False)
        self.producer_model_sha256, self.query_layout_id, self.runtime_artifacts_digest = verify_smpl_runtime_artifact(
            canonical_path, canonical, snapshot,
        )

        self.backbone = EfficientNetV2L().to(device).eval()
        self.head = LocalizerHead(GPSField(), NLF).to(device).eval()
        if self.use_fp16:
            self.head.layer.half()
        self._load_weights(weights)
        with torch.no_grad():
            filters, bias = self.head.get_weights_for_canonical_points(torch.from_numpy(canonical).to(device))
        self.register_buffer("query_filters", filters)
        self.register_buffer("query_bias", bias)

        self._trt_compiled = False
        if self.use_fp16:
            self.backbone.half()
            if use_trt:
                self._compile_backbone()

    def _load_weights(self, root: Path) -> None:
        def read(name):
            return torch.load(root / name, map_location=self.device, weights_only=True)

        backbone = read("backbone.pth") | read("backbone_buffers.pth")
        self.backbone.load_state_dict({f"features.{name}": tensor for name, tensor in backbone.items()})
        buffers = read("buffers.pth")
        field = "heatmap_head.weight_field."
        self.head.weight_field.load_state_dict(read("field.pth") | {
            name.removeprefix(field): tensor for name, tensor in buffers.items() if name.startswith(field)
        })
        layer = "heatmap_head.layer."
        self.head.layer.load_state_dict(
            {name.removeprefix("layer."): tensor for name, tensor in read("head.pth").items()}
            | {name.removeprefix(layer): tensor for name, tensor in buffers.items() if name.startswith(layer)}
        )

    def _compile_backbone(self) -> None:
        side = NLF.proc_side
        path = cfg.TRT_CACHE_ROOT / (
            f"nlf_backbone_b{self.trt_batch_size}_{self.runtime_artifacts_digest[:12]}.pt2"
        )
        example = torch.zeros(self.trt_batch_size, 3, side, side, device=self.device, dtype=torch.float16)
        try:
            self.backbone = cached_trt(path, lambda: self.backbone, example, precision=torch.float16)
            self._trt_compiled = True
        except Exception as error:
            warnings.warn(f"NLF TensorRT backbone unavailable. Using the PyTorch weights: {error}")

    def forward(self, crops: torch.Tensor) -> dict[str, torch.Tensor]:
        images = crops * 2 - 1
        if self.use_fp16:
            images = images.half()
        if self._trt_compiled and len(images) != self.trt_batch_size:
            batches = []
            for batch in images.split(self.trt_batch_size):
                count = len(batch)
                if count < self.trt_batch_size:
                    padding = batch.new_zeros((self.trt_batch_size - count, *batch.shape[1:]))
                    batch = torch.cat((batch, padding))
                batches.append(self.backbone(batch)[:count])
            features = torch.cat(batches)
        else:
            features = self.backbone(images)
        coords2d, coords3d, uncertainty = self.head.apply_weights(
            self.head.layer(features).float(), self.query_filters, self.query_bias,
        )
        return {"coords2d": coords2d.float(), "coords3d": coords3d.float(), "uncertainties": uncertainty.float()}
