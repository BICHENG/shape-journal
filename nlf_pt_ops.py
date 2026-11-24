"""Camera crops and recovery shared by the PyTorch and ONNX NLF models."""

from dataclasses import dataclass
import numpy as np
import torch

from nlf.pt import ptu3d
from nlf.pt.multiperson.warping import warp_images_with_pyramid


@dataclass(frozen=True)
class NLFConfig:
    proc_side: int = 384
    stride_test: int = 32
    depth: int = 8
    backbone_link_dim: int = 512
    box_size_m: float = 2.2
    uncert_bias: float = 0.0
    uncert_bias2: float = 0.001
    centered_stride: bool = True


NLF = NLFConfig()


def compute_intrinsics(h: int, w: int, fov_degrees: float = 55.0, device: str = "cuda") -> torch.Tensor:
    """Use a centered pinhole camera with the field of view on the long image axis."""
    focal = max(h, w) / (2 * np.tan(np.deg2rad(fov_degrees) / 2))
    return torch.tensor([
        [focal, 0, w / 2],
        [0, focal, h / 2],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)


def crop_cameras(boxes: torch.Tensor, intrinsic: torch.Tensor):
    """Return the camera and scale used to reproject each xyxy box."""
    side = NLF.proc_side
    x, y, x2, y2 = boxes.unbind(dim=-1)
    width, height = x2 - x, y2 - y
    cx, cy = x + width / 2, y + height / 2
    pixels = torch.stack([
        torch.stack(point, dim=-1)
        for point in ((cx, cy), (cx, y), (x + width, cy), (cx, y + height), (x, cy))
    ], dim=1)
    rays = torch.einsum("nij,npj->npi", torch.linalg.inv(intrinsic), ptu3d.to_homogeneous(pixels))
    rays = ptu3d.to_homogeneous(rays[..., :2])
    up = rays.new_tensor([0.0, -1.0, 0.0]).expand(len(boxes), -1)
    rotation = ptu3d.lookat_matrix(rays[:, 0], up)
    edges = ptu3d.project(torch.einsum("nij,npj->npi", intrinsic @ rotation, rays[:, 1:]))
    vertical = torch.linalg.norm(edges[:, 0] - edges[:, 2], dim=-1)
    horizontal = torch.linalg.norm(edges[:, 1] - edges[:, 3], dim=-1)
    scale = side / torch.maximum(vertical, horizontal)

    crop_intrinsic = torch.zeros(len(boxes), 3, 3, device=boxes.device)
    crop_intrinsic[:, 0, 0] = scale * intrinsic[:, 0, 0]
    crop_intrinsic[:, 1, 1] = scale * intrinsic[:, 1, 1]
    crop_intrinsic[:, :2, 2] = side / 2
    crop_intrinsic[:, 2, 2] = 1.0
    return crop_intrinsic, rotation, scale


def generate_crops(
    images: torch.Tensor,
    boxes_by_frame: list[torch.Tensor],
    intrinsics: torch.Tensor,
    *,
    n_pyramid_levels: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crop decoded RGB (0–255) around xyxy boxes and return the crop cameras."""
    device, side = images.device, NLF.proc_side
    nonempty = [(index, boxes) for index, boxes in enumerate(boxes_by_frame) if len(boxes)]
    if not nonempty:
        cameras = torch.empty(0, 3, 3, device=device)
        return torch.empty(0, 3, side, side, device=device), cameras, cameras.clone()
    boxes = torch.cat([boxes for _, boxes in nonempty])
    image_ids = torch.tensor([index for index, boxes in nonempty for _ in range(len(boxes))], device=device)
    intrinsic = intrinsics[image_ids]
    crop_intrinsic, rotation, scale = crop_cameras(boxes, intrinsic)
    linear = (images.to(torch.float16) / 255.0).pow_(2.2)
    if device.type == "cpu":
        linear = linear.float()
    # Inverting the input makes zero padding white in the finished crop.
    crops = warp_images_with_pyramid(
        1 - linear, intrinsic, torch.linalg.inv(crop_intrinsic @ rotation),
        scale, (side, side), image_ids, n_pyramid_levels=n_pyramid_levels,
    )
    return (1 - crops).clamp(0, 1) ** (1.79 / 2.2), crop_intrinsic, rotation


def restore_camera_coordinates(coords2d, coords3d, uncertainties, intrinsics, rotation):
    """Resolve crop predictions to the original camera frame in millimetres."""
    absolute = ptu3d.reconstruct_absolute(
        coords2d, coords3d, intrinsics,
        proc_side=NLF.proc_side, stride=NLF.stride_test, centered_stride=NLF.centered_stride,
        point_validity_mask=uncertainties < 0.3,
    ) * 1000
    return {
        "poses3d": torch.einsum("nij,npj->npi", rotation.transpose(-1, -2), absolute),
        "uncertainties": uncertainties * 3,
    }
