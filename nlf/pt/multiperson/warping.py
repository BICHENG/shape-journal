from typing import Tuple, Optional

import torch
import torch.nn.functional as F

from nlf.pt import ptu3d


def warp_images_with_pyramid(
    images: torch.Tensor,
    intrinsic_matrix: torch.Tensor,
    new_invprojmats: torch.Tensor,
    crop_scales: torch.Tensor,
    output_shape: Tuple[int, int],
    image_ids: torch.Tensor,
    n_pyramid_levels: int = 3,
):
    # Create a very simple pyramid with lower resolution images for simple antialiasing.
    image_levels = [images]
    for _ in range(1, n_pyramid_levels):
        # We use simple averaging (box filter) to create the pyramid, for efficiency
        image_levels.append(F.avg_pool2d(image_levels[-1], 2, 2))

    intrinsic_matrix_levels = [
        corner_aligned_scale_mat(1 / 2**i_level, device=intrinsic_matrix.device) @ intrinsic_matrix
        for i_level in range(n_pyramid_levels)
    ]

    # Decide which pyramid level is most appropriate for each crop
    i_pyramid_levels = torch.floor(-torch.log2(crop_scales))
    i_pyramid_levels = torch.clip(i_pyramid_levels, 0, n_pyramid_levels - 1).int()

    return torch.stack(
        [
            warp_single_image(
                image_levels[i_pyramid_levels[i]][image_ids[i]],
                intrinsic_matrix_levels[i_pyramid_levels[i]][i],
                new_invprojmats[i],
                output_shape,
            )
            for i in range(len(image_ids))
        ]
    )


def warp_single_image(
    image: torch.Tensor,
    intrinsic_matrix: torch.Tensor,
    new_invprojmat: torch.Tensor,
    output_shape: Tuple[int, int],
):
    device = image.device
    new_coords = torch.stack(
        torch.meshgrid(
            torch.arange(output_shape[1], device=device),
            torch.arange(output_shape[0], device=device),
            indexing='xy',
        ),
        dim=-1,
    ).float()
    new_coords_homog = ptu3d.to_homogeneous(new_coords)
    old_coords_homog = torch.einsum('hwc,Cc->hwC', new_coords_homog, new_invprojmat)
    old_coords_homog = ptu3d.to_homogeneous(ptu3d.project(old_coords_homog))
    old_coords = torch.einsum('hwc,Cc->hwC', old_coords_homog, intrinsic_matrix)[..., :2]
    size = torch.tensor([image.shape[2], image.shape[1]], dtype=old_coords.dtype, device=device)
    old_coords_normalized = old_coords.mul_(2.0 / (size - 1)).sub_(1.0)
    return F.grid_sample(
        image.unsqueeze(0),
        old_coords_normalized.unsqueeze(0).to(image.dtype),
        align_corners=True,
        mode='bilinear',
        padding_mode='zeros',
    ).squeeze(0)


def corner_aligned_scale_mat(factor: float, device: Optional[torch.device] = None) -> torch.Tensor:
    shift = (factor - 1.0) / 2.0
    return torch.tensor(
        [[factor, 0.0, shift], [0.0, factor, shift], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
