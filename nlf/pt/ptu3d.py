import torch


def reconstruct_absolute(
    coords2d: torch.Tensor,
    coords3d_rel: torch.Tensor,
    intrinsics: torch.Tensor,
    proc_side: int,
    stride: int,
    centered_stride: bool,
    point_validity_mask: torch.Tensor,
):
    inv_intrinsics = torch.linalg.inv(intrinsics.to(coords2d.dtype))
    normalized = (to_homogeneous(coords2d) @ inv_intrinsics.transpose(1, 2))[..., :2]
    inside_fit_area = is_within_fov(coords2d, proc_side, stride, centered_stride, 1.0)
    reference = reconstruct_ref_fullpersp(normalized, coords3d_rel, inside_fit_area & point_validity_mask)

    from_3d = coords3d_rel + reference.unsqueeze(1)
    depth = coords3d_rel[..., 2] + reference[:, 2:3]
    from_2d = to_homogeneous(normalized) * depth.unsqueeze(-1)
    inside_output_area = is_within_fov(project_pose(from_3d, intrinsics), proc_side, stride, centered_stride, 0.6)
    return torch.where((inside_output_area & (from_3d[..., 2] > 0.001))[..., None],
                       0.5 * from_3d + 0.5 * from_2d, from_3d)


def to_homogeneous(x):
    return torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)


def reconstruct_ref_fullpersp(normalized_2d, coords3d_rel, validity_mask):
    """Reconstructs the reference point location.

    Args:
      normalized_2d: normalized image coordinates of the joints
         (without intrinsics applied), shape [batch_size, n_points, 2]
      coords3d_rel: 3D camera coordinate offsets relative to the unknown reference
         point which we want to reconstruct, shape [batch_size, n_points, 3]
      validity_mask: boolean mask of shape [batch_size, n_points] containing True
         where the point is reliable and should be used in the reconstruction

    Returns:
      The 3D reference point in camera coordinates, shape [batch_size, 3]
    """

    n_batch = normalized_2d.shape[0]
    n_points = normalized_2d.shape[1]

    eyes2 = (
        torch.eye(2, device=normalized_2d.device, dtype=normalized_2d.dtype)
        .unsqueeze(0)
        .repeat(n_batch, n_points, 1)
    )
    scale2d, reshaped2d = rms_normalize_and_reshape(normalized_2d, validity_mask, n_points)
    A = torch.cat([eyes2, -reshaped2d], dim=2)

    rel_backproj = normalized_2d * coords3d_rel[:, :, 2:] - coords3d_rel[:, :, :2]
    scale_rel_backproj, b = rms_normalize_and_reshape(rel_backproj, validity_mask, n_points)

    weights = validity_mask.to(normalized_2d.dtype) + 1e-8
    weights = torch.repeat_interleave(weights, 2, 1)
    weighted_transpose = (weights.unsqueeze(-1) * A).mT
    gramian = weighted_transpose @ A
    gramian.diagonal(dim1=-2, dim2=-1).add_(1e-4)
    chol, _ = torch.linalg.cholesky_ex(gramian)
    ref = torch.cholesky_solve(weighted_transpose @ b, chol)

    ref = torch.cat(
        [ref[:, :2] * scale_rel_backproj, ref[:, 2:] * (scale_rel_backproj / scale2d)], dim=1
    )
    return torch.squeeze(ref, dim=-1)


def rms_normalize_and_reshape(x: torch.Tensor, validity_mask: torch.Tensor, n_points: int):
    valid = validity_mask[..., None]
    squared = torch.square(x)
    sum_valid = torch.where(valid, squared, torch.zeros_like(squared)).sum(dim=[1, 2], keepdim=True)
    count = valid.sum(dim=[1, 2], keepdim=True, dtype=x.dtype)
    scale = torch.sqrt(torch.nan_to_num(sum_valid / count) + 1e-10)
    return scale, (x / scale).reshape(-1, n_points * 2, 1)


def is_within_fov(
    imcoords: torch.Tensor,
    proc_side: int,
    stride: int,
    centered_stride: bool,
    border_factor: float = 0.75,
):
    offset = -stride / 2.0 if not centered_stride else 0.0

    lower = stride * border_factor + offset
    upper = proc_side - stride * border_factor + offset
    proj_in_fov = torch.all(torch.logical_and(imcoords >= lower, imcoords <= upper), dim=-1)
    return proj_in_fov


def project_pose(coords3d, intrinsic_matrix):
    projected = coords3d / torch.maximum(
        torch.tensor(0.1, device=coords3d.device), coords3d[..., 2:]
    )
    return torch.einsum('bnk,bjk->bnj', projected, intrinsic_matrix[..., :2, :])


def lookat_matrix(forward_vector, up_vector=None):
    if up_vector is None:
        up_vector = forward_vector.new_tensor([0.0, -1.0, 0.0]).expand(forward_vector.shape[0], -1)
    # Z will point forwards, towards the box center
    new_z = forward_vector / torch.linalg.norm(forward_vector, dim=-1, keepdim=True)
    # Get the X (right direction) as the cross of forward and up.
    new_x = torch.linalg.cross(new_z, up_vector)
    # When forward is vertical, world X is already perpendicular to it.
    vertical = (new_z[:, 0] == 0) & (new_z[:, 2] == 0)
    fallback_x = torch.where(vertical, 1.0, new_z[:, 2])
    new_x_alt = torch.stack([fallback_x, torch.zeros_like(fallback_x), -new_z[:, 0]], dim=1)
    new_x = torch.where(torch.linalg.norm(new_x, dim=-1, keepdim=True) == 0, new_x_alt, new_x)
    new_x = new_x / torch.linalg.norm(new_x, dim=-1, keepdim=True)
    # Complete the right-handed coordinate system to get Y
    new_y = torch.linalg.cross(new_z, new_x)
    # Stack the axis vectors to get the rotation matrix
    return torch.stack([new_x, new_y, new_z], dim=1)


def project(points):
    return points[..., :2] / points[..., 2:3]
