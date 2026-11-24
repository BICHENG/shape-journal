"""Measurement pose presets and SMPL pose helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeasurementPosePreset:
    id: str
    vertex_key: str
    arm_down_degrees: float


def _apose_preset(degrees: float) -> MeasurementPosePreset:
    stem = f"apose_arms_down_{degrees:g}deg"
    return MeasurementPosePreset(f"{stem}_v1", f"{stem}_vertices", degrees)


TPOSE_PRESET = MeasurementPosePreset("tpose_zero_v1", "tpose_vertices", 0.0)
DEFAULT_APOSE_PRESET = _apose_preset(35.0)
APOSE_EXPORT_PRESETS = (_apose_preset(15.0), DEFAULT_APOSE_PRESET, _apose_preset(55.0), _apose_preset(85.0))


def packed_apose_presets(mesh: Mapping) -> list[MeasurementPosePreset]:
    return [
        MeasurementPosePreset(str(alias), str(key), float(angle))
        for alias, key, angle in zip(
            mesh.get("apose_preset_ids", ()), mesh.get("apose_vertex_keys", ()),
            np.asarray(mesh.get("apose_arm_down_degrees", ())).reshape(-1),
        )
        if key != "apose_vertices" and key in mesh
    ]


def selected_apose(degrees: float) -> MeasurementPosePreset:
    return MeasurementPosePreset(f"apose_selected_{degrees:g}deg_v1", "apose_vertices", degrees)


def packed_pose_presets(mesh: Mapping, selected_angle: float) -> tuple[MeasurementPosePreset, ...]:
    return TPOSE_PRESET, selected_apose(selected_angle), *packed_apose_presets(mesh)


# Lower-body states stay independent from the real upper-body angle in pose_id.
LOWER_BODY_POSE_PRESETS = {
    "natural_standing": "natural",
    "legs_together": "foot_surface_gap_2_5mm",
    "legs_shoulder_width": "smpl_shoulder_joint_spacing_unconfirmed",
}

LOWER_BODY_ROTATED_JOINTS = (1, 2, 7, 8)


def make_apose_rotvecs(arm_down_degrees: float = DEFAULT_APOSE_PRESET.arm_down_degrees) -> np.ndarray:
    """Return SMPL rotvecs for arms lowered from T-pose by the requested angle."""
    pose = np.zeros((24, 3), dtype=np.float32)
    angle = np.deg2rad(arm_down_degrees)
    pose[16, 2] = -angle
    pose[17, 2] = angle
    return pose


def make_lower_body_pose_rotvecs(
    upper_body_pose: np.ndarray,
    lower_body_pose_id: str,
    hip_adduction_radians: float = 0.0,
) -> np.ndarray:
    """Apply one mirrored hip angle while counter-rotating the ankles to keep feet level."""
    if lower_body_pose_id not in LOWER_BODY_POSE_PRESETS:
        raise ValueError(f"Unsupported lower_body_pose_id: {lower_body_pose_id}")
    pose = np.asarray(upper_body_pose, dtype=np.float32).copy()
    if pose.shape != (24, 3) or not np.all(np.isfinite(pose)):
        raise ValueError("upper_body_pose must be a finite (24, 3) rotvec array")
    if np.any(pose[[1, 2, 4, 5, 7, 8, 10, 11]]):
        raise ValueError("upper_body_pose already contains lower-body rotations")
    angle = float(hip_adduction_radians)
    if not np.isfinite(angle):
        raise ValueError("hip_adduction_radians must be finite")
    if lower_body_pose_id == "natural_standing":
        if angle != 0.0:
            raise ValueError("natural_standing requires zero hip adduction")
        return pose
    pose[1, 2], pose[2, 2] = -angle, angle
    pose[7, 2], pose[8, 2] = angle, -angle
    return pose


def lower_body_vertex_key(vertex_key: str, lower_body_pose_id: str) -> str:
    if lower_body_pose_id not in LOWER_BODY_POSE_PRESETS:
        raise ValueError(f"Unsupported lower_body_pose_id: {lower_body_pose_id}")
    if lower_body_pose_id == "natural_standing":
        return vertex_key
    return f"{pose_key_stem(vertex_key)}_{lower_body_pose_id}_vertices"


def measurement_pose_id(upper_body_pose_id: str, lower_body_pose_id: str) -> str:
    if lower_body_pose_id not in LOWER_BODY_POSE_PRESETS:
        raise ValueError(f"Unsupported lower_body_pose_id: {lower_body_pose_id}")
    return f"{upper_body_pose_id}__{lower_body_pose_id}"


def pose_key_stem(vertex_key: str) -> str:
    """Return the shared NPZ key stem for one pose's arrays."""
    suffix = "_vertices"
    if not vertex_key.endswith(suffix):
        raise ValueError(f"Pose vertex key must end with {suffix!r}: {vertex_key!r}")
    return vertex_key[: -len(suffix)]
