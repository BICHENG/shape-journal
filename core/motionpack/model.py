"""Pure MotionPack data structures."""

from dataclasses import dataclass

import numpy as np


PRODUCTION_NLF_MODEL_SHA256 = "52bee28edb6ea9148691331df87cfc238d7e3d9134dc60104a5aaed282a9ddad"
PRODUCTION_NLF_QUERY_LAYOUT_ID = "98f279567691eeb9f1c739ae0bd607a92bb1488e8daad31fd8b272d4ec2613e9"
SMPL_VERTEX_SUBSET_SHA256 = "bf0f3fc9150d13049d841fc1a5a107481330fb043d7841c83d6cf70d505edcce"
POSE_METADATA = {
    "points3d_units": "mm",
    "points3d_coordinate_system": "camera_cv_rh_x_right_y_down_z_forward",
    "points2d_coordinate_system": "nlf_crop_pixels",
    "uncertainty_units": "mm",
    "uncertainty_semantics": "nlf_fit_target",
}
BODY_METADATA = {
    "units": "mm",
    "coordinate_system": "camera_cv_rh_x_right_y_down_z_forward",
    "source": "smpl_forward",
    "pose_representation": "axis_angle",
    "pose_units": "rad",
    "pose_shape": [24, 3],
}
BODY_MODEL_ASSET_IDENTITY = {
    "schema_version": 1,
    "model": "smpl_neutral",
    "model_sha256": "98e65c74ad9b998783132f00880d1025a8d64b158e040e6ef13a557e5098bc42",
    "kid_template_sha256": "da3ca58eca7fa4204181d2f0690a17fcfdd7e7a3860587f43547c2203cf2daa2",
    "joint_regressor_sha256": "0ac1a5fd9a13b59a1f5b55e10c4f4f7a13fd93be866682d6ec75448f429cce10",
    "faces_sha256": "7ee8e99db736acf178a6078ab5710ca942edc3738d34c72f41a35c40b370e045",
    "joint_schema_sha256": "1e1ac1691e93caab23123e0d7ad046dd13271e2f4d34e322c3a8b83686e6fa59",
    "asset_sha256": "05c4b904a46a433afea0b6ececd2e82ff7aeb27d466afd9995af2115467b4c9f",
}


_SMPL_JOINT_PARENTS = {
    "pelvis": None,
    "left_hip": 0,
    "right_hip": 0,
    "spine1": 0,
    "left_knee": 1,
    "right_knee": 2,
    "spine2": 3,
    "left_ankle": 4,
    "right_ankle": 5,
    "spine3": 6,
    "left_foot": 7,
    "right_foot": 8,
    "neck": 9,
    "left_collar": 9,
    "right_collar": 9,
    "head": 12,
    "left_shoulder": 13,
    "right_shoulder": 14,
    "left_elbow": 16,
    "right_elbow": 17,
    "left_wrist": 18,
    "right_wrist": 19,
    "left_hand": 20,
    "right_hand": 21,
}
SMPL_24_JOINT_SCHEMA = {
    "schema_version": 1,
    "model": "smpl",
    "format": "smpl_24",
    "names": tuple(_SMPL_JOINT_PARENTS),
    "edges": tuple(
        (parent, joint) for joint, parent in enumerate(_SMPL_JOINT_PARENTS.values()) if parent is not None
    ),
}
SMPL_24_JOINT_SIDES = tuple(
    name.split("_", 1)[0] if name.startswith(("left_", "right_")) else "center"
    for name in SMPL_24_JOINT_SCHEMA["names"]
)


def validate_joint_schema(schema: dict) -> dict:
    """Return the fixed SMPL-24 schema after strict validation."""
    if not isinstance(schema, dict):
        raise TypeError("joint_schema must be a dict")
    actual = dict(schema)
    try:
        actual["names"] = tuple(actual["names"])
        actual["edges"] = tuple(tuple(int(index) for index in edge) for edge in actual["edges"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("joint_schema has invalid names or edges") from exc
    if actual != SMPL_24_JOINT_SCHEMA:
        raise ValueError("joint_schema does not match SMPL_24_JOINT_SCHEMA")
    return actual


def validate_pose_pack_identity(model_sha256: str, layout_id: str, metadata: dict) -> dict:
    if model_sha256 != PRODUCTION_NLF_MODEL_SHA256:
        raise ValueError("PosePack producer_model_sha256 does not match the production NLF model")
    if layout_id != PRODUCTION_NLF_QUERY_LAYOUT_ID:
        raise ValueError("PosePack query_layout_id does not match the production 1048 layout")
    if not isinstance(metadata, dict) or metadata != POSE_METADATA:
        raise ValueError("PosePack coordinate and uncertainty metadata does not match POSE_METADATA")
    return dict(metadata)


def _source_frames(value, frame_count: int) -> np.ndarray:
    frames = np.asarray(value)
    if frames.shape != (frame_count,) or not np.issubdtype(frames.dtype, np.integer):
        raise ValueError(f"source_frames must be an integer array with shape [N], got {frames.dtype} {frames.shape}")
    if np.any(frames[1:] <= frames[:-1]):
        raise ValueError("source_frames must be strictly increasing")
    return frames


def validate_body_model_asset_identity(identity: dict) -> dict:
    if not isinstance(identity, dict) or identity != BODY_MODEL_ASSET_IDENTITY:
        raise ValueError("BodyPack body_model_asset_identity does not match the production BodyModel")
    return dict(identity)


@dataclass
class Track:
    """Detection stage output after det -> track -> densify."""

    id: int
    frames: np.ndarray
    bboxes: np.ndarray
    scores: np.ndarray


@dataclass
class PoseTrack:
    """Pose stage output after crop -> 2D/3D pose."""

    id: int
    poses3d: np.ndarray
    poses2d: np.ndarray
    uncertainties: np.ndarray
    source_frames: np.ndarray | None = None


@dataclass
class BodyTrack:
    """Body stage output after SMPL fitting."""

    id: int
    betas: np.ndarray
    pose_rotvecs: np.ndarray
    joints: np.ndarray
    frame_betas: np.ndarray | None = None
    source_frames: np.ndarray | None = None
    source_joint_uncertainties_mm: np.ndarray | None = None


def normalize_pose_track(track, require_v2: bool = False) -> PoseTrack:
    get = track.get if isinstance(track, dict) else lambda name, default=None: getattr(track, name, default)
    poses3d = np.asarray(get("poses3d"))
    poses2d = np.asarray(get("poses2d"))
    uncertainties = np.asarray(get("uncertainties"))
    n = len(poses3d)
    for field, array, expected in (("poses3d", poses3d, (1048, 3)),
                                   ("poses2d", poses2d, (1048, 2)),
                                   ("uncertainties", uncertainties, (1048,))):
        if array.shape != (n, *expected):
            dims = ",".join(map(str, expected))
            raise ValueError(f"{field} must have shape [N,{dims}], got {array.shape}")
    if not all(np.all(np.isfinite(array)) for array in (poses3d, poses2d, uncertainties)):
        raise ValueError("PoseTrack arrays must contain finite values")
    if require_v2 and np.any(uncertainties <= 0):
        raise ValueError("PoseTrack uncertainties must contain positive values")

    source_frames = get("source_frames")
    if source_frames is not None:
        source_frames = _source_frames(source_frames, n)
    elif require_v2:
        raise ValueError("MotionPack pose v2 track is missing source_frames")

    return PoseTrack(int(get("id")), poses3d, poses2d, uncertainties, source_frames)


def normalize_body_track(track, allow_legacy_pose: bool = False) -> BodyTrack:
    """Validate a body track, reshaping rotvecs only for schema v0/v1."""
    get = track.get if isinstance(track, dict) else lambda name, default=None: getattr(track, name, default)
    pose_rotvecs = np.asarray(get("pose_rotvecs"))
    if allow_legacy_pose and pose_rotvecs.ndim == 2 and pose_rotvecs.shape[1] == 72:
        pose_rotvecs = pose_rotvecs.reshape(-1, 24, 3)
    if pose_rotvecs.ndim != 3 or pose_rotvecs.shape[1:] != (24, 3):
        raise ValueError(f"pose_rotvecs must have shape [N,24,3], got {pose_rotvecs.shape}")

    joints = np.asarray(get("joints"))
    if joints.shape != pose_rotvecs.shape:
        raise ValueError(f"joints shape {joints.shape} does not match pose_rotvecs {pose_rotvecs.shape}")
    betas = np.asarray(get("betas"))
    if betas.ndim != 1:
        raise ValueError(f"betas must be one-dimensional, got {betas.shape}")

    frame_betas = get("frame_betas")
    if frame_betas is not None:
        frame_betas = np.asarray(frame_betas)
        if frame_betas.shape != (len(joints), len(betas)):
            raise ValueError(f"frame_betas must have shape [N,B], got {frame_betas.shape}")

    source_frames = get("source_frames")
    if source_frames is not None:
        source_frames = _source_frames(source_frames, len(joints))

    uncertainties = get("source_joint_uncertainties_mm")
    if uncertainties is not None:
        uncertainties = np.asarray(uncertainties)
        if uncertainties.shape != (len(joints), 24):
            raise ValueError(
                "source_joint_uncertainties_mm must have shape [N,24], "
                f"got {uncertainties.shape}"
            )
        if not np.all(np.isfinite(uncertainties)) or np.any(uncertainties <= 0):
            raise ValueError("source_joint_uncertainties_mm must contain finite positive values")

    for name, array in (
        ("betas", betas),
        ("pose_rotvecs", pose_rotvecs),
        ("joints", joints),
        ("frame_betas", frame_betas),
    ):
        if array is not None and not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain finite values")

    return BodyTrack(
        id=int(get("id")),
        betas=betas,
        pose_rotvecs=pose_rotvecs,
        joints=joints,
        frame_betas=frame_betas,
        source_frames=source_frames,
        source_joint_uncertainties_mm=uncertainties,
    )
