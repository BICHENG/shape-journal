"""Build consistently scaled SMPL meshes, poses and surface landmarks."""

import hashlib
import warnings
from functools import cache
from pathlib import Path
from typing import Any, Optional

import numpy as np

import config as cfg
from body_measurement.poses import (
    LOWER_BODY_POSE_PRESETS,
    APOSE_EXPORT_PRESETS,
    TPOSE_PRESET,
    lower_body_vertex_key,
    make_apose_rotvecs,
    make_lower_body_pose_rotvecs,
    measurement_pose_id,
    pose_key_stem,
    selected_apose,
)
from body_measurement.tape_state import run_measurement_engine
from core.motionpack import MotionPackBody
from core.motionpack.model import SMPL_24_JOINT_SCHEMA
from core.smpl_assets import current_body_model_asset_identity

try:
    from numpy.exceptions import VisibleDeprecationWarning
except ImportError:
    VisibleDeprecationWarning = DeprecationWarning

BODY_SEGMENTATION_ID = "smpl_skinning_body_segments_v1"
BODY_SEGMENT_IDS = (
    "pelvis",
    "torso",
    "head",
    "left_thigh",
    "left_shin",
    "left_foot",
    "right_thigh",
    "right_shin",
    "right_foot",
    "left_upper_arm",
    "left_forearm",
    "left_hand",
    "right_upper_arm",
    "right_forearm",
    "right_hand",
)
JOINT_TO_BODY_SEGMENT = np.array(
    (0, 3, 6, 1, 4, 7, 1, 5, 8, 1, 5, 8, 2, 1, 1, 2, 9, 12, 10, 13, 11, 14, 11, 14),
    dtype=np.uint8,
)
MEASUREMENT_LANDMARK_SET_ID = "lower_body_surface_hits_v2"
LEGS_TOGETHER_TARGET_GAP_MM = 2.5
LOWER_BODY_MAX_HIP_ANGLE_RADIANS = float(np.deg2rad(15.0))
PATELLA_SEARCH_HALF_HEIGHT_MM = 80.0
MALLEOLUS_SEARCH_HALF_HEIGHT_MM = 45.0
MALLEOLUS_SEARCH_RADIUS_MM = 70.0
MEASUREMENT_LANDMARKS = (
    ("left_hip_crest", "left", "左胯骨顶"),
    ("right_hip_crest", "right", "右胯骨顶"),
    ("crotch", "center", "会阴"),
    ("left_patella", "left", "左髌骨"),
    ("right_patella", "right", "右髌骨"),
    ("left_medial_malleolus", "left", "左内踝"),
    ("left_lateral_malleolus", "left", "左外踝"),
    ("right_medial_malleolus", "right", "右内踝"),
    ("right_lateral_malleolus", "right", "右外踝"),
)


@cache
def _load_smplfitter_body_model():
    import smplfitter.pt

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
        return smplfitter.pt.BodyModel("smpl", model_root=str(cfg.SMPL_FITTER_ROOT), num_betas=10, device="cpu")


def generate_face_body_segment(faces: np.ndarray) -> np.ndarray:
    """Classify faces by aggregated SMPL skinning weight in face order."""
    return _surface_topology(np.asarray(faces, dtype=np.int32))["face_segments"].copy()


def _surface_topology(faces: np.ndarray) -> dict[str, np.ndarray]:
    topology = _smpl_topology()
    if not np.array_equal(faces, topology["faces"]):
        raise ValueError("faces do not match the loaded SMPL topology")
    return topology


@cache
def _smpl_topology() -> dict[str, np.ndarray]:
    body_model = _load_smplfitter_body_model()
    faces = np.asarray(body_model.faces, dtype=np.int32)
    weights = body_model.weights.cpu().numpy()
    vertex_segment_weights = np.zeros((len(weights), len(BODY_SEGMENT_IDS)), dtype=np.float32)
    for joint_index, segment_index in enumerate(JOINT_TO_BODY_SEGMENT):
        vertex_segment_weights[:, segment_index] += weights[:, joint_index]
    edges = np.sort(faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1)
    _, groups, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    pairs = np.argsort(groups.ravel(), kind="stable")[np.repeat(counts == 2, counts)].reshape(-1, 2)
    pairs = pairs[np.argsort(pairs[:, 0])]
    return {
        "faces": faces,
        "dominant_joints": weights.argmax(axis=1),
        "face_segments": vertex_segment_weights[faces].sum(axis=1).argmax(axis=1).astype(np.uint8),
        "edges": edges[pairs[:, 0]],
        "adjacent_faces": pairs // 3,
    }


def generate_pose_model(
    betas: np.ndarray,
    target_height_cm: float,
    pose_rotvecs: Optional[np.ndarray] = None,
    *,
    height_scale: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate one SMPL pose in mm using the body's fixed reference scale."""
    import torch

    body_model = _load_smplfitter_body_model()
    betas_arr = np.asarray(betas, dtype=np.float32).reshape(-1)[:10]
    if betas_arr.shape != (10,) or not np.all(np.isfinite(betas_arr)):
        raise ValueError(f"SMPL betas must contain 10 finite values, got {betas_arr.shape}")
    if not np.isfinite(target_height_cm) or target_height_cm <= 0:
        raise ValueError(f"target_height_cm must be positive and finite, got {target_height_cm}")
    pose = (
        np.zeros((24, 3), dtype=np.float32)
        if pose_rotvecs is None
        else np.asarray(pose_rotvecs, dtype=np.float32)
    )
    if pose.shape != (24, 3):
        raise ValueError(f"pose_rotvecs shape {pose.shape} != (24, 3)")
    if not np.all(np.isfinite(pose)):
        raise ValueError("pose_rotvecs must contain finite values")
    shape = torch.from_numpy(betas_arr.reshape(1, 10))

    def forward(rotvecs: np.ndarray):
        with torch.inference_mode():
            return body_model(
                pose_rotvecs=torch.from_numpy(rotvecs.reshape(1, 24, 3)),
                shape_betas=shape,
            )

    out = forward(pose)
    vertices_m = out["vertices"][0].cpu().numpy()
    joints_m = out["joints"][0].cpu().numpy()
    if vertices_m.shape != (6890, 3):
        raise ValueError(f"SMPL vertices shape {vertices_m.shape} != (6890, 3)")
    if joints_m.shape != (24, 3):
        raise ValueError(f"SMPL joints shape {joints_m.shape} != (24, 3)")
    if height_scale is None:
        reference = (
            forward(np.zeros_like(pose))["vertices"][0].cpu().numpy()
            if np.any(pose)
            else vertices_m
        )
        reference_height_mm = float(np.ptp(reference[:, 1]) * 1000.0)
        if not np.isfinite(reference_height_mm) or reference_height_mm <= 0:
            raise ValueError("SMPL reference mesh height must be positive and finite")
        height_scale = target_height_cm * 10.0 / reference_height_mm
    if not np.isfinite(height_scale) or height_scale <= 0:
        raise ValueError("SMPL height scale must be positive and finite")
    vertices = (vertices_m * 1000.0 * height_scale).astype(np.float32)
    joints = (joints_m * 1000.0 * height_scale).astype(np.float32)
    return vertices, joints, pose.copy(), float(height_scale)


def make_body_frame(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return pelvis origin and orthonormal right/up/front axes for one SMPL pose."""
    joints = np.asarray(joints, dtype=np.float32)
    if joints.shape != (24, 3):
        raise ValueError(f"SMPL joints shape {joints.shape} != (24, 3)")
    names = SMPL_24_JOINT_SCHEMA["names"]
    pelvis = joints[names.index("pelvis")]
    right = joints[names.index("right_hip")] - joints[names.index("left_hip")]
    up = joints[names.index("spine3")] - pelvis

    def normalize(axis: np.ndarray) -> np.ndarray:
        length = float(np.linalg.norm(axis))
        if length <= 1e-8:
            raise ValueError("Cannot construct body frame from coincident SMPL joints")
        return axis / length

    right = normalize(right)
    up = normalize(up - np.dot(up, right) * right)
    front = normalize(np.cross(up, right))
    right = normalize(np.cross(front, up))
    axes = np.stack((right, up, front)).astype(np.float32)
    if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-5):
        raise ValueError("SMPL body frame is not orthonormal")
    return pelvis.copy(), axes


def measure_lower_body_pose_spacing(
    vertices: np.ndarray,
    joints: np.ndarray,
) -> dict[str, float]:
    """Measure medial foot gap plus ankle and shoulder center spacing in millimeters."""
    vertices = np.asarray(vertices, dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32)
    if vertices.shape != (6890, 3) or joints.shape != (24, 3):
        raise ValueError("lower-body spacing requires SMPL vertices (6890, 3) and joints (24, 3)")
    names = SMPL_24_JOINT_SCHEMA["names"]
    dominant_joint = _smpl_topology()["dominant_joints"]
    left_foot = np.isin(
        dominant_joint,
        (names.index("left_ankle"), names.index("left_foot")),
    )
    right_foot = np.isin(
        dominant_joint,
        (names.index("right_ankle"), names.index("right_foot")),
    )
    if not np.any(left_foot) or not np.any(right_foot):
        raise ValueError("SMPL skinning weights do not identify both feet")
    _, axes = make_body_frame(joints)
    lateral_vertices = vertices @ axes[0]
    lateral_joints = joints @ axes[0]
    return {
        "foot_surface_gap_mm": float(
            lateral_vertices[right_foot].min() - lateral_vertices[left_foot].max()
        ),
        "ankle_spacing_mm": float(
            lateral_joints[names.index("right_ankle")]
            - lateral_joints[names.index("left_ankle")]
        ),
        "shoulder_spacing_mm": float(
            lateral_joints[names.index("right_shoulder")]
            - lateral_joints[names.index("left_shoulder")]
        ),
    }


def solve_lower_body_pose_solutions(
    betas: np.ndarray,
    target_height_cm: float,
    *,
    height_scale: Optional[float] = None,
) -> dict[str, dict[str, float | str]]:
    """Solve the two measurement stances in mm at the reference body scale."""
    return _solve_stances(
        betas, target_height_cm, generate_pose_model(betas, target_height_cm, height_scale=height_scale),
    )


def _solve_stances(betas, target_height_cm, reference):
    vertices, joints, base_pose, height_scale = reference
    natural = measure_lower_body_pose_spacing(vertices, joints)
    cache = {0.0: natural}

    def evaluate(angle: float) -> dict[str, float]:
        key = float(angle)
        if key not in cache:
            pose = make_lower_body_pose_rotvecs(base_pose, "legs_together", key)
            vertices, joints, _, _ = generate_pose_model(
                betas, target_height_cm, pose, height_scale=height_scale
            )
            cache[key] = measure_lower_body_pose_spacing(vertices, joints)
        return cache[key]

    targets = {
        "legs_together": ("foot_surface_gap_mm", LEGS_TOGETHER_TARGET_GAP_MM),
        "legs_shoulder_width": ("ankle_spacing_mm", natural["shoulder_spacing_mm"]),
    }
    solutions: dict[str, dict[str, float | str]] = {
        "natural_standing": {
            "hip_adduction_radians": 0.0,
            **natural,
            "target_basis": LOWER_BODY_POSE_PRESETS["natural_standing"],
        }
    }
    for pose_id, (metric, target_mm) in targets.items():
        lower = -LOWER_BODY_MAX_HIP_ANGLE_RADIANS
        upper = LOWER_BODY_MAX_HIP_ANGLE_RADIANS
        lower_value = evaluate(lower)[metric]
        upper_value = evaluate(upper)[metric]
        if not lower_value >= target_mm >= upper_value:
            raise ValueError(
                f"{pose_id} target {target_mm:.3f} mm is outside the reachable "
                f"range [{upper_value:.3f}, {lower_value:.3f}] mm"
            )
        for _ in range(14):
            angle = (lower + upper) / 2.0
            if evaluate(angle)[metric] > target_mm:
                lower = angle
            else:
                upper = angle
        angle = (lower + upper) / 2.0
        measured = evaluate(angle)
        solutions[pose_id] = {
            "hip_adduction_radians": angle,
            **measured,
            "target_metric": metric,
            "target_mm": target_mm,
            "target_basis": LOWER_BODY_POSE_PRESETS[pose_id],
        }

    together_gap = float(solutions["legs_together"]["foot_surface_gap_mm"])
    if not 0.0 <= together_gap <= 5.0:
        raise ValueError(f"legs_together foot surface gap {together_gap:.3f} mm is outside 0-5 mm")
    shoulder = solutions["legs_shoulder_width"]
    if abs(float(shoulder["ankle_spacing_mm"]) - float(shoulder["target_mm"])) > 5.0:
        raise ValueError("legs_shoulder_width did not reach the provisional shoulder-joint target")
    return solutions


# SMPL顶点索引定义 (基于SMPL-Anthropometry研究)
# 这些索引对应SMPL模型的特定解剖位置
CROTCH_VERTEX_INDEX = 1210


def generate_hip_crest_hits(
    vertices: np.ndarray, joints: np.ndarray, body_axes: np.ndarray, faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate topology-stable hip-crest hits on the pelvis-to-torso ring."""
    topology = _surface_topology(faces)
    surface = vertices.astype(np.float64)
    frame_origin = joints[0].astype(np.float64)
    right64, up64, front64 = body_axes.astype(np.float64)
    face_segments = topology["face_segments"]
    triangles = surface[faces]
    facing = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) @ front64
    edges, neighbors = topology["edges"], topology["adjacent_faces"]
    silhouette = facing[neighbors].prod(axis=1) <= 0
    silhouette_segments: dict[int, set[int]] = {}
    for edge, adjacent in zip(edges[silhouette], neighbors[silhouette], strict=True):
        for vertex_index in edge:
            silhouette_segments.setdefault(vertex_index, set()).update(face_segments[adjacent])

    pelvis = BODY_SEGMENT_IDS.index("pelvis")
    torso = BODY_SEGMENT_IDS.index("torso")
    transitions = np.asarray([
        vertex_index
        for vertex_index, segments in silhouette_segments.items()
        if pelvis in segments and torso in segments
    ], dtype=np.int32)
    lateral = (surface - frame_origin) @ right64
    left = transitions[lateral[transitions] < 0.0]
    right_side = transitions[lateral[transitions] > 0.0]
    if not len(left) or not len(right_side):
        raise ValueError("pelvis-to-torso silhouette transition is missing on one side")
    transition_vertices = (
        int(left[np.argmin(lateral[left])]),
        int(right_side[np.argmax(lateral[right_side])]),
    )
    height = float(((surface[list(transition_vertices)] - frame_origin) @ up64).mean())
    plane_origin = frame_origin + height * up64
    allowed_faces = np.flatnonzero(np.isin(face_segments, (pelvis, torso)))
    ring = run_measurement_engine(landmark_section={
        "vertices": surface.tolist(),
        "faces": faces.tolist(),
        "plane_origin": plane_origin.tolist(),
        "plane_normal": up64.tolist(),
        "allowed_faces": allowed_faces.tolist(),
    })
    ring_faces, ring_weights = np.asarray(ring["path_faces"], dtype=np.int32), np.asarray(ring["path_weights"])
    ring_points = np.sum(surface[faces[ring_faces]] * ring_weights[:, :, None], axis=1)
    ring_lateral = (ring_points - frame_origin) @ right64
    left_right = [int(np.argmin(ring_lateral)), int(np.argmax(ring_lateral))]
    return ring_faces[left_right], ring_weights[left_right].astype(np.float32)


def generate_measurement_landmark_hits(
    vertices: np.ndarray,
    joints: np.ndarray,
    body_axes: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate pose-independent lower-body surface hits as face indices and barycentrics."""
    vertices = np.asarray(vertices, dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32)
    body_axes = np.asarray(body_axes, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.shape != (6890, 3) or joints.shape != (24, 3) or body_axes.shape != (3, 3):
        raise ValueError("invalid SMPL arrays for measurement landmarks")
    topology = _surface_topology(faces)
    dominant_joint = topology["dominant_joints"]
    names = SMPL_24_JOINT_SCHEMA["names"]
    right, up, front = body_axes

    def patella(side: str) -> int:
        hip = names.index(f"{side}_hip")
        knee = names.index(f"{side}_knee")
        offset = vertices - joints[knee]
        along_up = offset @ up
        candidates = np.flatnonzero(
            np.isin(dominant_joint, (hip, knee))
            & (np.abs(along_up) <= PATELLA_SEARCH_HALF_HEIGHT_MM)
        )
        if not len(candidates):
            raise ValueError(f"no {side} patella surface candidates")
        score = offset[candidates] @ front - np.abs(along_up[candidates])
        return int(candidates[np.argmax(score)])

    def malleoli(side: str) -> tuple[int, int]:
        ankle = names.index(f"{side}_ankle")
        foot = names.index(f"{side}_foot")
        offset = vertices - joints[ankle]
        candidates = np.flatnonzero(
            np.isin(dominant_joint, (ankle, foot))
            & (np.abs(offset @ up) <= MALLEOLUS_SEARCH_HALF_HEIGHT_MM)
            & (np.linalg.norm(offset, axis=1) <= MALLEOLUS_SEARCH_RADIUS_MM)
        )
        if len(candidates) < 2:
            raise ValueError(f"not enough {side} malleolus surface candidates")
        outward = -right if side == "left" else right
        score = offset[candidates] @ outward
        return int(candidates[np.argmin(score)]), int(candidates[np.argmax(score)])

    left_medial, left_lateral = malleoli("left")
    right_medial, right_lateral = malleoli("right")
    vertex_indices = (
        CROTCH_VERTEX_INDEX,
        patella("left"),
        patella("right"),
        left_medial,
        left_lateral,
        right_medial,
        right_lateral,
    )
    hip_faces, hip_weights = generate_hip_crest_hits(vertices, joints, body_axes, faces)
    face_indices = np.empty(len(MEASUREMENT_LANDMARKS), dtype=np.int32)
    barycentric = np.zeros((len(MEASUREMENT_LANDMARKS), 3), dtype=np.float32)
    face_indices[:2] = hip_faces
    barycentric[:2] = hip_weights
    for hit_index, vertex_index in enumerate(vertex_indices, start=2):
        match = np.argwhere(faces == vertex_index)
        if not len(match):
            raise ValueError(f"SMPL landmark vertex {vertex_index} is not used by any face")
        face_index, corner = match[0]
        face_indices[hit_index] = face_index
        barycentric[hit_index, corner] = 1.0
    return face_indices, barycentric


def _mesh_faces() -> np.ndarray:
    faces = np.asarray(_load_smplfitter_body_model().faces, dtype=np.int32)
    if cfg.SMPL_FACES_6890_PATH.exists():
        configured_faces = np.asarray(np.load(cfg.SMPL_FACES_6890_PATH), dtype=np.int32)
        if not np.array_equal(faces, configured_faces):
            raise ValueError("configured SMPL faces do not match the BodyModel topology")
    return faces


def _mesh_id(body_path: Path, track, target_height_cm: float, faces: np.ndarray) -> str:
    digest = hashlib.sha256()
    with body_path.open("rb") as body_file:
        digest.update(b"body\0" + hashlib.file_digest(body_file, "sha256").digest())
    digest.update(b"track\0" + np.asarray(track.id, dtype="<i8").tobytes())
    digest.update(np.asarray(track.betas, dtype="<f4").reshape(-1)[:10].tobytes())
    smpl_asset = getattr(track, "smpl_asset_sha256", None)
    if smpl_asset:
        digest.update(b"smpl\0" + str(smpl_asset).encode("ascii"))
    else:
        with cfg.SMPL_NEUTRAL_PATH.open("rb") as smpl_file:
            digest.update(b"smpl\0" + hashlib.file_digest(smpl_file, "sha256").digest())
    digest.update(b"topology\0smpl-6890-13776-v1\0")
    digest.update(b"faces\0" + np.asarray(faces, dtype="<i4").tobytes())
    digest.update(b"body-segments\0" + BODY_SEGMENTATION_ID.encode("ascii"))
    digest.update(b"landmarks\0" + MEASUREMENT_LANDMARK_SET_ID.encode("ascii"))
    digest.update(b"height-scale\0tpose_zero_v1__natural_standing\0")
    digest.update(np.asarray(target_height_cm, dtype="<f4").tobytes())
    return digest.hexdigest()[:24]


SMPL_24_JOINT_SCHEMA_ID = "smpl_24@1"


def build_measurement_mesh(
    body: MotionPackBody,
    height_cm: float,
    *,
    track_id: int | None = None,
    arm_down_degrees: float,
) -> dict[str, Any]:
    if body.body_model_asset_identity is None or body.body_model_asset_identity != current_body_model_asset_identity():
        raise ValueError("BodyPack BodyModel component identity does not match the measurement BodyModel")
    track = body.select_track(track_id)
    faces = _mesh_faces()
    reference = generate_pose_model(track.betas, height_cm)
    height_scale = reference[3]
    lower_body_solutions = _solve_stances(track.betas, height_cm, reference)
    payload: dict[str, Any] = {
        "track_id": int(track.id),
        "target_height_cm": height_cm,
        "shape_betas": np.asarray(track.betas, dtype=np.float32)[:10],
        "mesh_id": _mesh_id(body.path, track, height_cm, faces),
        "joint_schema_id": SMPL_24_JOINT_SCHEMA_ID,
        "units": "mm",
        "coordinate_system": "smpl_mesh_local_rh",
        "faces": faces,
        "face_body_segment": generate_face_body_segment(faces),
        "body_segmentation_id": BODY_SEGMENTATION_ID,
        "measurement_landmark_set_id": MEASUREMENT_LANDMARK_SET_ID,
        "lower_body_pose_ids": np.array(list(LOWER_BODY_POSE_PRESETS)),
        "lower_body_hip_adduction_radians": np.array(
            [
                lower_body_solutions[pose_id]["hip_adduction_radians"]
                for pose_id in LOWER_BODY_POSE_PRESETS
            ],
            dtype=np.float32,
        ),
        "legs_together_foot_surface_gap_mm": np.array(
            lower_body_solutions["legs_together"]["foot_surface_gap_mm"], dtype=np.float32
        ),
        "legs_shoulder_width_ankle_spacing_mm": np.array(
            lower_body_solutions["legs_shoulder_width"]["ankle_spacing_mm"], dtype=np.float32
        ),
        "legs_shoulder_width_target_mm": np.array(
            lower_body_solutions["legs_shoulder_width"]["target_mm"], dtype=np.float32
        ),
        "legs_shoulder_width_target_basis": np.array(
            lower_body_solutions["legs_shoulder_width"]["target_basis"]
        ),
    }
    selected = selected_apose(arm_down_degrees)
    geometry = {reference[2].tobytes(): (*reference[:3], *make_body_frame(reference[1]))}
    for preset in (TPOSE_PRESET, selected, *APOSE_EXPORT_PRESETS):
        upper_pose = reference[2] if preset is TPOSE_PRESET else make_apose_rotvecs(preset.arm_down_degrees)
        for stance, solution in lower_body_solutions.items():
            key = lower_body_vertex_key(preset.vertex_key, stance)
            stem = pose_key_stem(key)
            pose = make_lower_body_pose_rotvecs(upper_pose, stance, float(solution["hip_adduction_radians"]))
            rotation_key = pose.tobytes()
            if rotation_key not in geometry:
                vertices, joints, rotvecs, _ = generate_pose_model(
                    track.betas, height_cm, pose, height_scale=height_scale,
                )
                geometry[rotation_key] = (vertices, joints, rotvecs, *make_body_frame(joints))
            vertices, joints, rotvecs, origin, axes = geometry[rotation_key]
            payload.update({
                key: vertices,
                f"{stem}_joints": joints,
                f"{stem}_pose_rotvecs": rotvecs,
                f"{stem}_body_origin": origin,
                f"{stem}_body_axes": axes,
                f"{stem}_height_scale": np.array(height_scale, dtype=np.float32),
                f"{stem}_pose_id": np.array(measurement_pose_id(preset.id, stance)),
                f"{stem}_lower_body_pose_id": np.array(stance),
            })
    payload["selected_apose_vertices"] = payload["apose_vertices"]
    payload["selected_apose_arm_down_degrees"] = np.array(arm_down_degrees, dtype=np.float32)
    payload["selected_apose_pose_rotvecs"] = payload["apose_pose_rotvecs"]
    payload["apose_preset_ids"] = np.array([preset.id for preset in APOSE_EXPORT_PRESETS])
    payload["apose_vertex_keys"] = np.array([preset.vertex_key for preset in APOSE_EXPORT_PRESETS])
    payload["apose_arm_down_degrees"] = np.array(
        [preset.arm_down_degrees for preset in APOSE_EXPORT_PRESETS], dtype=np.float32
    )
    face_indices, barycentric = generate_measurement_landmark_hits(
        payload["apose_vertices"],
        payload["apose_joints"],
        payload["apose_body_axes"],
        faces,
    )
    payload["measurement_landmark_face_indices"] = face_indices
    payload["measurement_landmark_barycentric"] = barycentric
    return payload
