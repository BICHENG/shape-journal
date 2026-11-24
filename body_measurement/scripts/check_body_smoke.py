"""Check the MotionPack Body measurement smoke path."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from body_measurement.mesh import (
    BODY_SEGMENTATION_ID,
    BODY_SEGMENT_IDS,
    MEASUREMENT_LANDMARKS,
    MEASUREMENT_LANDMARK_SET_ID,
    CROTCH_VERTEX_INDEX,
    JOINT_TO_BODY_SEGMENT,
    _load_smplfitter_body_model,
    generate_pose_model,
    make_body_frame,
)
from body_measurement.poses import (
    APOSE_EXPORT_PRESETS,
    LOWER_BODY_POSE_PRESETS,
    lower_body_vertex_key,
    make_apose_rotvecs,
    pose_key_stem,
)
from body_measurement.result_package import build_workbench
from core.motionpack import MotionPackBody, load_body
from core.motionpack.io import BODY_SCHEMA_VERSION
from core.motionpack.model import (
    BODY_MODEL_ASSET_IDENTITY,
    SMPL_24_JOINT_SCHEMA,
    SMPL_24_JOINT_SIDES,
)

NON_DEFAULT_APOSE_ARM_DOWN_DEGREES = 55.0
# Float reconstruction should keep one generated ring level to one micron.
HIP_CREST_NATURAL_LEVEL_TOLERANCE_MM = 1e-3
# Reusing the hits after solved leg adduction may shift them, but must remain sub-millimeter.
HIP_CREST_POSE_LEVEL_TOLERANCE_MM = 0.25


def _assert_height(vertices: np.ndarray, height_cm: float) -> None:
    _assert_close(float(np.ptp(vertices[:, 1])) / 10.0, height_cm, "mesh height")


def _assert_close(actual: float, expected: float, label: str, tolerance: float = 1e-4) -> None:
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance, equal_nan=False, err_msg=label)


def _assert_selected_apose_rotvecs(mesh: dict[str, np.ndarray], arm_down_degrees: float) -> None:
    selected = mesh["selected_apose_pose_rotvecs"]
    expected = make_apose_rotvecs(arm_down_degrees)
    np.testing.assert_allclose(selected, expected, atol=1e-6, rtol=1e-5, equal_nan=False,
                               err_msg="selected_apose_pose_rotvecs does not match requested APOSE angle")
    shoulder_angle = float(np.deg2rad(arm_down_degrees))
    left_shoulder_z = float(selected[16, 2])
    right_shoulder_z = float(selected[17, 2])
    if left_shoulder_z >= 0 or right_shoulder_z <= 0:
        raise AssertionError(
            f"APOSE shoulder rotvec z signs are wrong: left={left_shoulder_z}, right={right_shoulder_z}"
        )
    _assert_close(left_shoulder_z, -shoulder_angle, "left shoulder rotvec z", 1e-6)
    _assert_close(right_shoulder_z, shoulder_angle, "right shoulder rotvec z", 1e-6)


def _assert_canonical_crotch_structure(
    vertices: np.ndarray,
    joints: np.ndarray,
    axes: np.ndarray,
    faces: np.ndarray,
) -> None:
    """Protect the signed SMPL-neutral crotch landmark without a shape heuristic."""
    crotch = CROTCH_VERTEX_INDEX
    if crotch != 1210 or crotch == 3149:
        raise AssertionError(f"canonical crotch vertex is {crotch}, expected signed vertex 1210")

    model = _load_smplfitter_body_model()
    weights = model.weights.cpu().numpy()
    vertex_segment_weights = np.zeros((len(weights), len(BODY_SEGMENT_IDS)), dtype=np.float32)
    for joint_index, segment_index in enumerate(JOINT_TO_BODY_SEGMENT):
        vertex_segment_weights[:, segment_index] += weights[:, joint_index]
    vertex_segments = np.argmax(vertex_segment_weights, axis=1)
    adjacency = [set() for _ in range(len(vertices))]
    for face in faces:
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            adjacency[int(start)].add(int(end))
            adjacency[int(end)].add(int(start))

    coordinates = (vertices - joints[0]) @ axes.T
    crotch_point = coordinates[crotch]
    hip_coordinates = (joints[[1, 2]] - joints[0]) @ axes.T
    if not hip_coordinates[:, 0].min() < crotch_point[0] < hip_coordinates[:, 0].max():
        raise AssertionError("canonical crotch is outside the left/right hip interval")
    lower_segments = {
        BODY_SEGMENT_IDS[int(vertex_segments[index])]
        for index in adjacency[crotch]
        if coordinates[index, 1] < crotch_point[1]
    }
    if not {"left_thigh", "right_thigh"}.issubset(lower_segments):
        raise AssertionError("canonical crotch does not have both descending thigh branches")


def _assert_pose_arrays(
    mesh: dict[str, np.ndarray],
    vertex_key: str,
    height_cm: float,
    lower_body_pose_id: str,
) -> None:
    import torch

    stem = pose_key_stem(vertex_key)
    vertices = mesh[vertex_key]
    joints = mesh[f"{stem}_joints"]
    rotvecs = mesh[f"{stem}_pose_rotvecs"]
    origin = mesh[f"{stem}_body_origin"]
    axes = mesh[f"{stem}_body_axes"]
    pose_id = str(np.asarray(mesh[f"{stem}_pose_id"]).item())
    exported_lower_body_pose_id = str(np.asarray(mesh[f"{stem}_lower_body_pose_id"]).item())
    height_scale = float(np.asarray(mesh[f"{stem}_height_scale"]).item())
    for key, shape in (
        (vertex_key, (6890, 3)),
        (f"{stem}_joints", (24, 3)),
        (f"{stem}_pose_rotvecs", (24, 3)),
        (f"{stem}_body_origin", (3,)),
        (f"{stem}_body_axes", (3, 3)),
    ):
        if mesh[key].shape != shape:
            raise AssertionError(f"{key} shape {mesh[key].shape} != {shape}")
    if height_scale <= 0:
        raise AssertionError(f"{stem}_height_scale must be positive, got {height_scale}")
    if exported_lower_body_pose_id != lower_body_pose_id or not pose_id.endswith(f"__{lower_body_pose_id}"):
        raise AssertionError(f"{stem} has inconsistent pose identity")
    if vertex_key == "tpose_vertices":
        _assert_height(vertices, height_cm)
    reference_scale = float(mesh["tpose_height_scale"].item())
    _assert_close(height_scale, reference_scale, f"{stem}_height_scale", 1e-7)
    with torch.inference_mode():
        raw = _load_smplfitter_body_model()(
            pose_rotvecs=torch.from_numpy(rotvecs.reshape(1, 24, 3)),
            shape_betas=torch.from_numpy(mesh["shape_betas"].reshape(1, 10)),
        )
    for name, scaled in (("vertices", vertices), ("joints", joints)):
        expected = raw[name][0].cpu().numpy() * 1000.0 * height_scale
        np.testing.assert_allclose(scaled, expected, atol=1e-4, rtol=1e-5, equal_nan=False,
                                   err_msg=f"{stem}_{name} does not use its saved height scale")
    pelvis_index = SMPL_24_JOINT_SCHEMA["names"].index("pelvis")
    for label, actual, expected in (
        ("origin", origin, joints[pelvis_index]),
        ("orthonormal axes", axes @ axes.T, np.eye(3)),
        ("front axis", axes[2], np.cross(axes[1], axes[0])),
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5, equal_nan=False,
                                   err_msg=f"{stem} {label}")


def _assert_pose_payloads(manifest: dict, mesh: dict[str, np.ndarray]) -> None:
    payloads = manifest["pose_payloads"]
    if not {"tpose", "apose", "apose_arms_down_85deg"}.issubset(payloads):
        raise AssertionError("pose_payloads must retain tpose, apose, and A-pose 85-degree aliases")
    if "natural_standing" in payloads:
        raise AssertionError("natural_standing must remain a lower-body state, not a PosePayload alias")
    expected_schema = json.loads(json.dumps({**SMPL_24_JOINT_SCHEMA, "sides": SMPL_24_JOINT_SIDES}))
    if manifest["joint_schemas"] != {"smpl_24@1": expected_schema}:
        raise AssertionError("manifest joint_schemas does not match SMPL_24_JOINT_SCHEMA")
    expected_mesh_id = str(np.asarray(mesh["mesh_id"]).item())
    expected_joint_metadata = {
        "schema_version": 1,
        "schema_id": "smpl_24@1",
        "units": "mm",
        "coordinate_system": "smpl_mesh_local_rh",
        "source": {
            "kind": "body_model_forward",
            "track_id": int(np.asarray(mesh["track_id"]).item()),
            "frame_id": None,
            "producer": BODY_MODEL_ASSET_IDENTITY,
        },
        "uncertainty": None,
    }
    for alias, payload in payloads.items():
        if payload["mesh_id"] != expected_mesh_id:
            raise AssertionError(f"PosePayload {alias} has a different mesh_id")
        joints = payload["joints"]
        metadata = {key: value for key, value in joints.items() if key != "points"}
        if metadata != expected_joint_metadata or "points" not in joints:
            raise AssertionError(f"PosePayload {alias} has an invalid JointSet")
        if any(key in payload for key in (
            "source_track_id", "pose_source", "source_frame_id", "units", "coordinate_system",
        )):
            raise AssertionError(f"PosePayload {alias} duplicates JointSet metadata")
        lower_body_pose_id = payload.get("lower_body_pose_id")
        if lower_body_pose_id not in LOWER_BODY_POSE_PRESETS:
            raise AssertionError(f"PosePayload {alias} has an unknown lower-body state")
        references = (
            payload["vertices"],
            payload["joints"]["points"],
            payload["pose_rotvecs"],
            payload["body_frame"]["origin"],
            payload["body_frame"]["axes"],
        )
        for reference in references:
            if reference["key"] not in mesh:
                raise AssertionError(f"PosePayload {alias} references missing NPZ key {reference['key']}")
        stem = pose_key_stem(payload["vertices"]["key"])
        if payload.get("pose_id") != str(np.asarray(mesh[f"{stem}_pose_id"]).item()):
            raise AssertionError(f"PosePayload {alias} pose_id does not match its NPZ identity")
        if lower_body_pose_id != str(np.asarray(mesh[f"{stem}_lower_body_pose_id"]).item()):
            raise AssertionError(f"PosePayload {alias} lower_body_pose_id does not match its NPZ identity")
        if payload["body_frame"]["axis_order"] != ["right", "up", "front"]:
            raise AssertionError(f"PosePayload {alias} has invalid body-frame axis order")
        if payload["body_frame"].get("axis_convention") != "front=cross(up,right)":
            raise AssertionError(f"PosePayload {alias} has invalid body-frame axis convention")
        if lower_body_pose_id == "natural_standing":
            rotvecs = mesh[payload["pose_rotvecs"]["key"]]
            joints = mesh[payload["joints"]["points"]["key"]]
            if np.any(rotvecs[[1, 2, 4, 5, 7, 8]]):
                raise AssertionError(f"PosePayload {alias} does not have natural hip/knee/ankle rotvecs")
            if np.linalg.norm(joints[7] - joints[8]) <= 50.0:
                raise AssertionError(f"PosePayload {alias} does not have separated feet")


def _assert_surface_semantics(manifest: dict, mesh: dict[str, np.ndarray]) -> None:
    faces = mesh["faces"]
    face_segments = mesh["face_body_segment"]
    if faces.shape != (13776, 3) or face_segments.shape != (len(faces),):
        raise AssertionError("faces or face_body_segment has an invalid shape")
    if np.any(face_segments < 0) or np.any(face_segments >= len(BODY_SEGMENT_IDS)):
        raise AssertionError("face_body_segment contains an unknown segment id")
    segmentation = manifest["body_segmentations"][BODY_SEGMENTATION_ID]
    if segmentation["segment_ids"] != list(BODY_SEGMENT_IDS):
        raise AssertionError("manifest body segments do not match the generated segment ids")
    if segmentation["mesh_id"] != str(np.asarray(mesh["mesh_id"]).item()):
        raise AssertionError("body segmentation is bound to a different mesh")
    if segmentation["face_body_segment"]["key"] != "face_body_segment":
        raise AssertionError("manifest does not reference face_body_segment")

    landmark_set = manifest["measurement_landmark_sets"][MEASUREMENT_LANDMARK_SET_ID]
    expected_landmarks = [
        {"id": landmark_id, "side": side, "label": label}
        for landmark_id, side, label in MEASUREMENT_LANDMARKS
    ]
    if landmark_set["landmarks"] != expected_landmarks:
        raise AssertionError("MeasurementLandmarkSet contains unexpected landmarks")
    if landmark_set["mesh_id"] != str(np.asarray(mesh["mesh_id"]).item()):
        raise AssertionError("MeasurementLandmarkSet is bound to a different mesh")
    if landmark_set["landmark_kind"] != "SurfaceHit" or "pose_payload" in landmark_set:
        raise AssertionError("MeasurementLandmarkSet must contain pose-independent SurfaceHits")
    hits = landmark_set["surface_hits"]
    if set(hits) != {"face_indices", "barycentric"} or "measurement_landmark_points" in mesh:
        raise AssertionError("MeasurementLandmarkSet must not persist pose-specific points")
    for reference in hits.values():
        if reference["key"] not in mesh:
            raise AssertionError(f"MeasurementLandmarkSet references missing key {reference['key']}")
    face_indices = mesh[hits["face_indices"]["key"]]
    barycentric = mesh[hits["barycentric"]["key"]]
    count = len(MEASUREMENT_LANDMARKS)
    if face_indices.shape != (count,) or barycentric.shape != (count, 3):
        raise AssertionError("MeasurementLandmarkSet arrays have invalid shapes")
    if np.any(face_indices < 0) or np.any(face_indices >= len(faces)):
        raise AssertionError("MeasurementLandmarkSet contains an invalid face index")
    if np.any(barycentric < 0) or not np.allclose(barycentric.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("MeasurementLandmarkSet contains invalid barycentric coordinates")
    crotch_face = faces[face_indices[2]]
    if CROTCH_VERTEX_INDEX not in crotch_face or 3149 in crotch_face:
        raise AssertionError("crotch SurfaceHit does not resolve to the signed canonical vertex")
    _assert_canonical_crotch_structure(
        mesh["apose_vertices"], mesh["apose_joints"], mesh["apose_body_axes"], faces
    )
    zero_vertices, zero_joints, _, _ = generate_pose_model(np.zeros(10), 175.0)
    _, zero_axes = make_body_frame(zero_joints)
    _assert_canonical_crotch_structure(zero_vertices, zero_joints, zero_axes, faces)
    hip_segments = {BODY_SEGMENT_IDS[int(face_segments[index])] for index in face_indices[:2]}
    if hip_segments != {"pelvis", "torso"}:
        raise AssertionError("hip crest SurfaceHits do not span the pelvis-to-torso transition")
    def hip_points(alias):
        return np.sum(mesh[f"{alias}_vertices"][faces[face_indices[:2]]] * barycentric[:2, :, None], axis=1)

    natural_hips = hip_points("apose")
    natural_lateral = (natural_hips - mesh["apose_joints"][0]) @ mesh["apose_body_axes"][0]
    if not natural_lateral[0] < 0.0 < natural_lateral[1]:
        raise AssertionError("hip crest SurfaceHits do not preserve left/right ordering")
    for alias, points, tolerance in (
        ("apose", natural_hips, HIP_CREST_NATURAL_LEVEL_TOLERANCE_MM),
        ("apose_legs_together", hip_points("apose_legs_together"), HIP_CREST_POSE_LEVEL_TOLERANCE_MM),
    ):
        delta = float((points[0] - points[1]) @ mesh[f"{alias}_body_axes"][1])
        _assert_close(delta, 0.0, f"{alias} hip crest SurfaceHits level", tolerance)
    for alias in ("tpose", "apose_arms_down_85deg"):
        payload = manifest["pose_payloads"][alias]
        vertices = mesh[payload["vertices"]["key"]]
        joints = mesh[payload["joints"]["points"]["key"]]
        axes = mesh[payload["body_frame"]["axes"]["key"]]
        points = np.sum(vertices[faces[face_indices]] * barycentric[:, :, None], axis=1)
        if not np.all(np.isfinite(points)):
            raise AssertionError(f"SurfaceHits did not reconstruct on PosePayload {alias}")
        landmark_joints = (1, 2, 0, 4, 5, 7, 7, 8, 8)
        if any(np.linalg.norm(point - joints[joint]) <= 5.0 for point, joint in zip(points, landmark_joints)):
            raise AssertionError(f"a SurfaceHit collapsed to a joint center on PosePayload {alias}")
        if np.any((points[3:5] - joints[[4, 5]]) @ axes[2] <= 0):
            raise AssertionError(f"patella SurfaceHits are behind the knees on PosePayload {alias}")
        if np.dot(points[6] - points[5], -axes[0]) <= 0 or np.dot(points[8] - points[7], axes[0]) <= 0:
            raise AssertionError(f"malleolus ordering is invalid on PosePayload {alias}")


def run_smoke(body_path: Path | MotionPackBody, output_dir: Path, height_cm: float) -> dict:
    body = body_path if isinstance(body_path, MotionPackBody) else load_body(body_path)
    if body.schema_version != BODY_SCHEMA_VERSION:
        raise ValueError(
            f"BodyPack {body.path} uses schema_version {body.schema_version}; "
            "rebuild PosePack and BodyPack from the source video before measurement"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    track = body.select_track()
    package = build_workbench(
        body,
        output_dir / "result_package",
        height_cm=height_cm,
        apose_arm_down_degrees=NON_DEFAULT_APOSE_ARM_DOWN_DEGREES,
        source_identity={
            "subject_id": "smoke-subject",
            "session_id": "smoke-session",
            "subject_name": "测试人物",
            "device_id": "smoke-device",
            "device_model": "测试设备",
        },
        reference_measurements=[{
            "session_id": "smoke-session",
            "metric_id": "hip_circumference",
            "side": "unknown",
            "value_cm": 90.0,
        }],
    )
    manifest = json.loads(Path(package["manifest_path"]).read_text(encoding="utf-8"))
    if manifest["source_display"]["subject"] != "测试人物" or manifest["source_display"]["group"] != "测试设备":
        raise AssertionError("package source display does not carry subject and device identity")
    if manifest["reference_measurements"][0]["value_cm"] != 90.0:
        raise AssertionError("package does not carry reference measurements")
    mesh_path = Path(package["package_dir"]) / manifest["artifacts"]["meshes"]
    with np.load(mesh_path) as archive:
        mesh = {key: archive[key] for key in archive.files}
    _assert_pose_payloads(manifest, mesh)
    _assert_surface_semantics(manifest, mesh)
    base_vertex_keys = ["tpose_vertices", "apose_vertices", *(preset.vertex_key for preset in APOSE_EXPORT_PRESETS)]
    for vertex_key in base_vertex_keys:
        for lower_body_pose_id in LOWER_BODY_POSE_PRESETS:
            key = lower_body_vertex_key(vertex_key, lower_body_pose_id)
            _assert_pose_arrays(mesh, key, height_cm, lower_body_pose_id)
    _assert_close(
        float(mesh["selected_apose_arm_down_degrees"].item()),
        NON_DEFAULT_APOSE_ARM_DOWN_DEGREES,
        "selected_apose_arm_down_degrees",
    )
    _assert_selected_apose_rotvecs(mesh, NON_DEFAULT_APOSE_ARM_DOWN_DEGREES)
    pose_semantics = {"arm_down_degrees": NON_DEFAULT_APOSE_ARM_DOWN_DEGREES}
    joints = mesh["apose_joints"]
    scale = float(mesh["apose_height_scale"].item())
    names = SMPL_24_JOINT_SCHEMA["names"]
    for side in ("left", "right"):
        shoulder = joints[names.index(f"{side}_shoulder"), 1]
        for joint, minimum in (("elbow", 10.0), ("wrist", 25.0)):
            drop = float(shoulder - joints[names.index(f"{side}_{joint}"), 1]) / scale
            if drop < minimum:
                raise AssertionError(f"Saved A-pose {side} {joint} only dropped {drop:.2f} mm")
            pose_semantics[f"{side}_{joint}_drop_mm"] = drop
    if mesh["shape_betas"].shape != (10,):
        raise AssertionError(f"shape_betas shape {mesh['shape_betas'].shape} != (10,)")
    pose_status = manifest["pose_generation_status"]
    if set(pose_status) != {"natural_standing", "legs_together", "legs_shoulder_width"}:
        raise AssertionError("pose_generation_status contains an invalid lower-body state")
    for pose_name, status in pose_status.items():
        expected_aliases = [
            alias
            for alias, payload in manifest["pose_payloads"].items()
            if payload["lower_body_pose_id"] == pose_name
        ]
        if status.get("available") is not True or status.get("compatible_pose_payloads") != expected_aliases:
            raise AssertionError(f"{pose_name} does not list its generated PosePayloads")
        if "reason" in status or "next_step" in status:
            raise AssertionError(f"available {pose_name} retains unavailable-state text")
    gap_mm = float(np.asarray(mesh["legs_together_foot_surface_gap_mm"]).item())
    if not 0.0 <= gap_mm <= 5.0:
        raise AssertionError(f"legs_together foot surface gap {gap_mm}mm is outside 0-5mm")
    _assert_close(
        float(pose_status["legs_together"]["validation"]["foot_surface_gap_mm"]),
        gap_mm,
        "legs_together manifest gap",
    )
    ankle_spacing_mm = float(np.asarray(mesh["legs_shoulder_width_ankle_spacing_mm"]).item())
    target_mm = float(np.asarray(mesh["legs_shoulder_width_target_mm"]).item())
    target_basis = str(np.asarray(mesh["legs_shoulder_width_target_basis"]).item())
    if abs(ankle_spacing_mm - target_mm) > 5.0:
        raise AssertionError("legs_shoulder_width ankle spacing is outside its 5mm tolerance")
    shoulder_validation = pose_status["legs_shoulder_width"]["validation"]
    _assert_close(float(shoulder_validation["ankle_spacing_mm"]), ankle_spacing_mm, "manifest ankle spacing")
    _assert_close(float(shoulder_validation["target_mm"]), target_mm, "manifest shoulder target")
    if shoulder_validation.get("target_basis") != target_basis:
        raise AssertionError("legs_shoulder_width target basis does not match its NPZ value")

    return {
        "body": str(body.path),
        "output_dir": str(output_dir),
        "height_cm": height_cm,
        "valid_frames": len(track.joints),
        "selected_apose_arm_down_degrees": NON_DEFAULT_APOSE_ARM_DOWN_DEGREES,
        "mesh_keys": sorted(mesh),
        "selected_apose_semantics": pose_semantics,
        "selected_apose_readback_checked": True,
        "pose_payloads_checked": sorted(manifest["pose_payloads"]),
        "lower_body_pose_validation_checked": True,
        "surface_semantics_checked": True,
        "accuracy_validated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MotionPack .body measurement smoke")
    parser.add_argument("--body", required=True, help="MotionPack .body path")
    parser.add_argument("--height", type=float, default=170.0, help="Target height in cm")
    parser.add_argument(
        "--output-dir",
        default=str(Path(tempfile.gettempdir()) / "body_measurement_dev" / "measurement_smoke"),
    )
    args = parser.parse_args()
    summary = run_smoke(Path(args.body), Path(args.output_dir), args.height)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
