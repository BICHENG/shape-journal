"""Build the files for a new portable measurement workbench."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from io import StringIO
import json
from math import isfinite
from pathlib import Path
from uuid import uuid4

import numpy as np

from body_measurement.mesh import (
    BODY_SEGMENTATION_ID,
    BODY_SEGMENT_IDS,
    MEASUREMENT_LANDMARKS,
    MEASUREMENT_LANDMARK_SET_ID,
    build_measurement_mesh,
)
from body_measurement.poses import (
    DEFAULT_APOSE_PRESET,
    LOWER_BODY_POSE_PRESETS,
    lower_body_vertex_key,
    packed_pose_presets,
    pose_key_stem,
)
from body_measurement.tape_state import run_measurement_engine, surface_path_digest
from body_measurement.viewer import (
    MEASUREMENT_DEFINITIONS_PATH,
    SOURCE_REVIEW_DIR_NAME,
    VIEWER_ASSET_RUNTIME_FILES,
    VIEWER_CONTRACT_VERSION,
    WORKBENCH_TAPE_SCHEMA_ID,
    ViewerPackage,
    _ensure_viewer_assets,
    _project_viewer_payload,
    _safe_display_file_stem,
    build_viewer_payload,
    read_definition_catalog,
    render_viewer_html,
    validate_viewer_state,
)
from core.motionpack import MotionPackBody
from core.motionpack.io import _file_sha256
from core.output_directory import new_output_directory
from core.motionpack.model import (
    BODY_MODEL_ASSET_IDENTITY,
    SMPL_24_JOINT_SCHEMA,
    SMPL_24_JOINT_SIDES,
)


DEFAULT_HEIGHT_CM = 170
RECORDS_ARTIFACT = "surface_tape_records"


@dataclass(frozen=True)
class MeasurementSnapshot:
    workspace: Path
    progress_path: Path
    csv_path: Path
    progress_sha256: str
    csv_sha256: str
    progress: dict
    csv_rows: list[dict]


    def public_rows(self, contract: dict[tuple[str, str], dict], subject_id: str) -> dict:
        measurements = self.progress.get("measurements")
        if not isinstance(measurements, list) or any(not isinstance(record, dict) for record in measurements):
            raise ValueError("SurfaceTapeRecord collection must contain objects")
        public_ids = {item_id for item_id, _ in contract}
        records = []
        seen: set[tuple[str, str]] = set()
        for record in measurements:
            if record.get("item_id") not in public_ids:
                continue
            key = (record.get("item_id"), record.get("side"))
            definition = contract.get(key)
            if definition is None or key in seen:
                raise ValueError(f"unexpected or duplicate SurfaceTapeRecord identity: {key}")
            for field in ("id", "item_id", "metric_id", "definition_id", "version", "subject_id"):
                if not isinstance(record.get(field), str) or not record[field]:
                    raise ValueError(f"SurfaceTapeRecord {key} is missing {field}")
            if record["subject_id"] != subject_id:
                raise ValueError(f"SurfaceTapeRecord {key} has wrong subject_id")
            seen.add(key)
            if record["definition_id"] != definition["definition_id"] or record["version"] != definition["version"]:
                continue
            if record["metric_id"] != definition["metric_id"]:
                raise ValueError(f"SurfaceTapeRecord {key} has wrong metric_id")
            if record.get("preset_schema_id") != WORKBENCH_TAPE_SCHEMA_ID:
                raise ValueError(f"SurfaceTapeRecord {key} has wrong preset_schema_id")
            if not all(isinstance(record.get(field), list) for field in ("path_faces", "path_weights")):
                raise ValueError(f"SurfaceTapeRecord {key} has no saved path")
            if record.get("path_sha256") != surface_path_digest(record):
                raise ValueError(f"SurfaceTapeRecord {key} path SHA-256 mismatch")
            value_mm = record.get("value_mm")
            if not isinstance(value_mm, (int, float)) or isinstance(value_mm, bool) or not isfinite(value_mm) or value_mm <= 0:
                raise ValueError(f"SurfaceTapeRecord {key} has no positive value_mm")
            records.append(record)
        return verify_snapshot_rows(self.csv_rows, records, contract.keys())


def _snapshot_files(package: dict, digests: tuple[str, str] | None = None) -> MeasurementSnapshot:
    workspace = Path(package["package_dir"]).resolve()
    paths = []
    for key in (RECORDS_ARTIFACT, "csv"):
        name = package["artifacts"].get(key)
        if not isinstance(name, str) or not name:
            raise ValueError(f"Measurement package is missing artifacts.{key}")
        path = (workspace / name).resolve()
        if not path.is_relative_to(workspace) or not path.is_file():
            raise ValueError(f"Measurement package artifact is missing or outside package: {key}")
        paths.append(path)
    records, table = paths
    progress, csv_content = records.read_bytes(), table.read_bytes()
    actual = hashlib.sha256(progress).hexdigest(), hashlib.sha256(csv_content).hexdigest()
    if digests is not None and digests != actual:
        raise ValueError("measurement snapshot SHA-256 mismatch")
    state = json.loads(progress)
    if not isinstance(state, dict):
        raise ValueError("Measurement progress must be a JSON object")
    return MeasurementSnapshot(
        workspace, records, table, *actual,
        state, list(csv.DictReader(StringIO(csv_content.decode("utf-8-sig"), newline=""))),
    )


def read_snapshot(package: dict, *, digests: tuple[str, str] | None = None) -> MeasurementSnapshot:
    snapshot = _snapshot_files(package, digests)
    validation = validate_viewer_state(snapshot.workspace, snapshot.progress)
    if not validation["passed"]:
        raise ValueError(f"measurement package validation failed: {validation.get('failures')}")
    return snapshot


def verify_snapshot_rows(rows: list[dict], records: list[dict], slots=None) -> dict:
    """Map current CSV rows to their saved records and verify their rounded values."""
    measured = {(record["item_id"], record["side"]): record for record in records}
    if len(measured) != len(records):
        raise ValueError("Snapshot contains duplicate record identities")
    found = {}
    for row in rows:
        try:
            key = (row["量体项ID"], row["侧别"])
            value = float(row["当前cm"]) if row["当前cm"] else None
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Current measurement CSV contains an invalid value") from error
        if key in found:
            raise ValueError(f"Current measurement CSV contains a duplicate row: {key}")
        record = measured.pop(key, None)
        if record is None and value is not None:
            raise ValueError(f"unmeasured CSV row has a value: {key}")
        if record is not None and (value is None or not abs(value - record["value_mm"] / 10) <= 0.0505):
            raise ValueError(f"current measurement CSV does not match SurfaceTapeRecord {key}")
        found[key] = record
    if measured or (slots is not None and found.keys() != slots):
        raise ValueError("current measurement CSV does not match expected item and side identities")
    return found


def _validate_height(height_cm: float | None) -> None:
    if height_cm is not None and (
        isinstance(height_cm, bool) or not isfinite(height_cm) or height_cm <= 0
    ):
        raise ValueError("Height cm must be positive and finite")


def _source_display(body_path: Path, video_path: str | None, height_cm: float, identity: dict) -> dict:
    source = Path(video_path) if isinstance(video_path, str) and video_path.strip() else None
    path = source or body_path
    parent_name = source.parent.name if source else ""
    subject_name, device_model = identity.get("subject_name"), identity.get("device_model")
    subject = subject_name.strip() if isinstance(subject_name, str) and subject_name.strip() else path.stem
    group = parent_name if parent_name.startswith("from-") else ""
    if isinstance(device_model, str) and device_model.strip():
        group = device_model.strip()
    return {
        "subject": subject,
        "group": group,
        "source_file": path.name,
        "height_cm": float(height_cm),
        "file_stem": _safe_display_file_stem(f"{subject}-{float(height_cm):.0f}cm"),
    }


def _reference_context(
    source_identity: dict[str, object] | None,
    reference_measurements: list[dict[str, object]] | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    identity = dict(source_identity or {})
    references = [dict(item) for item in (reference_measurements or [])]
    if not identity:
        if references:
            raise ValueError("reference_measurements require source_identity")
        return {}, []
    for key in ("subject_id", "session_id"):
        if not isinstance(identity.get(key), str) or not str(identity[key]).strip():
            raise ValueError(f"source_identity.{key} is required")
    seen: set[tuple[object, object]] = set()
    for reference in references:
        if reference.get("session_id") != identity["session_id"]:
            raise ValueError("reference_measurement session_id does not match source_identity")
        key = (reference.get("metric_id"), reference.get("side"))
        if key in seen:
            raise ValueError(f"duplicate reference_measurement: {key}")
        seen.add(key)
    return identity, references


def _mesh_manifest(mesh: dict, artifact: str, selected_angle: float) -> dict:
    def scalar(key):
        return np.asarray(mesh[key]).item()

    def ref(key):
        return {"artifact": artifact, "key": key}

    mesh_id = str(scalar("mesh_id"))
    schema_id = str(scalar("joint_schema_id"))
    joint_source = {
        "units": str(scalar("units")),
        "coordinate_system": str(scalar("coordinate_system")),
        "source": {
            "kind": "body_model_forward",
            "track_id": int(scalar("track_id")),
            "frame_id": None,
            "producer": dict(BODY_MODEL_ASSET_IDENTITY),
        },
        "uncertainty": None,
    }
    tpose, *aposes = packed_pose_presets(mesh, selected_angle)
    payloads = {}
    aliases_by_stance = {stance: [] for stance in LOWER_BODY_POSE_PRESETS}
    for preset in (tpose, *aposes):
        base_alias = pose_key_stem(preset.vertex_key)
        for stance in LOWER_BODY_POSE_PRESETS:
            key = lower_body_vertex_key(preset.vertex_key, stance)
            stem = pose_key_stem(key)
            alias = base_alias if stance == "natural_standing" else f"{base_alias}__{stance}"
            aliases_by_stance[stance].append(alias)
            pose_keys = {name: f"{stem}_{name}" for name in (
                "joints", "pose_rotvecs", "body_origin", "body_axes", "pose_id", "lower_body_pose_id",
            )}
            if missing := ({key, *pose_keys.values()} - mesh.keys()):
                raise RuntimeError(f"measurement_meshes.npz is missing PosePayload arrays: {sorted(missing)}")
            payloads[alias] = {
                "schema_version": 1,
                "mesh_id": mesh_id,
                "pose_id": str(scalar(pose_keys["pose_id"])),
                "lower_body_pose_id": str(scalar(pose_keys["lower_body_pose_id"])),
                "vertices": ref(key),
                "joints": {
                    "schema_version": 1, "schema_id": schema_id,
                    "points": ref(pose_keys["joints"]),
                    **joint_source,
                },
                "pose_rotvecs": ref(pose_keys["pose_rotvecs"]),
                "body_frame": {
                    "origin": ref(pose_keys["body_origin"]),
                    "axes": ref(pose_keys["body_axes"]),
                    "axis_order": ["right", "up", "front"],
                    "axis_convention": "front=cross(up,right)",
                },
                **({"arm_down_degrees": preset.arm_down_degrees} if preset is not tpose else {}),
            }

    if scalar("body_segmentation_id") != BODY_SEGMENTATION_ID:
        raise RuntimeError("Unexpected body segmentation ID in measurement mesh")
    if scalar("measurement_landmark_set_id") != MEASUREMENT_LANDMARK_SET_ID:
        raise RuntimeError("Unexpected measurement landmark set ID in measurement mesh")
    validation = {
        "natural_standing": "hip_knee_ankle_rotvecs_zero_and_feet_separated",
        "legs_together": {"foot_surface_gap_mm": float(scalar("legs_together_foot_surface_gap_mm")),
                          "range_mm": [0.0, 5.0]},
        "legs_shoulder_width": {"ankle_spacing_mm": float(scalar("legs_shoulder_width_ankle_spacing_mm")),
                                "target_mm": float(scalar("legs_shoulder_width_target_mm")),
                                "target_basis": str(scalar("legs_shoulder_width_target_basis")),
                                "tolerance_mm": 5.0},
    }
    return {
        "joint_schemas": {schema_id: {**SMPL_24_JOINT_SCHEMA, "sides": SMPL_24_JOINT_SIDES}},
        "pose_payloads": payloads,
        "body_segmentations": {BODY_SEGMENTATION_ID: {
            "schema_version": 1, "body_segmentation_id": BODY_SEGMENTATION_ID, "mesh_id": mesh_id,
            "segment_ids": list(BODY_SEGMENT_IDS), "face_body_segment": ref("face_body_segment"),
        }},
        "measurement_landmark_sets": {MEASUREMENT_LANDMARK_SET_ID: {
            "schema_version": 1, "landmark_set_id": MEASUREMENT_LANDMARK_SET_ID, "mesh_id": mesh_id,
            "landmarks": [{"id": key, "side": side, "label": label} for key, side, label in MEASUREMENT_LANDMARKS],
            "landmark_kind": "SurfaceHit",
            "surface_hits": {"face_indices": ref("measurement_landmark_face_indices"),
                             "barycentric": ref("measurement_landmark_barycentric")},
        }},
        "pose_generation_status": {
            stance: {"available": True, "compatible_pose_payloads": aliases, "validation": validation[stance]}
            for stance, aliases in aliases_by_stance.items()
        },
    }


def _extract_source_reference_frame(
    video_path: str | None, output_path: Path, *, video_sha256: str | None,
) -> dict:
    source = str(video_path or "").strip()
    summary: dict[str, object] = {
        "available": False,
        "source_file": Path(source).name if source else "",
        "reason": "这份工作台从已保存的人台生成。",
    }
    if not source:
        return summary
    path = Path(source)
    if not path.is_file():
        summary["reason"] = "原视频文件当前不可访问。"
        return summary
    if video_sha256:
        try:
            digest = _file_sha256(path)
        except OSError:
            summary["reason"] = "原视频文件当前不可访问。"
            return summary
        if digest != video_sha256:
            summary["reason"] = "原视频和保存的 BodyPack 来源不一致。"
            return summary
    try:
        import cv2
        from video_io import VideoDecodeError, VideoReader
    except ModuleNotFoundError:
        summary["reason"] = "当前环境缺少视频读取组件，暂未生成原视频参考图。"
        return summary
    try:
        with VideoReader(str(path), device="cpu") as reader:
            frame_index = reader.frame_count // 2 if reader.frame_count > 2 else 0
            try:
                frame = reader.get_frame(frame_index)
            except VideoDecodeError:
                if not frame_index:
                    raise
                frame_index, frame = 0, reader.get_frame(0)
    except VideoDecodeError:
        return {**summary, "reason": "原视频没有读到可用画面，工作台仍可操作人台。"}
    bgr = cv2.cvtColor(frame.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(output_path.suffix or ".jpg", bgr)
    if not ok:
        return {**summary, "reason": "原视频参考图写入失败，工作台仍可操作人台。"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded.tobytes())
    return {**summary, "available": True, "artifact": output_path.name, "frame_index": frame_index, "reason": ""}


def build_workbench(
    body: MotionPackBody,
    output_dir: str | Path,
    *,
    height_cm: float | None = None,
    track_id: int | None = None,
    apose_arm_down_degrees: float | None = None,
    source_identity=None,
    reference_measurements=None,
    public_item_ids: list[str] | None = None,
    include_snapshot: bool = False,
) -> dict:
    """Publish a workbench. A missing stature uses the provisional preview scale."""
    _validate_height(height_cm)
    height_is_estimate = height_cm is None
    preview_height = DEFAULT_HEIGHT_CM if height_cm is None else height_cm
    output = Path(output_dir).absolute()
    source_identity, reference_measurements = _reference_context(source_identity, reference_measurements)
    expected_source = source_identity.get("input_sha256")
    if expected_source and expected_source != body.source.video_sha256:
        raise ValueError("BodyPack video does not match the declared input SHA-256")
    selected_angle = DEFAULT_APOSE_PRESET.arm_down_degrees if apose_arm_down_degrees is None else float(apose_arm_down_degrees)
    with new_output_directory(output) as staging:
        video_path = body.source.video_path
        source_display = _source_display(body.path, video_path, preview_height, source_identity)
        artifacts = {
            role: f"{SOURCE_REVIEW_DIR_NAME}/{name}"
            for role, name in (
                ("meshes", "measurement_meshes.npz"),
                ("manifest", "manifest.json"),
                ("source_reference_frame", "source_reference_frame.jpg"),
            )
        }
        (staging / SOURCE_REVIEW_DIR_NAME).mkdir()
        mesh_path = staging / artifacts["meshes"]
        manifest_path = staging / artifacts["manifest"]
        mesh = build_measurement_mesh(body, preview_height, track_id=track_id, arm_down_degrees=selected_angle)
        np.savez_compressed(mesh_path, **mesh)
        source_reference = _extract_source_reference_frame(
            video_path, staging / artifacts["source_reference_frame"], video_sha256=body.source.video_sha256,
        )
        if not source_reference["available"]:
            del artifacts["source_reference_frame"]
        manifest = {
            "contract_version": VIEWER_CONTRACT_VERSION,
            "workbench_id": uuid4().hex,
            "measurement_definitions": read_definition_catalog(MEASUREMENT_DEFINITIONS_PATH),
            **({"public_item_ids": public_item_ids} if public_item_ids is not None else {}),
            "source_video_sha256": body.source.video_sha256,
            "height_is_estimate": height_is_estimate,
            "source_display": source_display,
            "source_identity": source_identity,
            "reference_measurements": reference_measurements,
            "height_cm": preview_height,
            "poses": {"apose": {
                "track_id": mesh["track_id"],
                "pose_preset_id": DEFAULT_APOSE_PRESET.id,
                "apose_arm_down_degrees": selected_angle,
                "target_height_cm": float(preview_height),
                "source_file": source_display["source_file"],
            }},
            **_mesh_manifest(mesh, artifacts["meshes"], selected_angle),
            "artifacts": artifacts,
        }
        manifest["readback"] = {
            **{key: {"artifact": artifacts["meshes"], "key": key} for key in (
                "shape_betas", "selected_apose_arm_down_degrees", "selected_apose_pose_rotvecs",
            )},
            "source_reference": source_reference,
        }
        manifest["readback"]["selected_apose_arm_down_degrees"]["value"] = selected_angle
        package = ViewerPackage(staging.resolve(), mesh_path, manifest)
        payload = _project_viewer_payload(package, mesh)
        _ensure_viewer_assets(staging)
        (staging / "viewer.html").write_text(render_viewer_html(payload), encoding="utf-8")
        artifacts["viewer_html"] = "viewer.html"
        artifacts.update({name: f"viewer_assets/{name}" for name in VIEWER_ASSET_RUNTIME_FILES})
        if include_snapshot:
            artifacts.update(_write_snapshot(payload, staging, staging / "measurements"))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "package_dir": str(output),
        "manifest_path": str(output / artifacts["manifest"]),
        "artifacts": artifacts,
        "manifest": manifest,
    }


def write_snapshot(
    workspace: Path,
    directory: Path,
    *,
    progress: dict | None = None,
) -> dict[str, str]:
    """Export the frozen workbench's current table and progress with its measurement engine."""
    return _write_snapshot(build_viewer_payload(workspace), workspace, directory, progress)


def _replay_snapshot(workspace: Path, directory: Path, progress: dict | None) -> MeasurementSnapshot:
    return _snapshot_files({"package_dir": workspace, "artifacts": write_snapshot(workspace, directory, progress=progress)})


def _write_snapshot(
    payload: dict, workspace: Path, directory: Path, progress: dict | None = None,
) -> dict[str, str]:
    workspace, directory = workspace.resolve(), directory.resolve()
    if not directory.is_relative_to(workspace):
        raise ValueError("Snapshot directory must stay inside the workbench")
    export = run_measurement_engine(payload=payload, progress=progress)
    state = export["progress"]
    if "public_item_ids" in payload:
        for record in state["measurements"]:
            record["path_sha256"] = surface_path_digest(record)

    files = {
        RECORDS_ARTIFACT: ("工作进度.bmwork", json.dumps(state, ensure_ascii=False).encode("utf-8")),
        "csv": ("当前量体表.csv", export["csv"].encode("utf-8")),
    }
    with new_output_directory(directory) as staging:
        for name, data in files.values():
            staged = staging / name
            staged.write_bytes(data)
            if progress is not None and staged.read_bytes() != data:
                raise OSError(f"Snapshot write differs from verified data: {name}")
    return {key: (directory / name).relative_to(workspace).as_posix() for key, (name, _) in files.items()}
