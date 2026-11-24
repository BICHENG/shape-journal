"""Static viewer for body measurement workspace outputs."""

from __future__ import annotations

import hashlib
import json
import shutil
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from body_measurement.poses import (
    DEFAULT_APOSE_PRESET,
    LOWER_BODY_POSE_PRESETS,
    lower_body_vertex_key,
    packed_pose_presets,
)
from body_measurement.tape_state import (
    VIEWER_STATE_KIND,
    run_measurement_engine,
    viewer_state_schema,
)

VIEWER_CONTRACT_VERSION = 2
WORKBENCH_TAPE_SCHEMA_ID = "lower-body-8-v1"
VIEWER_TEMPLATE_PATH = Path(__file__).with_name("viewer_template.html")
VIEWER_ASSETS_SOURCE_DIR = Path(__file__).with_name("viewer_assets")
MEASUREMENT_DEFINITIONS_PATH = Path(__file__).with_name("specs") / "measurement_definitions.json"
VIEWER_RUNTIME_SOURCES = ("viewer_runtime.js", "measurement_records.js", "surface_measurement.js")
VIEWER_ASSET_RUNTIME_FILES = (
    "tech_stack.json",
    "VERSIONS.md",
    "LICENSES.md",
    "viewer_runtime.bundle.js",
)
SOURCE_REVIEW_DIR_NAME = "来源复查"


@dataclass(frozen=True)
class ViewerPackage:
    package_dir: Path
    mesh_path: Path
    manifest: dict[str, Any]


def _safe_display_file_stem(value: str) -> str:
    cleaned = "".join("_" if char in r'\/:*?"<>|' or char.isspace() else char for char in value)
    cleaned = cleaned.strip("._")
    return cleaned or "量体结果"


def _source_display(package: ViewerPackage, selected_angle: float) -> dict[str, Any]:
    manifest = package.manifest
    display = manifest.get("source_display") if isinstance(manifest.get("source_display"), dict) else {}
    subject = str(display.get("subject") or package.package_dir.parent.name)
    group = str(display.get("group") or "")
    height = manifest.get("height_cm")
    height_label = f"{float(height):.1f} cm" if isinstance(height, (int, float)) and not isinstance(height, bool) else ""
    pose_label = f"A字站姿 {float(selected_angle):g}度"
    file_stem = str(display.get("file_stem") or "")
    if not file_stem:
        stem_parts = [subject]
        if isinstance(height, (int, float)) and not isinstance(height, bool):
            stem_parts.append(f"{float(height):.0f}cm")
        file_stem = "-".join(part for part in stem_parts if part)
    return {
        "subject": subject,
        "group": group,
        "height_label": height_label,
        "pose_label": pose_label,
        "file_stem": _safe_display_file_stem(file_stem),
    }


def load_viewer_package(package_dir: str | Path) -> ViewerPackage:
    root = Path(package_dir).resolve()
    if root.is_dir() and root.name == SOURCE_REVIEW_DIR_NAME and (root / "manifest.json").exists():
        root = root.parent
    manifest = json.loads(_find_manifest_path(root).read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    mesh_path = _package_file(root, artifacts.get("meshes", "measurement_meshes.npz"))
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)
    if "pose_preview" in artifacts or manifest.get("contract_version", 1) == 1:
        preview = _package_file(root, artifacts.get("pose_preview", "pose_model_preview.png"))
        if not preview.is_file():
            raise FileNotFoundError(preview)
    return ViewerPackage(root, mesh_path, manifest)


def _package_file(root: Path, relative: str) -> Path:
    path = Path(relative)
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root):
        raise ValueError(f"Artifact path must stay inside the workbench: {relative}")
    return resolved


def _find_manifest_path(root: Path) -> Path:
    candidates = (
        root / "manifest.json",
        root / SOURCE_REVIEW_DIR_NAME / "manifest.json",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def build_viewer_payload(package_dir: str | Path, *, precision: int = 3) -> dict[str, Any]:
    package = load_viewer_package(package_dir)
    with np.load(package.mesh_path) as mesh:
        return _project_viewer_payload(package, mesh, precision=precision)


def _project_viewer_payload(
    package: ViewerPackage, mesh: Mapping[str, Any], *, precision: int = 3,
) -> dict[str, Any]:
    manifest = package.manifest
    artifacts = manifest.get("artifacts", {})
    contract_version = manifest.get("contract_version", 1)
    if type(contract_version) is not int or contract_version not in (1, VIEWER_CONTRACT_VERSION):
        raise ValueError(f"Unsupported workbench contract {contract_version!r}")
    workbench_id = manifest.get("workbench_id")
    if contract_version == VIEWER_CONTRACT_VERSION and not (
        isinstance(workbench_id, str) and re.fullmatch(r"[a-f0-9]{32}", workbench_id)
    ):
        raise ValueError("Workbench manifest is missing its workbench_id")
    measurement_definitions = _package_definitions(package, contract_version)
    public_item_ids = manifest.get("public_item_ids")
    if "public_item_ids" in manifest:
        workbook_items = {
            definition["item_id"] for definition in measurement_definitions["definitions"]
            if definition.get("preset_schema_id") == WORKBENCH_TAPE_SCHEMA_ID
        }
        if not isinstance(public_item_ids, list) or any(
            not isinstance(item, str) or item not in workbook_items for item in public_item_ids
        ) or len(set(public_item_ids)) != len(public_item_ids):
            raise ValueError("public_item_ids must contain unique workbook item IDs")
    source_identity = manifest.get("source_identity")
    references = manifest.get("reference_measurements")
    calibration = manifest.get("personal_calibration")
    source_identity = source_identity if isinstance(source_identity, dict) else {}
    references = references if isinstance(references, list) else []
    calibration = calibration if isinstance(calibration, dict) else None
    required = {"faces", "tpose_vertices", "apose_vertices", "mesh_id", "body_segmentation_id", "face_body_segment"}
    if missing := sorted(required - mesh.keys()):
        raise RuntimeError(f"{package.mesh_path} is missing viewer arrays: {missing}")
    faces = np.asarray(mesh["faces"], dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(f"{package.mesh_path} faces must have shape [F,3], got {faces.shape}")
    mesh_id = str(np.asarray(mesh["mesh_id"]).item())
    mesh_artifact = artifacts.get("meshes", "measurement_meshes.npz")
    pose_payloads = manifest.get("pose_payloads")
    if not isinstance(pose_payloads, dict) or not pose_payloads:
        raise RuntimeError("manifest is missing pose_payloads")
    segmentation_id = str(np.asarray(mesh["body_segmentation_id"]).item())
    segmentation = manifest.get("body_segmentations", {}).get(segmentation_id)
    for owner, record in [*pose_payloads.items(), (segmentation_id, segmentation)]:
        if not isinstance(record, dict) or record.get("mesh_id") != mesh_id:
            raise RuntimeError(f"{owner} does not match measurement mesh {mesh_id!r}")
    segment_ids = segmentation.get("segment_ids")
    indices = np.asarray(mesh["face_body_segment"], dtype=np.int64)
    if not isinstance(segment_ids, list) or not segment_ids or (
        indices.shape != (len(faces),) or np.any(indices < 0) or np.any(indices >= len(segment_ids))
    ):
        raise RuntimeError("face_body_segment must match faces and reference manifest segment_ids")
    face_body_segment = [segment_ids[index] for index in indices]
    selected_angle = _selected_apose_arm_down_degrees(mesh, manifest)
    joint_schemas = manifest.get("joint_schemas", {})
    poses = _collect_poses(mesh, pose_payloads, mesh_artifact, joint_schemas, selected_angle, precision)
    landmarks = _measurement_landmark_sets(manifest, mesh, mesh_artifact, mesh_id, len(faces))

    source_display = _source_display(package, selected_angle)
    subject_id = str(source_identity.get("subject_id") or source_display["subject"])
    if calibration and calibration.get("subject_id") != subject_id:
        raise RuntimeError("personal_calibration subject_id does not match viewer subject")
    for entry in [*poses, *landmarks.values()]:
        entry["subject_id"] = subject_id
    apose_angles = sorted({pose["arm_down_degrees"] for pose in poses if pose["kind"] == "apose"})
    return {
        "contract_version": contract_version,
        **({"workbench_id": workbench_id} if workbench_id else {}),
        "viewer_state_schema": viewer_state_schema(),
        "height_is_estimate": bool(manifest.get("height_is_estimate")),
        "source_display": source_display,
        "source_identity": {
            **source_identity,
            "height_cm": manifest.get("height_cm"),
            "track_id": manifest.get("poses", {}).get("apose", {}).get("track_id"),
            "manifest_path": artifacts.get("manifest", "manifest.json"),
            "mesh_path": artifacts.get("meshes", "measurement_meshes.npz"),
            **({"preview_path": artifacts.get("pose_preview", "pose_model_preview.png")}
               if "pose_preview" in artifacts or contract_version == 1 else {}),
        },
        "reference_measurements": references,
        **({"personal_calibration": calibration} if calibration else {}),
        "source_reference": _source_reference_payload(package),
        "measurement_definitions": measurement_definitions,
        **({"public_item_ids": public_item_ids} if public_item_ids is not None else {}),
        "joint_schemas": joint_schemas,
        "measurement_landmark_sets": landmarks,
        "mesh": {
            "mesh_id": mesh_id,
            "unit": "mm",
            "coordinate_system": "x,y,z with body height on y axis",
            "faces_count": int(len(faces)),
            "faces": faces.astype(int).tolist(),
            "face_body_segment": face_body_segment,
            "face_side": [
                "left" if segment.startswith("left_") else "right" if segment.startswith("right_") else "center"
                for segment in face_body_segment
            ],
        },
        "selected_apose_arm_down_degrees": selected_angle,
        "apose_angle_range": {
            "min": min(apose_angles) if apose_angles else selected_angle,
            "max": max(apose_angles) if apose_angles else selected_angle,
            "values": apose_angles,
        },
        "poses": poses,
    }


def _referenced_array(mesh: Mapping[str, Any], reference: dict, artifact: str, dtype=None) -> np.ndarray:
    if not isinstance(reference, dict) or reference.get("artifact") != artifact or reference.get("key") not in mesh:
        raise RuntimeError("Invalid measurement mesh array reference")
    return np.asarray(mesh[reference["key"]], dtype=dtype)


def _measurement_landmark_sets(manifest, mesh, mesh_artifact, mesh_id, faces_count) -> dict:
    set_id = str(np.asarray(mesh["measurement_landmark_set_id"]).item())
    landmark_set = manifest.get("measurement_landmark_sets", {}).get(set_id)
    if not isinstance(landmark_set, dict) or landmark_set.get("mesh_id") != mesh_id:
        raise RuntimeError(f"measurement landmark set {set_id!r} does not match mesh {mesh_id!r}")
    landmarks = landmark_set.get("landmarks")
    refs = landmark_set.get("surface_hits")
    if not isinstance(landmarks, list) or not isinstance(refs, dict):
        raise RuntimeError(f"measurement landmark set {set_id!r} is incomplete")
    ids = [item.get("id") if isinstance(item, dict) else None for item in landmarks]
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise RuntimeError(f"measurement landmark set {set_id!r} has invalid landmark ids")
    face_indices = _referenced_array(mesh, refs.get("face_indices"), mesh_artifact, np.int64)
    weights = _referenced_array(mesh, refs.get("barycentric"), mesh_artifact, np.float64)
    if (
        face_indices.shape != (len(ids),)
        or weights.shape != (len(ids), 3)
        or np.any(face_indices < 0)
        or np.any(face_indices >= faces_count)
        or not np.all(np.isfinite(weights))
        or np.any(weights < -1e-9)
        or np.any(weights > 1 + 1e-9)
        or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-9)
    ):
        raise RuntimeError(f"measurement landmark set {set_id!r} contains invalid SurfaceHits")
    return {
        set_id: {
            **landmark_set,
            "hits": {
                landmark_id: {"face_index": int(face_index), "weights": barycentric.tolist()}
                for landmark_id, face_index, barycentric in zip(ids, face_indices, weights)
            },
        }
    }


def read_definition_catalog(path: Path) -> dict[str, Any]:
    return _validate_definition_catalog(json.loads(path.read_text(encoding="utf-8")))


def _validate_definition_catalog(catalog: Any) -> dict[str, Any]:
    if not isinstance(catalog, dict) or not isinstance(catalog.get("definitions"), list):
        raise ValueError("Workbench must contain a measurement definitions array")
    history = catalog.get("historical_definitions", [])
    if not isinstance(history, list):
        raise ValueError("historical_definitions must be an array")
    keys = set()
    active_items = set()
    for index, definition in enumerate(catalog["definitions"] + history):
        if not isinstance(definition, dict) or any(
            not isinstance(definition.get(field), str) or not definition[field]
            for field in ("item_id", "definition_id", "version")
        ):
            raise ValueError("Measurement definitions must have an item, definition ID and version")
        key = (definition["item_id"], definition["version"])
        if key in keys:
            raise ValueError(f"Duplicate measurement definition {key}")
        keys.add(key)
        if index < len(catalog["definitions"]):
            if definition["item_id"] in active_items:
                raise ValueError(f"Duplicate active measurement item {definition['item_id']}")
            active_items.add(definition["item_id"])
    return catalog


def read_embedded_payload(page: Path) -> dict[str, Any]:
    html = page.read_text(encoding="utf-8-sig")
    match = re.search(r'<script id="viewer-payload" type="application/json">(.*?)</script>', html, re.S)
    if not match:
        raise ValueError("Workbench page is missing its embedded payload")
    return json.loads(match[1])


def _package_definitions(package: ViewerPackage, version: int) -> dict[str, Any]:
    if version == VIEWER_CONTRACT_VERSION:
        return _validate_definition_catalog(package.manifest.get("measurement_definitions"))

    # The earliest portable workbenches kept their definition snapshot in the page.
    embedded = read_embedded_payload(package.package_dir / "viewer.html")
    catalog = _validate_definition_catalog(embedded.get("measurement_definitions"))
    if "workbench_tape_definitions" not in embedded:
        # Before the presets moved into JSON, the viewer used both pelvis and torso for iliac waist.
        catalog = {**catalog, "definitions": [
            {**item, "body_segments": ["pelvis", "torso"]}
            if item["item_id"] == "iliac_waist" and item["version"] == "v1"
            and item.get("body_segments") == ["pelvis"] else item
            for item in catalog["definitions"]
        ]}
    if "historical_definitions" in catalog:
        return catalog
    seen = {(item["item_id"], item["version"]) for item in catalog["definitions"]}
    history = read_definition_catalog(MEASUREMENT_DEFINITIONS_PATH).get("historical_definitions", [])
    return {**catalog, "historical_definitions": [
        item for item in history if (item["item_id"], item["version"]) not in seen
    ]}


def _source_reference_payload(package: ViewerPackage) -> dict[str, Any]:
    artifacts = package.manifest.get("artifacts", {})
    readback_reference = package.manifest.get("readback", {}).get("source_reference", {})
    if not isinstance(readback_reference, dict):
        readback_reference = {}
    frame_rel = artifacts.get("source_reference_frame")
    frame_path = _package_file(package.package_dir, frame_rel) if isinstance(frame_rel, str) else None
    frame_available = bool(
        readback_reference.get("available")
        and isinstance(frame_rel, str)
        and frame_path is not None
        and frame_path.exists()
    )
    pose_preview = artifacts.get("pose_preview")
    if not pose_preview and (package.package_dir / "pose_model_preview.png").is_file():
        pose_preview = "pose_model_preview.png"
    return {
        **({"pose_preview": {"available": True, "image_path": pose_preview, "label": "模型姿态参考"}}
           if pose_preview else {}),
        "source_frame": {
            "available": frame_available,
            "image_path": frame_rel if frame_available else None,
            "reason": ""
            if frame_available
            else readback_reference.get("reason")
            or "原视频参考图未生成，当前工作台仍可操作人台。",
            "label": "原视频参考",
        },
    }


def viewer_source_stamp(source_dir: Path = VIEWER_ASSETS_SOURCE_DIR / "src") -> str:
    digest = hashlib.sha256()
    for name in VIEWER_RUNTIME_SOURCES:
        digest.update(name.encode("ascii") + b"\0")
        digest.update((source_dir / name).read_text(encoding="utf-8").encode("utf-8"))
        digest.update(b"\0")
    return "// viewer-source-sha256: " + digest.hexdigest()


def _ensure_viewer_assets(package_dir: Path) -> None:
    source = VIEWER_ASSETS_SOURCE_DIR.resolve()
    target = (package_dir / "viewer_assets").resolve()
    with (source / "viewer_runtime.bundle.js").open(encoding="utf-8-sig") as bundle:
        if bundle.readline().strip() != viewer_source_stamp():
            raise RuntimeError("Viewer bundle is out of date. Run body_measurement.scripts.build_viewer_runtime.")
    if source == target:
        return
    target.mkdir(parents=True, exist_ok=True)
    for filename in VIEWER_ASSET_RUNTIME_FILES:
        source_file = source / filename
        if not source_file.exists():
            raise FileNotFoundError(source_file)
        shutil.copy2(source_file, target / filename)


def validate_viewer_state(package_dir: str | Path, state: dict[str, Any]) -> dict[str, Any]:
    return validate_work_progress(build_viewer_payload(package_dir), state)


def validate_work_progress(payload: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_measurement_engine(payload=payload, progress=state, validate_only=True)
    except ValueError as error:
        return {"passed": False, "checks": [], "failures": [str(error)]}


def render_viewer_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = VIEWER_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.replace("__VIEWER_PAYLOAD__", payload_json)


def _selected_apose_arm_down_degrees(mesh: Mapping[str, Any], manifest: dict[str, Any]) -> float:
    if "selected_apose_arm_down_degrees" in mesh:
        return float(np.asarray(mesh["selected_apose_arm_down_degrees"]).reshape(-1)[0])
    if "apose_arm_down_degrees" in mesh:
        values = [float(value) for value in np.asarray(mesh["apose_arm_down_degrees"]).reshape(-1).tolist()]
        if values:
            target = DEFAULT_APOSE_PRESET.arm_down_degrees
            return min(values, key=lambda value: abs(value - target))
    pose = manifest.get("poses", {}).get("apose", {}) if isinstance(manifest.get("poses"), dict) else {}
    value = pose.get("apose_arm_down_degrees") if isinstance(pose, dict) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    preset_id = str(pose.get("pose_preset_id") or "") if isinstance(pose, dict) else ""
    match = re.search(r"(\d+(?:\.\d+)?)deg", preset_id)
    if match:
        return float(match.group(1))
    raise RuntimeError("measurement workspace is missing A-pose arm angle")


def _collect_poses(mesh, pose_payloads, mesh_artifact, joint_schemas, selected_angle, precision) -> list[dict]:
    tpose, selected, *presets = packed_pose_presets(mesh, selected_angle)
    by_angle = {}
    for preset in sorted([selected, *presets], key=lambda item: (item.arm_down_degrees, item is not selected, item.id)):
        angle = round(preset.arm_down_degrees, 6)
        if angle not in by_angle or preset is selected:
            by_angle[angle] = preset

    by_vertex_key = {}
    for ref in pose_payloads.values():
        vertices = ref.get("vertices")
        if isinstance(vertices, dict) and isinstance(vertices.get("key"), str):
            by_vertex_key.setdefault(vertices["key"], []).append(ref)

    poses = []
    for preset in (tpose, *by_angle.values()):
        angle = preset.arm_down_degrees
        alias = "tpose" if preset is tpose else "selected_apose" if preset is selected else preset.id
        kind = "tpose" if preset is tpose else "apose"
        label = "T-pose" if preset is tpose else f"A-pose {angle:g} deg"
        for lower in LOWER_BODY_POSE_PRESETS:
            key = lower_body_vertex_key(preset.vertex_key, lower)
            matches = by_vertex_key.get(key, [])
            if len(matches) != 1:
                raise RuntimeError(f"manifest must contain one PosePayload for {key}")
            ref = matches[0]
            try:
                vertices = _referenced_array(mesh, ref["vertices"], mesh_artifact)
                frame, joints = ref["body_frame"], ref["joints"]
                axes = _referenced_array(mesh, frame["axes"], mesh_artifact, np.float64)
                joint_points = _referenced_array(mesh, joints["points"], mesh_artifact, np.float64)
                order = frame["axis_order"]
                if joints["schema_id"] not in joint_schemas or not isinstance(order, list) or order.count("up") != 1:
                    raise ValueError("Invalid axis order or joint schema")
            except (KeyError, RuntimeError, TypeError, ValueError) as error:
                raise RuntimeError(f"PosePayload for {key} has an invalid body-frame or joints reference") from error
            up = axes[order.index("up")] if axes.shape == (3, 3) else np.asarray([])
            if up.shape != (3,) or not np.all(np.isfinite(up)) or not np.isclose(np.linalg.norm(up), 1.0, atol=1e-5):
                raise RuntimeError(f"PosePayload for {key} has an invalid body up axis")
            if joint_points.ndim != 2 or joint_points.shape[1:] != (3,) or not np.all(np.isfinite(joint_points)):
                raise RuntimeError(f"PosePayload for {key} has invalid joint points")
            if not all(isinstance(ref.get(field), str) and ref[field] for field in ("pose_id", "lower_body_pose_id")):
                raise RuntimeError(f"PosePayload for {key} is missing pose identity")
            natural = lower == "natural_standing"
            poses.append({
                "id": alias if natural else f"{alias}__{lower}",
                "label": label if natural else f"{label} · {lower}",
                "kind": kind,
                "vertex_key": key,
                "arm_down_degrees": angle,
                "visual_proxy": False,
                "vertices": np.round(vertices, precision).tolist(),
                "height_cm": round(float(np.ptp(vertices[:, 1])) / 10.0, 6),
                "mesh_id": ref["mesh_id"],
                "pose_id": ref["pose_id"],
                "lower_body_pose_id": ref["lower_body_pose_id"],
                "joints": {**joints, "points": joint_points.tolist()},
                "body_up": np.round(up, 6).tolist(),
            })
    return poses
