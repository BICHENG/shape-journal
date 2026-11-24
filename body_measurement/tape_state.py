"""Record schema and versioned measurement definition binding."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any

SURFACE_TAPE_SCHEMA_VERSION = 1
SURFACE_PATH_METHOD = "surface_path"
MESH_SECTION_METHOD = "mesh_section"
STRAIGHT_METHOD = "straight"
SURFACE_TAPE_METHODS = {SURFACE_PATH_METHOD, MESH_SECTION_METHOD, STRAIGHT_METHOD}
SURFACE_TAPE_SIDES = {"left", "right", "center"}
SURFACE_TAPE_UNITS = "mm"
VALUE_TOLERANCE_MM = 1e-3
VIEWER_STATE_KIND = "body_measurement_viewer_state"

LEGACY_RECORD_FIELDS = {
    "closed",
    "contact_faces",
    "contact_point_count",
    "contact_points",
    "contact_weights",
    "measurement_kind",
    "measurement_method",
    "mesh_query",
    "path_kind",
    "physical_solver",
    "point_count",
    "points",
    "slice_point_count",
    "slice_points",
    "slice_y",
    "surface_path_complete",
    "surface_point_count",
    "surface_points",
    "surface_value_mm",
    "surface_vertex_delta_mm",
    "unit",
    "value",
    "value_source",
    "vertex_indices",
    "vertex_path_value_mm",
}


def surface_path_digest(record: dict) -> str:
    path = {name: record[name] for name in ("path_faces", "path_weights")}
    return sha256(json.dumps(path, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run_measurement_engine(**request) -> dict:
    engine = Path(__file__).with_name("viewer_assets") / "src" / "measurement_snapshot.js"
    completed = subprocess.run(
        ["node", str(engine)], input=json.dumps(request, ensure_ascii=False),
        encoding="utf-8", capture_output=True, timeout=120,
    )
    if completed.returncode:
        error = completed.stderr.strip() or "Measurement engine failed"
        if request.get("validate_only") or request.get("progress") is not None or "landmark_section" in request:
            raise ValueError(error)
        raise RuntimeError(error)
    return json.loads(completed.stdout)


def viewer_state_schema() -> dict[str, Any]:
    return {
        "state_kind": VIEWER_STATE_KIND,
        "surface_tape_record": {
            "schema_version": SURFACE_TAPE_SCHEMA_VERSION,
            "retired_fields": sorted(LEGACY_RECORD_FIELDS),
            "required_fields": [
                "mesh_id",
                "pose_id",
                "item_id",
                "metric_id",
                "definition_id",
                "version",
                "side",
                "method",
                "path_faces",
                "path_weights",
                "units",
                "value_mm",
            ],
            "methods": sorted(SURFACE_TAPE_METHODS),
            "sides": sorted(SURFACE_TAPE_SIDES),
            "units": SURFACE_TAPE_UNITS,
            "value_tolerance_mm": VALUE_TOLERANCE_MM,
        },
    }
