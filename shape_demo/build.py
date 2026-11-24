"""Reconstruct a video and export its SMPL mesh, demo stages, and offline preview."""

import argparse
import base64
from collections import Counter
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from body_measurement.mesh import (
    BODY_SEGMENT_IDS,
    _load_smplfitter_body_model,
    generate_face_body_segment,
    generate_hip_crest_hits,
    generate_pose_model,
    make_body_frame,
)
from body_measurement.poses import make_apose_rotvecs
from body_measurement.scripts.build_viewer_runtime import _find_npm
from body_measurement.shape_tracking import ShapeObservation, estimate_visit
from body_measurement.viewer import MEASUREMENT_DEFINITIONS_PATH, read_definition_catalog
from core.motionpack import load_artifacts, load_body, reconstruct_subject
from core.motionpack.io import _file_sha256
from core.motionpack.model import SMPL_24_JOINT_SCHEMA
from core.output_directory import new_output_directory
from core.smpl_assets import current_body_model_asset_identity
from shape_demo.alignment import source_alignment, source_motion

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "shape_demo"
STACK = ROOT / "body_measurement/viewer_assets/tech_stack.json"
DEMO_RECORDS = (
    ("start", "Start", 0.0),
    ("early", "Early", 0.15625),
    ("middle", "Middle", 0.5),
    ("late", "Late", 0.84375),
    ("finish", "End", 1.0),
)
PREVIEW_POSE = make_apose_rotvecs()
DEFAULT_VIDEO = ROOT / "test_data/bc1.mp4"
DEFAULT_PACK = ROOT / "output/shape-tracer-bc1-capture"


def limb_delta(vertices, joints, skin, side, start, end, influences, strength, hold_height=False):
    line = joints[end] - joints[start]
    along = (vertices - joints[start]) @ line / (line @ line)
    center = joints[start] + along[:, None] * line
    envelope = np.maximum(0, np.sin(np.pi * np.clip((along - 0.07) / 0.87, 0, 1))) ** 1.6
    weight = np.clip((skin[:, influences].sum(axis=1) - 0.20) / 0.70, 0, 1)
    delta = (vertices - center) * (strength * envelope * weight)[:, None]
    if hold_height:
        delta[:, 1] = 0
        # Grow toward the outside where the two thighs share a narrow gap.
        delta[:, 0] = np.where(side * delta[:, 0] < 0, 0.45 * delta[:, 0], delta[:, 0])
    return delta


def write_stl(path, triangles, normals):
    facets = np.zeros(len(triangles), dtype=[
        ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2"),
    ])
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError(f"Invalid mesh triangles for {path.name}")
    facets["normal"] = normals / lengths[:, None]
    facets["vertices"] = triangles
    header = b"Shape Journal | SMPL | mm | Y up".ljust(80, b"\0")
    path.write_bytes(header + struct.pack("<I", len(facets)) + facets.tobytes())


def surface_coverage(faces, vertex_count):
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    edges, uses = np.unique(edges, axis=0, return_counts=True)
    border = np.zeros(vertex_count, dtype=bool)
    border[edges[uses == 1].ravel()] = True
    coverage = np.full(vertex_count, 255, dtype=np.uint8)
    coverage[edges[border[edges].any(axis=1)].ravel()] = 144
    coverage[border] = 0
    return coverage


def arm_faces(centers, segments, joints):
    selected = np.isin(segments, ("left_upper_arm", "right_upper_arm"))
    for side, collar, shoulder, elbow, wrist in (
        ("left", 13, 16, 18, 20), ("right", 14, 17, 19, 21),
    ):
        shoulder_clearance = .53 * np.linalg.norm(joints[elbow] - joints[shoulder])
        sign = np.sign(joints[shoulder, 0] - joints[0, 0])
        lateral = sign * (centers[:, 0] - joints[[collar, shoulder], 0].mean()) > 0
        near_shoulder = np.linalg.norm(centers - joints[shoulder], axis=1) <= shoulder_clearance
        selected |= (segments == "torso") & lateral & near_shoulder
        forearm = joints[wrist] - joints[elbow]
        selected |= (segments == f"{side}_forearm") & ((centers - joints[elbow]) @ forearm <= .29 * (forearm @ forearm))
    return selected


def region_views(vertices, faces, joints):
    segments = np.asarray(BODY_SEGMENT_IDS)[generate_face_body_segment(faces)]
    centers = vertices[faces].mean(axis=1)
    face_y = centers[:, 1]
    # Keep the waist band between the upper pelvis and lower ribs in the reference body.
    lower = (joints[[1, 2], 1].mean() + joints[0, 1]) / 2
    upper = joints[[6, 9], 1].mean()
    views = (
        ("abdomen", np.isin(segments, ("pelvis", "torso")) & (face_y >= lower) & (face_y <= upper)),
        ("legs", np.isin(segments, ("left_thigh", "left_shin", "left_foot", "right_thigh", "right_shin", "right_foot"))),
        ("arms", arm_faces(centers, segments, joints)),
    )
    result = []
    for name, selected in views:
        triangles = faces[selected]
        result.append({"id": name, "triangles": triangles,
                       "coverage": surface_coverage(triangles, len(vertices))})
    return result


def demo_mesh(body, height_cm, dates=None, subject="demo"):
    if body.body_model_asset_identity != current_body_model_asset_identity():
        raise ValueError("BodyPack and preview have different SMPL model assets")
    track = body.select_track()
    vertices, joints, _, _ = generate_pose_model(track.betas, height_cm, PREVIEW_POSE)
    model = _load_smplfitter_body_model()
    faces = np.asarray(model.faces, dtype=np.int32)
    skin = model.weights.cpu().numpy()
    original = vertices.copy()
    joints = joints.copy()
    lift = -float(vertices[:, 1].min())
    original[:, 1] += lift
    joints[:, 1] += lift

    views = region_views(original, faces, joints)
    belt = joints[3] + [0, 28, 12]
    level = (original[:, 1] - belt[1]) / 193.0
    torso = skin[:, [0, 3, 6, 9]].sum(axis=1)
    region = np.maximum(0, 1 - level * level) ** 2 * np.clip(torso / 0.83, 0, 1)
    belt = belt.tolist()
    abdomen = np.zeros_like(original)
    abdomen[:, 0] = -0.018 * region * (original[:, 0] - belt[0])
    front = original[:, 2] - belt[2]
    abdomen[:, 2] = -np.where(front > 0, 0.040, 0.010) * region * front

    legs = np.zeros_like(original)
    arms = np.zeros_like(original)
    for direction, hip, knee, ankle, shoulder, elbow, wrist in (
        (1, 1, 4, 7, 16, 18, 20),
        (-1, 2, 5, 8, 17, 19, 21),
    ):
        legs += limb_delta(original, joints, skin, direction, hip, knee, [hip, knee], 0.036, hold_height=True)
        legs += limb_delta(original, joints, skin, direction, knee, ankle, [knee, ankle], 0.021, hold_height=True)
        arms += limb_delta(original, joints, skin, direction, shoulder, elbow, [shoulder, elbow], 0.042)
        arms += limb_delta(original, joints, skin, direction, elbow, wrist, [elbow, wrist], 0.010)

    change = abdomen + legs + arms
    if dates is None:
        records = [
            {"key": key, "label": label, "delta": weight * change if weight else None}
            for key, label, weight in DEMO_RECORDS
        ]
    else:
        days = [date.fromisoformat(day).toordinal() for day in dates]
        if not 6 <= len(days) <= 10 or sorted(set(days)) != days:
            raise ValueError("Demo dates must contain six to ten distinct days in order")
        span = days[-1] - days[0]
        records = [
            {"key": f"{subject}-{i + 1:02}", "date": day, "title": day, "label": day[5:].replace("-", "."),
             "delta": change * ((days[i] - days[0]) / span) if i else None}
            for i, day in enumerate(dates)
        ]
    return original, faces, records, views, joints


def visit_key(visit):
    identity = json.dumps([visit["id"], visit["protocol_id"]]).encode("ascii")
    return hashlib.sha256(identity).hexdigest()


def archive_mesh(archive, load_capture=None):
    subject = archive["subject_id"]
    height = positive_height(archive["height_cm"])
    model_id = current_body_model_asset_identity()["asset_sha256"]
    visits = sorted(
        ({**visit, "date": date.fromisoformat(visit["date"]).isoformat()} for visit in archive["visits"]),
        key=lambda visit: visit["date"],
    )
    if not visits:
        raise ValueError("The archive must contain at least one visit")
    seen_sources = set()
    seen_groups = set()
    shapes, joint_models = [], []
    saved = []
    for visit in visits:
        group = (visit["id"], visit["protocol_id"])
        if group in seen_groups:
            raise ValueError(f"Put repeated captures into one visit and protocol: {group}")
        seen_groups.add(group)
        observations = []
        for capture in visit["captures"]:
            if isinstance(capture, dict):
                item = ShapeObservation.from_json(json.dumps(capture))
            elif load_capture is not None:
                item = load_capture(capture, subject, *group)
            else:
                raise ValueError("A saved capture must be a shape observation")
            if (item.subject_id, item.visit_id, item.protocol_id, item.body_model_asset_sha256) != (
                subject, *group, model_id,
            ):
                raise ValueError(f"Observation identity does not match visit {group} or the installed SMPL")
            observations.append(item)
        fit = estimate_visit(observations)
        if seen_sources.intersection(fit.video_sha256s):
            raise ValueError("The same source recording appears in different visits")
        seen_sources.update(fit.video_sha256s)
        saved.append({**visit, "captures": [
            json.loads(item.to_json()) for item in dict.fromkeys(observations)
        ]})
        points, joints, _, _ = generate_pose_model(np.asarray(fit.betas), height, PREVIEW_POSE)
        joint_models.append(joints - joints[0])
        shapes.append(points - joints[0])

    canonical = {"subject_id": subject, "height_cm": height, "visits": saved}
    base = shapes[0].copy()
    reference_joints = joint_models[0].copy()
    lift = -float(base[:, 1].min())
    base[:, 1] += lift
    reference_joints[:, 1] += lift
    records = []
    date_counts = Counter(visit["date"] for visit in visits)
    date_indices = Counter()
    for i, (visit, shape) in enumerate(zip(visits, shapes, strict=True)):
        date_label = visit["date"]
        date_indices[date_label] += 1
        suffix = f" · {date_indices[date_label]}" if date_counts[date_label] > 1 else ""
        records.append({
            "key": visit_key(visit), "date": date_label, "label": date_label[5:].replace("-", ".") + suffix,
            "title": date_label + suffix, "device": visit["protocol_id"], "count": len(saved[i]["captures"]),
            "delta": shape - shapes[0] if i else None,
            "joints": joint_models[i].ravel().tolist() if i else None,
        })
    model = _load_smplfitter_body_model()
    faces = np.asarray(model.faces, dtype=np.int32)
    views = region_views(base, faces, reference_joints)
    return canonical, (base, faces, records, views, reference_joints)


def archive_from_file(source, run, height_cm):
    archive = json.loads(source.read_text(encoding="utf-8-sig"))
    if height_cm is not None:
        archive["height_cm"] = height_cm
    loaded = {}

    def load_capture(capture, subject_id, visit_id, protocol_id):
        path = (source.parent / capture).resolve(strict=True)
        is_body = path.suffix.lower() == ".body"
        key = (is_body, _file_sha256(path))
        if key not in loaded:
            loaded[key] = load_body(path) if is_body else reconstruct_subject(path, run / "motionpacks" / str(len(loaded) + 1))
        return ShapeObservation.from_body(
            loaded[key], subject_id=subject_id, visit_id=visit_id, protocol_id=protocol_id,
        )

    canonical, mesh = archive_mesh(archive, load_capture)
    (run / "archive.json").write_text(json.dumps(canonical, ensure_ascii=False, indent=2), encoding="utf-8")
    return mesh


def encoded(array, dtype):
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode("ascii")


def mesh_payload(original, faces, records, views, joints, stl_dir=None):
    start_faces = original[faces]
    start_normals = np.cross(start_faces[:, 1] - start_faces[:, 0], start_faces[:, 2] - start_faces[:, 0])
    if not np.isfinite(start_normals).all() or np.any(np.linalg.norm(start_normals, axis=1) <= 0):
        raise ValueError("Invalid reference triangles")
    if stl_dir is not None:
        stl_dir.mkdir()
    payload = []
    for record in records:
        delta = record["delta"]
        stage_faces = start_faces if delta is None else (original + delta)[faces]
        normals = np.cross(stage_faces[:, 1] - stage_faces[:, 0], stage_faces[:, 2] - stage_faces[:, 0])
        key = record["key"]
        if not np.all(np.einsum("ij,ij->i", start_normals, normals) > 0):
            raise ValueError(f"Shape deformation folds a mesh triangle in record {key}")
        name = f"shape-journal-{record['date']}-{key[:12]}" if "date" in record else f"shape-journal-demo-{key}"
        filename = name + "-mm.stl"
        if stl_dir is not None:
            write_stl(stl_dir / filename, stage_faces, normals)
        payload.append({
            **record, "stl": filename,
            "delta": encoded(delta, "<f4") if delta is not None else None,
        })
    model = _load_smplfitter_body_model()
    weights = model.weights.numpy()
    influences = np.argsort(weights, axis=1)[:, -4:]
    _, axes = make_body_frame(joints)
    try:
        hit_faces, hit_weights = generate_hip_crest_hits(original, joints, axes, faces)
        landmarks = {
            label: {"face_index": int(face), "weights": weights.tolist()}
            for label, face, weights in zip(("left_hip_crest", "right_hip_crest"), hit_faces, hit_weights, strict=True)
        }
    except ValueError:
        landmarks = {}
    definitions = read_definition_catalog(MEASUREMENT_DEFINITIONS_PATH)["definitions"]
    return {
        "positions": encoded(original, "<f4"),
        "tape": {
            "definitions": {item["item_id"]: item for item in definitions
                            if item["item_id"] in {"natural_waist", "calf", "upper_arm"}},
            "segment_ids": BODY_SEGMENT_IDS,
            "face_segments": encoded(generate_face_body_segment(faces), "u1"),
            "up": axes[1].tolist(),
            "joint_names": SMPL_24_JOINT_SCHEMA["names"],
            "landmark_hits": landmarks,
        },
        "pose_root": joints[0].tolist(),
        "rig": {
            "parents": list(map(int, model.kintree_parents)),
            "rotations": PREVIEW_POSE.ravel().tolist(),
            "joints": (joints - joints[0]).ravel().tolist(),
            "influences": encoded(influences, "u1"),
            "weights": encoded(np.take_along_axis(weights, influences, axis=1), "<f4"),
        },
        "triangles": encoded(faces, "<u2"),
        "records": payload,
        "views": [
            {"id": view["id"], "triangles": encoded(view["triangles"], "<u2"),
             "coverage": encoded(view["coverage"], "u1")}
            for view in views
        ],
    }


def bundle_scene(modules=None):
    specs = json.loads(STACK.read_text(encoding="utf-8"))["runtime_policy"]["build_packages"]
    packages = {name: version for name, version in (spec.rsplit("@", 1) for spec in specs)
                if name in {"three", "esbuild"}}
    with TemporaryDirectory(prefix="shape-journal-build-") as directory:
        temporary = Path(directory)
        if modules is None:
            subprocess.run(
                [_find_npm(), "install", "--prefix", str(temporary), "--ignore-scripts", "--no-audit",
                 "--no-fund", "--no-save", *(f"{name}@{version}" for name, version in packages.items())],
                cwd=ROOT, check=True,
            )
            modules = temporary / "node_modules"
        modules = modules.resolve(strict=True)
        for name, version in packages.items():
            installed = json.loads((modules / name / "package.json").read_bytes())
            if installed["version"] != version:
                raise ValueError(f"Expected {name}@{version} in {modules}")
        bundle = temporary / "demo.js"
        subprocess.run(
            ["node", str(modules / "esbuild/bin/esbuild"), str(DEMO / "scene.js"),
             "--bundle", "--minify", "--format=iife", "--target=es2022", "--charset=utf8",
             "--log-level=warning", f"--outfile={bundle}"],
            cwd=ROOT, check=True, env={**os.environ, "NODE_PATH": str(modules)},
        )
        template = (DEMO / "index.html").read_text(encoding="utf-8")
        return template.replace("__SCENE_CODE__", bundle.read_text(encoding="utf-8").replace("</script", "<\\/script"))


def render_preview(template, data, intake=None):
    def for_script(value):
        return json.dumps(value, separators=(",", ":")).replace("<", "\\u003c")
    return template.replace("__MESH_DATA__", for_script(data)).replace("__APP_DATA__", for_script(intake))


def positive_height(value):
    if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
        raise ValueError("Height must be a positive, finite number in centimetres")
    return float(value)


def build(source: Path, output: Path, *, height_cm: float | None = None, body_path: Path | None = None,
          modules: Path | None = None):
    if height_cm is not None:
        height_cm = positive_height(height_cm)
    source = source.resolve(strict=True)
    is_archive = source.suffix.lower() == ".json"
    with new_output_directory(output.absolute()) as run:
        if is_archive:
            if body_path is not None:
                raise ValueError("Declare archive captures in the input JSON")
            mesh = archive_from_file(source, run, height_cm)
        else:
            height_cm = 170.0 if height_cm is None else height_cm
            if body_path is None:
                body_path = reconstruct_subject(
                    source, run / "motionpack", on_stage=lambda stage: print(f"Video: {stage}", flush=True),
                ).path
            pack = load_artifacts(body_path)
            body = pack.body
            if body.source.video_sha256 != _file_sha256(source):
                raise ValueError("BodyPack and video have different source fingerprints")
            mesh = demo_mesh(body, height_cm)
            print(f"{body.source.total_frames} source frames, {height_cm:g} cm reference height")
        data = mesh_payload(*mesh, stl_dir=run / "stl")
        data["mode"] = "archive" if is_archive else "demo"
        aligned, pipeline = (None, None) if is_archive else source_alignment(source, pack, tail=True)
        if aligned:
            data["source"] = aligned
            with TemporaryDirectory(prefix="shape-intro-") as temporary:
                motion = source_motion(source, body, aligned, Path(temporary) / "source.bin", pipeline)
                if motion is not None:
                    packed = base64.b64encode(motion.read_bytes()).decode("ascii")
                    aligned["motion"] = "data:application/octet-stream;base64," + packed
        (run / "Shape-Journal.html").write_text(render_preview(bundle_scene(modules), data), encoding="utf-8")
    print(output.absolute() / "Shape-Journal.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source video or a capture archive JSON")
    parser.add_argument("output", type=Path, help="new output directory")
    parser.add_argument("--height-cm", type=float)
    parser.add_argument("--body", type=Path, help="reuse an existing BodyPack from this exact video")
    parser.add_argument("--node-modules", type=Path)
    args = parser.parse_args()
    build(args.source, args.output, height_cm=args.height_cm, body_path=args.body, modules=args.node_modules)
