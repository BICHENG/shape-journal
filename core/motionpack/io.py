"""MotionPack persistence primitives."""

import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from joblib import numpy_pickle

from .model import (
    BODY_METADATA,
    POSE_METADATA,
    PRODUCTION_NLF_MODEL_SHA256,
    PRODUCTION_NLF_QUERY_LAYOUT_ID,
    SMPL_VERTEX_SUBSET_SHA256,
    BodyTrack,
    PoseTrack,
    Track,
    normalize_body_track,
    normalize_pose_track,
    validate_body_model_asset_identity,
    validate_joint_schema,
    validate_pose_pack_identity,
)

SCHEMA_VERSION = 1
POSE_SCHEMA_VERSION = 2
BODY_SCHEMA_VERSION = 3
BODY_COMPAT_SCHEMA_VERSION = 2
PORTABLE_BODY_PREFIX = b"MOXION motionpack/1\n"
LEGACY_SCHEMA_VERSION = 0
_VERSIONS = {
    "detection": (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION),
    "pose": (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION, POSE_SCHEMA_VERSION),
    "body": (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION, BODY_COMPAT_SCHEMA_VERSION, BODY_SCHEMA_VERSION),
}
_OLD_EXECUTION_MODULES = {"nlf.motion_pack_v3", "core.motionpack.builders"}
_OLD_TRACK_MODULES = _OLD_EXECUTION_MODULES | {"scripts.run_to_build_motion_pack", "run_to_build_motion_pack", "__main__"}
_TRACK_CLASSES = {cls.__name__: cls for cls in (Track, PoseTrack, BodyTrack)}


class _MotionPackUnpickler(numpy_pickle.NumpyUnpickler):
    def find_class(self, module, name):
        if module in _OLD_EXECUTION_MODULES and name in {"Pose", "Detection"}:
            return SimpleNamespace
        if module in _OLD_TRACK_MODULES and name in _TRACK_CLASSES:
            return _TRACK_CLASSES[name]
        return super().find_class(module, name)


def _base_payload(kind: str) -> dict:
    return {"kind": kind, "schema_version": _VERSIONS[kind][-1]}


def _read_portable_payload(stream) -> dict:
    with gzip.GzipFile(fileobj=stream) as zipped:
        body = json.loads(zipped.readline())
        detection_bytes = zipped.readline().removesuffix(b"\n")
        pose_bytes = zipped.read()
    if body.get("pose_ref") != "pose.pose" or hashlib.sha256(pose_bytes).hexdigest() != body.get("pose_sha256"):
        raise ValueError("MotionPack body pose_sha256 does not match the embedded PosePack")
    pose = json.loads(pose_bytes)
    if hashlib.sha256(detection_bytes).hexdigest() != pose.get("det_sha256"):
        raise ValueError("MotionPack pose det_sha256 does not match the embedded Detection")
    detection = json.loads(detection_bytes)
    for data, kind, version, arrays in (
        (body, "body", BODY_SCHEMA_VERSION,
         ("betas", "pose_rotvecs", "joints", "frame_betas", "source_joint_uncertainties_mm")),
        (pose, "pose", POSE_SCHEMA_VERSION, ("poses3d", "poses2d", "uncertainties")),
        (detection, "detection", SCHEMA_VERSION, ("bboxes", "scores")),
    ):
        if data.get("kind") != kind or data.get("schema_version") != version:
            raise ValueError(f"Unsupported portable MotionPack {kind} payload")
        for track in data["tracks"].values():
            for name in arrays:
                track[name] = np.asarray(track[name], dtype=np.float32)
    if pose.get("det_ref") or pose.get("detection") is not None:
        raise ValueError("Portable PosePack must use its embedded Detection")
    tracks = {}
    for key, track in detection["tracks"].items():
        frames, boxes, scores = np.asarray(track["frames"]), track["bboxes"], track["scores"]
        if (track["id"] != int(key) or not np.issubdtype(frames.dtype, np.integer)
                or frames.ndim != 1 or not frames.size or frames[0] < 0 or frames[-1] >= detection["n_frames"]
                or np.any(np.diff(frames) != 1) or boxes.shape != (len(frames), 4)
                or scores.shape != (len(frames),) or not np.isfinite(boxes).all()
                or not np.isfinite(scores).all() or np.any(boxes[:, 2:] <= 0)):
            raise ValueError("Portable MotionPack detection track is invalid")
        tracks[int(key)] = Track(int(key), frames.astype(np.int32), boxes, scores)
    detection["tracks"] = tracks
    detection["shape"] = tuple(detection["shape"])
    body["_embedded_records"] = (detection, pose)
    return body


def _load_payload(path: str | Path, expected_kind: str, *, content: bytes | None = None) -> dict:
    path = str(path)
    with (open(path, "rb") if content is None else BytesIO(content)) as f:
        if expected_kind == "body":
            if f.read(len(PORTABLE_BODY_PREFIX)) == PORTABLE_BODY_PREFIX:
                return _read_portable_payload(f)
            f.seek(0)
        with numpy_pickle._validate_fileobject_and_memmap(f, path, None) as (fobj, mmap_mode):
            if isinstance(fobj, str):
                if content is not None:
                    raise ValueError("Digest-bound PosePack must be stored in a single file")
                data = joblib.load(fobj)
            else:
                data = _MotionPackUnpickler(path, fobj, True, mmap_mode=mmap_mode).load()
    version = data.get("schema_version", LEGACY_SCHEMA_VERSION)
    kind = data.get("kind", expected_kind)
    if kind != expected_kind:
        raise ValueError(f"Expected MotionPack {expected_kind}, got {kind}")
    if version not in _VERSIONS[expected_kind]:
        raise ValueError(f"Unsupported MotionPack schema_version: {version}")
    return data


def _pose_identity(data: dict) -> tuple[str | None, str | None, dict | None]:
    if int(data.get("schema_version", LEGACY_SCHEMA_VERSION)) != POSE_SCHEMA_VERSION:
        return None, None, None
    metadata = {name: data.get(name) for name in POSE_METADATA}
    metadata = validate_pose_pack_identity(
        data.get("producer_model_sha256"),
        data.get("query_layout_id"),
        metadata,
    )
    return data["producer_model_sha256"], data["query_layout_id"], metadata


def _pose_tracks(data: dict) -> dict[int, PoseTrack]:
    require_v2 = int(data.get("schema_version", LEGACY_SCHEMA_VERSION)) == POSE_SCHEMA_VERSION
    return {
        int(track_id): normalize_pose_track(track, require_v2=require_v2)
        for track_id, track in data["tracks"].items()
    }


def _body_fields(data: dict) -> dict:
    version = int(data.get("schema_version", LEGACY_SCHEMA_VERSION))
    modern = version in {BODY_COMPAT_SCHEMA_VERSION, BODY_SCHEMA_VERSION}
    identity = subset = schema = metadata = None
    if modern:
        identity = validate_body_model_asset_identity(data.get("body_model_asset_identity"))
        subset = data.get("smpl_vertex_subset_sha256")
        if subset != SMPL_VERTEX_SUBSET_SHA256:
            raise ValueError("BodyPack smpl_vertex_subset_sha256 does not match the production 1048 layout")
        if data.get("joint_schema") is None:
            raise ValueError(f"MotionPack body v{version} is missing joint_schema")
        schema = validate_joint_schema(data["joint_schema"])
        if not isinstance(data.get("metadata"), dict) or data["metadata"] != BODY_METADATA:
            raise ValueError(f"MotionPack body v{version} metadata does not match BODY_METADATA")
        metadata = dict(data["metadata"])

    tracks = {int(tid): normalize_body_track(track, allow_legacy_pose=not modern)
              for tid, track in data["tracks"].items()}
    if modern:
        for track in tracks.values():
            if track.source_frames is None or track.source_joint_uncertainties_mm is None:
                raise ValueError(f"MotionPack body v{version} track {track.id} is missing source frame metadata")
    return {
        "body_model_asset_identity": identity,
        "smpl_vertex_subset_sha256": subset,
        "joint_schema": schema,
        "metadata": metadata,
        "tracks": tracks,
    }


def _resolve_ref(ref_path: str, owner_path: str | Path) -> str:
    ref = Path(ref_path)
    owner = Path(owner_path).resolve().parent
    if not ref.is_absolute() and (owner / ref).is_file():
        return str(owner / ref)
    if ref.is_file():
        return str(ref)
    sibling = owner / ref.name
    return str(sibling if sibling.is_file() else ref)


def _file_sha256(path: str | Path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_bound_pose(path: Path, expected_sha256: str | None = None) -> tuple[dict, str]:
    if not path.is_file():
        raise ValueError(f"MotionPack body pose_ref does not exist: {path}")
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and expected_sha256 != digest:
        raise ValueError("MotionPack body pose_sha256 does not match pose_ref")
    return _load_payload(path, "pose", content=content), digest


def _validate_body_pose_tracks(pose_tracks, body_tracks) -> None:
    for track_id, track in body_tracks.items():
        pose_track = pose_tracks.get(track_id)
        if pose_track is None or not np.array_equal(track.source_frames, pose_track.source_frames):
            raise ValueError(f"MotionPack body track {track_id} source_frames do not match PosePack")


def dump_detection(detection, path: str) -> None:
    data = _base_payload("detection")
    data.update(
        {
            "video": detection.video_path,
            "fps": detection.fps,
            "shape": detection.shape,
            "n_frames": detection.n_frames,
            "tracks": detection.tracks,
        }
    )
    if video_sha256 := getattr(detection, "video_sha256", None):
        data["video_sha256"] = video_sha256
    joblib.dump(
        data,
        path,
        compress=3,
    )


def dump_pose(pose, path: str, det_path: str = None) -> None:
    data = _base_payload("pose")
    metadata = validate_pose_pack_identity(
        getattr(pose, "producer_model_sha256", None),
        getattr(pose, "query_layout_id", None),
        getattr(pose, "metadata", None),
    )
    data["producer_model_sha256"] = PRODUCTION_NLF_MODEL_SHA256
    data["query_layout_id"] = PRODUCTION_NLF_QUERY_LAYOUT_ID
    data.update(metadata)
    tracks = _pose_tracks({"schema_version": POSE_SCHEMA_VERSION, "tracks": pose.tracks})
    detection_tracks = getattr(getattr(pose, "detection", None), "tracks", {})
    for track_id, track in tracks.items():
        detection_track = detection_tracks.get(track_id)
        if detection_track is None or not np.array_equal(track.source_frames, detection_track.frames):
            raise ValueError(f"PoseTrack {track_id} source_frames do not match Detection Track.frames")
    data["tracks"] = tracks
    if det_path:
        data["det_ref"] = det_path
    else:
        data["detection"] = pose.detection
    joblib.dump(data, path, compress=3)


def dump_body(body, path: str, pose_path: str = None) -> None:
    _write_body(body, path, pose_path)


def _write_body(body, path: str | Path, pose_ref: str | None) -> dict:
    data = {"schema_version": BODY_SCHEMA_VERSION}
    data.update(_body_fields({
        **data,
        "body_model_asset_identity": getattr(body, "body_model_asset_identity", None),
        "smpl_vertex_subset_sha256": getattr(body, "smpl_vertex_subset_sha256", None),
        "joint_schema": getattr(body, "joint_schema", None),
        "metadata": getattr(body, "metadata", None),
        "tracks": body.tracks,
    }))
    if not pose_ref:
        raise ValueError(f"BodyPack v{BODY_SCHEMA_VERSION} writer requires pose_path")
    data["pose_ref"] = pose_ref
    pose_payload, data["pose_sha256"] = _load_bound_pose(Path(_resolve_ref(pose_ref, path)))
    _pose_identity(pose_payload)
    _validate_body_pose_tracks(_pose_tracks(pose_payload), data["tracks"])
    joblib.dump({"kind": "body", **data}, path, compress=3)
    return data
