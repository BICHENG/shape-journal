"""Load existing MotionPack base-data artifacts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .io import (
    BODY_SCHEMA_VERSION,
    _body_fields,
    _load_bound_pose,
    _load_payload,
    _pose_identity,
    _pose_tracks,
    _resolve_ref,
    _validate_body_pose_tracks,
)
from .model import SMPL_24_JOINT_SCHEMA, BodyTrack, PoseTrack, Track


@dataclass(frozen=True)
class MotionPackSource:
    video_path: str = ""
    total_frames: int = 0
    source_fps: float = 0.0
    detection_ref: str | None = None
    video_sha256: str | None = None


@dataclass(frozen=True)
class MotionPackDetection:
    path: Path
    tracks: dict[int, Track]
    schema_version: int
    video_path: str
    source_fps: float
    shape: tuple[int, int] | None
    total_frames: int
    video_sha256: str | None = None

    @property
    def source(self) -> MotionPackSource:
        return MotionPackSource(
            self.video_path, self.total_frames, self.source_fps, str(self.path), self.video_sha256,
        )


@dataclass(frozen=True)
class MotionPackPose:
    path: Path
    tracks: dict[int, PoseTrack]
    schema_version: int
    producer_model_sha256: str | None
    query_layout_id: str | None
    metadata: dict | None
    detection_ref: str | None
    source: MotionPackSource
    detection: MotionPackDetection | SimpleNamespace | None = None


@dataclass(frozen=True)
class MotionPackBody:
    path: Path
    tracks: dict[int, BodyTrack]
    schema_version: int
    body_model_asset_identity: dict | None
    smpl_vertex_subset_sha256: str | None
    joint_schema: dict | None
    metadata: dict | None
    pose_ref: str | None
    pose_sha256: str | None
    source: MotionPackSource

    def track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.tracks))

    def select_track(self, track_id: int | None = None) -> BodyTrack:
        if not self.tracks:
            raise ValueError("MotionPack body has no tracks")
        selected = track_id if track_id is not None else max(self.tracks, key=lambda tid: len(self.tracks[tid].joints))
        if selected not in self.tracks:
            raise KeyError(f"Track {selected} not found")
        return self.tracks[selected]

    def summary(self) -> dict:
        return {
            "path": str(self.path),
            "schema_version": self.schema_version,
            "body_model_asset_identity": self.body_model_asset_identity,
            "smpl_vertex_subset_sha256": self.smpl_vertex_subset_sha256,
            "joint_schema": self.joint_schema,
            "metadata": self.metadata,
            "pose_ref": self.pose_ref,
            "pose_sha256": self.pose_sha256,
            "source": asdict(self.source),
            "track_count": len(self.tracks),
            "tracks": {
                int(tid): {
                    "frames": len(track.joints),
                    **{label: list(array.shape) if array is not None else None for label, array in (
                        ("betas_shape", track.betas), ("pose_rotvecs_shape", track.pose_rotvecs),
                        ("joints_shape", track.joints), ("frame_betas_shape", track.frame_betas),
                        ("source_frames_shape", track.source_frames),
                        ("source_joint_uncertainties_shape", track.source_joint_uncertainties_mm),
                    )},
                }
                for tid, track in self.tracks.items()
            },
        }


@dataclass(frozen=True)
class MotionPackArtifacts:
    path: Path
    body: MotionPackBody
    pose: MotionPackPose | None = None

    @property
    def tracks(self) -> dict[int, BodyTrack]:
        return self.body.tracks

    def track_frame_ranges(self) -> dict[int, dict[str, int]]:
        detection = self.pose.detection if self.pose else None
        if detection is None:
            return {}
        return {
            tid: {"start_frame": int(track.frames[0]), "end_frame": int(track.frames[-1])}
            for tid, track in detection.tracks.items() if tid in self.body.tracks and len(track.frames)
        }


def _embedded_source(detection) -> MotionPackSource:
    detection = getattr(detection, "detection", detection)
    if isinstance(detection, MotionPackDetection):
        return detection.source
    return MotionPackSource(
        video_path=str(getattr(detection, "video_path", "") or ""),
        total_frames=int(getattr(detection, "n_frames", 0) or 0),
        source_fps=float(getattr(detection, "fps", 0.0) or 0.0),
        video_sha256=getattr(detection, "video_sha256", None),
    )


def load_detection(path: str | Path) -> MotionPackDetection:
    path = Path(path)
    return _read_detection(path, _load_payload(path, "detection"))


def _read_detection(path: Path, payload: dict) -> MotionPackDetection:
    return MotionPackDetection(
        path,
        dict(payload["tracks"]),
        int(payload.get("schema_version", 0)),
        str(payload.get("video", "") or ""),
        float(payload.get("fps", 0.0) or 0.0),
        payload.get("shape"),
        int(payload.get("n_frames", 0) or 0),
        payload.get("video_sha256"),
    )


def _pose_detection(path: Path, payload: dict, detection=None):
    if det_ref := payload.get("det_ref"):
        det_path = Path(_resolve_ref(det_ref, str(path)))
        if detection is None and det_path.is_file():
            detection = load_detection(det_path)
        source = detection.source if isinstance(detection, MotionPackDetection) else _embedded_source(detection)
        return detection, replace(source, detection_ref=str(det_path))
    detection = detection if detection is not None else payload.get("detection")
    return detection, _embedded_source(detection)


def _read_pose(path: Path, payload: dict, detection=None) -> MotionPackPose:
    producer, layout, metadata = _pose_identity(payload)
    tracks = _pose_tracks(payload)
    detection, source = _pose_detection(path, payload, detection)
    return MotionPackPose(
        path, tracks, int(payload.get("schema_version", 0)),
        producer, layout, metadata, payload.get("det_ref"), source, detection,
    )


def load_pose(path: str | Path) -> MotionPackPose:
    path = Path(path)
    return _read_pose(path, _load_payload(path, "pose"))


def _read_body(path: Path, *, with_pose: bool = False):
    payload = _load_payload(path, "body")
    version = int(payload.get("schema_version", 0))
    fields = _body_fields(payload)
    pose_ref = payload.get("pose_ref")
    modern = version == BODY_SCHEMA_VERSION
    if modern and (not isinstance(pose_ref, str) or not pose_ref):
        raise ValueError(f"MotionPack body v{version} is missing pose_ref")
    if modern and not isinstance(payload.get("pose_sha256"), str):
        raise ValueError("MotionPack body pose_sha256 does not match pose_ref")

    embedded = payload.get("_embedded_records")
    pose_path = Path(_resolve_ref(pose_ref, path)) if pose_ref and embedded is None else None
    pose_sha256 = None
    pose = payload.get("pose")
    source = MotionPackSource() if pose_path else _embedded_source(getattr(pose, "detection", None))
    if embedded is not None:
        detection, pose_payload = embedded
        pose = _read_pose(path, pose_payload, _read_detection(path, detection))
        for tid, track in pose.tracks.items():
            boxes = pose.detection.tracks.get(tid)
            if boxes is None or not np.array_equal(boxes.frames, track.source_frames):
                raise ValueError(f"MotionPack pose track {tid} source_frames do not match Detection")
        _validate_body_pose_tracks(pose.tracks, fields["tracks"])
        joint_count = len(SMPL_24_JOINT_SCHEMA["names"])
        for tid, track in fields["tracks"].items():
            if not np.array_equal(track.source_joint_uncertainties_mm, pose.tracks[tid].uncertainties[:, -joint_count:]):
                raise ValueError(f"MotionPack body track {tid} uncertainty does not match PosePack")
        pose_sha256, source = payload["pose_sha256"], pose.source
    elif modern:
        pose_payload, pose_sha256 = _load_bound_pose(pose_path, payload["pose_sha256"])
        pose = _read_pose(pose_path, pose_payload)
        _validate_body_pose_tracks(pose.tracks, fields["tracks"])
        source = pose.source
    elif pose_path and pose_path.is_file():
        pose_payload = _load_payload(pose_path, "pose")
        if with_pose:
            pose = _read_pose(pose_path, pose_payload)
            source = pose.source
        else:
            _, source = _pose_detection(pose_path, pose_payload)
    body = MotionPackBody(
        path=path, schema_version=version, pose_ref=pose_ref, pose_sha256=pose_sha256, source=source, **fields,
    )
    return body, pose


def load_body(path: str | Path) -> MotionPackBody:
    body, _ = _read_body(Path(path))
    return body


def load_artifacts(path: str | Path) -> MotionPackArtifacts:
    path = Path(path)
    body_path = path / "body.body" if path.is_dir() else path
    path = body_path.parent
    body, pose = _read_body(body_path, with_pose=True)
    if not isinstance(pose, MotionPackPose):
        sibling = path / "pose.pose"
        pose = load_pose(sibling) if sibling.is_file() else None
    return MotionPackArtifacts(path, body, pose)
