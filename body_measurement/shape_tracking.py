"""Portable shape observations and canonical comparisons for repeated captures."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from math import isfinite
from statistics import median
from typing import TYPE_CHECKING, Iterable, Literal

if TYPE_CHECKING:
    import numpy as np

    from core.motionpack import MotionPackBody

__all__ = ["ShapeObservation", "estimate_visit", "compare_shapes"]

_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ShapeObservation:
    """A single recording's fitted shape, with identity supplied by the caller."""

    subject_id: str
    visit_id: str
    protocol_id: str
    video_sha256: str
    body_model_asset_sha256: str
    track_id: int
    betas: tuple[float, ...]

    def __post_init__(self) -> None:
        for field in ("subject_id", "visit_id", "protocol_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a nonempty string")
        for field in ("video_sha256", "body_model_asset_sha256"):
            value = getattr(self, field)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        if type(self.track_id) is not int or self.track_id < 0:
            raise ValueError("track_id must be a nonnegative integer")
        if not isinstance(self.betas, (list, tuple)) or len(self.betas) != 10 or not all(
            type(value) in (int, float) and isfinite(value) for value in self.betas
        ):
            raise ValueError("betas must contain 10 finite numbers")
        object.__setattr__(self, "betas", tuple(map(float, self.betas)))

    @classmethod
    def from_body(
        cls,
        body: MotionPackBody,
        *,
        subject_id: str,
        visit_id: str,
        protocol_id: str,
        track_id: int | None = None,
        video_sha256: str | None = None,
    ) -> ShapeObservation:
        """Extract from a trusted, already loaded BodyPack."""
        if body.body_model_asset_identity is None:
            raise ValueError("BodyPack has no body model identity")
        source_sha256 = body.source.video_sha256
        if source_sha256 is not None and video_sha256 not in (None, source_sha256):
            raise ValueError("supplied video_sha256 does not match the BodyPack source")
        video_sha256 = source_sha256 if source_sha256 is not None else video_sha256
        if video_sha256 is None:
            raise ValueError("supply the verified source video SHA-256 for this BodyPack")
        if track_id is None and len(body.tracks) > 1:
            raise ValueError("select a track when a BodyPack contains multiple people")
        track = body.select_track(track_id)
        return cls(
            subject_id=subject_id,
            visit_id=visit_id,
            protocol_id=protocol_id,
            video_sha256=video_sha256,
            body_model_asset_sha256=body.body_model_asset_identity["asset_sha256"],
            track_id=int(track.id),
            betas=tuple(map(float, track.betas)),
        )

    def to_json(self) -> str:
        return json.dumps({"schema_version": _SCHEMA_VERSION, **asdict(self)}, ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> ShapeObservation:
        fields = json.loads(data)
        if not isinstance(fields, dict):
            raise ValueError("shape observation must be a JSON object")
        version = fields.pop("schema_version", None)
        if type(version) is not int or version != _SCHEMA_VERSION:
            raise ValueError("unsupported shape observation schema version")
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class VisitShape:
    subject_id: str
    visit_id: str
    protocol_id: str
    body_model_asset_sha256: str
    video_sha256s: tuple[str, ...]
    betas: tuple[float, ...]


def estimate_visit(observations: Iterable[ShapeObservation]) -> VisitShape:
    """Take the coordinate median of independent recordings from one visit and protocol."""
    captures: dict[str, ShapeObservation] = {}
    for observation in observations:
        stored = captures.setdefault(observation.video_sha256, observation)
        if stored != observation:
            raise ValueError("conflicting observations for the same source video")
    if not captures:
        raise ValueError("at least one shape observation is required")

    first = next(iter(captures.values()))
    if any(
        observation.subject_id != first.subject_id
        or observation.visit_id != first.visit_id
        or observation.protocol_id != first.protocol_id
        or observation.body_model_asset_sha256 != first.body_model_asset_sha256
        for observation in captures.values()
    ):
        raise ValueError("a visit estimate requires the same subject, visit, protocol and body model")
    beta_columns = zip(*(observation.betas for observation in captures.values()), strict=True)
    return VisitShape(
        subject_id=first.subject_id,
        visit_id=first.visit_id,
        protocol_id=first.protocol_id,
        body_model_asset_sha256=first.body_model_asset_sha256,
        video_sha256s=tuple(sorted(captures)),
        betas=tuple(median(column) for column in beta_columns),
    )


@dataclass(frozen=True, slots=True, eq=False)
class ShapeComparison:
    kind: Literal["repeat", "cross_protocol", "longitudinal"]
    height_cm: float
    faces: np.ndarray
    before_vertices_mm: np.ndarray
    after_vertices_mm: np.ndarray
    displacement_mm: np.ndarray
    rms_mm: float
    p95_mm: float


def compare_shapes(before: VisitShape, after: VisitShape, *, height_cm: float) -> ShapeComparison:
    """Compare corresponding canonical vertices at one measured reference stature."""
    if before.subject_id != after.subject_id:
        raise ValueError("compare shapes from the same subject")
    if before.body_model_asset_sha256 != after.body_model_asset_sha256:
        raise ValueError("compare shapes from the same body model")
    if not set(before.video_sha256s).isdisjoint(after.video_sha256s):
        raise ValueError("compared estimates share a recording")
    if isinstance(height_cm, bool) or not isfinite(height_cm) or height_cm <= 0:
        raise ValueError("height_cm must be a positive, finite reference stature")

    import numpy as np

    from core.motionpack.model import SMPL_24_JOINT_SCHEMA
    from core.smpl_assets import current_body_model_asset_identity
    from .mesh import _mesh_faces, generate_pose_model

    if before.body_model_asset_sha256 != current_body_model_asset_identity()["asset_sha256"]:
        raise ValueError("shape body model does not match the installed model")
    pelvis = SMPL_24_JOINT_SCHEMA["names"].index("pelvis")
    before_vertices, before_joints, _, _ = generate_pose_model(np.asarray(before.betas), height_cm)
    after_vertices, after_joints, _, _ = generate_pose_model(np.asarray(after.betas), height_cm)
    before_vertices -= before_joints[pelvis]
    after_vertices -= after_joints[pelvis]
    displacement = after_vertices - before_vertices
    distance = np.linalg.norm(displacement, axis=1)
    kind: Literal["repeat", "cross_protocol", "longitudinal"]
    if before.protocol_id != after.protocol_id:
        kind = "cross_protocol"
    else:
        kind = "repeat" if before.visit_id == after.visit_id else "longitudinal"
    return ShapeComparison(
        kind=kind,
        height_cm=float(height_cm),
        faces=_mesh_faces().copy(),
        before_vertices_mm=before_vertices,
        after_vertices_mm=after_vertices,
        displacement_mm=displacement,
        rms_mm=float(np.sqrt(np.mean(distance ** 2))),
        p95_mm=float(np.percentile(distance, 95)),
    )
