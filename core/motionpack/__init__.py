"""MotionPack read and reconstruction entry points."""

from .loading import (
    MotionPackArtifacts,
    MotionPackBody,
    MotionPackDetection,
    MotionPackPose,
    MotionPackSource,
    load_artifacts,
    load_body,
    load_detection,
    load_pose,
)
from .model import BodyTrack, PoseTrack, Track

_BUILDERS = ("Body", "Detection", "NoPersonTrackError", "Pose", "build_artifacts", "reconstruct_subject")
__all__ = sorted([
    *_BUILDERS, "BodyTrack", "MotionPackArtifacts", "MotionPackBody", "MotionPackDetection", "MotionPackPose",
    "MotionPackSource", "PoseTrack", "Track", "load_artifacts", "load_body", "load_detection", "load_pose",
])


def __getattr__(name: str):
    if name in _BUILDERS:
        from . import builders

        return getattr(builders, name)
    raise AttributeError(f"module 'core.motionpack' has no attribute {name!r}")
