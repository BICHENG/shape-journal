"""Place a fitted frame back into the pinhole camera of its actual source image."""

from base64 import b64encode
from fractions import Fraction
import gzip
import logging
from pathlib import Path
import shutil
import struct
import subprocess
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch

from body_measurement.mesh import _load_smplfitter_body_model
from nlf_pt_ops import NLF, compute_intrinsics, crop_cameras


LOG = logging.getLogger(__name__)
MEASURE_TAIL_SECONDS = 3.25


def camera_vertices(track, rows):
    model = _load_smplfitter_body_model()
    with torch.inference_mode():
        local = model(
            pose_rotvecs=torch.as_tensor(track.pose_rotvecs[rows], dtype=torch.float32),
            shape_betas=torch.as_tensor(track.frame_betas[rows], dtype=torch.float32),
        )
    joints = local["joints"].numpy()
    recorded = track.joints[rows] * 0.001
    translation = np.mean(recorded - joints, axis=1, keepdims=True)
    to_view = [1, -1, -1]
    vertices = (local["vertices"].numpy() + translation) * to_view
    if not np.isfinite(vertices).all() or np.any(vertices[..., 2] >= 0):
        raise ValueError("Aligned vertices must be in front of the source camera")
    pelvis = (joints + translation)[:, 0] * to_view
    return vertices, np.column_stack((pelvis, track.pose_rotvecs[rows, 0]))


def inference_frames(pack, track_id, frames, intrinsics):
    pose = pack.pose
    detection = getattr(pose, "detection", None)
    observed = pose.tracks.get(track_id) if pose is not None else None
    detection_tracks = getattr(detection, "tracks", {})
    followed = detection_tracks.get(track_id)
    if observed is None or followed is None or observed.source_frames is None:
        return None
    pose_rows = np.searchsorted(observed.source_frames, frames)
    box_rows = np.searchsorted(followed.frames, frames)
    if (np.any(pose_rows >= len(observed.source_frames)) or np.any(box_rows >= len(followed.frames))
            or not np.array_equal(observed.source_frames[pose_rows], frames)
            or not np.array_equal(followed.frames[box_rows], frames)):
        raise ValueError("Model features, follow boxes and fitted poses must share source frames")

    boxes = followed.bboxes[box_rows]
    xyxy = torch.tensor(np.column_stack((boxes[:, :2], boxes[:, :2] + boxes[:, 2:])), dtype=torch.float32)
    raw = intrinsics.expand(len(frames), -1, -1)
    crop, rotation, _ = crop_cameras(xyxy, raw)
    crop_to_source = (raw @ torch.linalg.inv(crop @ rotation)).numpy().reshape(-1, 9)
    cameras = np.column_stack((crop_to_source, boxes, followed.scores[box_rows])).astype("<f4")

    # Store camera-space points in millimetres and crop pixels to quarter-pixel precision.
    coords = np.concatenate((
        observed.poses2d[pose_rows] * 4,
        observed.poses3d[pose_rows] * np.array([1, -1, -1]),
        observed.uncertainties[pose_rows, :, None] * 1000,
    ), axis=-1)
    if not np.isfinite(coords).all() or np.abs(coords).max() > 32767:
        raise ValueError("Intermediate points exceed the preview encoding")
    return cameras, np.rint(coords).astype("<i2")


def source_alignment(video, pack, *, tail=False, excerpt=False):
    body = pack.body
    track = body.select_track()
    if track.source_frames is None or track.frame_betas is None:
        return None, None
    breaks = np.r_[0, np.flatnonzero(np.diff(track.source_frames) != 1) + 1, len(track.source_frames)]
    span = int(np.argmax(np.diff(breaks)))
    start, end = map(int, breaks[span:span + 2])
    fps = float(body.source.source_fps or 30)
    measure_start = max(start, end - round(MEASURE_TAIL_SECONDS * fps))
    if excerpt:
        start = measure_start
    index = measure_start if tail or excerpt else start + round((end - start - 1) * .75)
    frame_id = int(track.source_frames[index])
    trajectory = track.joints[start:end] * [0.001, -0.001, -0.001]
    center_x, _, center_z = np.median(trajectory[:, 0], axis=0)
    floor_y = float(np.percentile(trajectory[:, [10, 11], 1], 1)) - 0.05
    plane_depth = max(3.4, 0.55 - float(trajectory[..., 2].min()))
    reader = cv2.VideoCapture(str(video))
    try:
        for _ in range(frame_id):
            if not reader.grab():
                raise ValueError("Recording ended before the aligned frame")
        ok, image = reader.read()
        if not ok:
            raise ValueError("Could not read the aligned frame")
    finally:
        reader.release()

    vertices, roots = camera_vertices(track, slice(index, index + 1))
    height, width = image.shape[:2]
    intrinsics = compute_intrinsics(height, width, device="cpu")
    lens = intrinsics.numpy()
    encoded, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not encoded:
        raise ValueError("Could not encode the aligned image")
    # Local joint rotations in radians and fitted joint centers in view-space millimetres.
    postures = np.concatenate((track.pose_rotvecs[start:end], trajectory * 1000), axis=-1)
    source = {
        "image": "data:image/jpeg;base64," + b64encode(jpeg.tobytes()).decode("ascii"),
        "positions": b64encode(np.asarray(vertices, dtype="<f4").tobytes()).decode("ascii"),
        "pose_root": roots[0].tolist(),
        "origin": [float(-center_x), -floor_y, float(-center_z)],
        "plane_depth": plane_depth,
        "width": width, "height": height, "frame": frame_id,
        "first": int(track.source_frames[start]), "count": end - start,
        "total": body.source.total_frames,
        "fps": fps, "measure_start": int(track.source_frames[measure_start]),
        "lens": [float(lens[0, 0]), float(lens[1, 1]), float(lens[0, 2]), float(lens[1, 2])],
        "postures": b64encode(postures.astype("<f4").tobytes()).decode("ascii"),
    }
    track_id = next(tid for tid, item in body.tracks.items() if item is track)
    pipeline = inference_frames(pack, track_id, track.source_frames[start:end], intrinsics)
    if pipeline:
        cameras, points = pipeline
        source["pipeline"] = {
            "side": NLF.proc_side, "stride": NLF.stride_test, "count": points.shape[1], "joints": track.joints.shape[1],
            "camera": cameras[index - start].tolist(),
            "points": b64encode(points[index - start].tobytes()).decode("ascii"),
        }
    return source, pipeline


def source_motion(video, body, alignment, destination, pipeline):
    """Pack the fitted span and its matching video for frame-accurate playback and seeking."""
    if destination.exists():
        return destination
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or alignment["count"] < 2:
        return None
    track = body.select_track()
    frame, count, fps = (alignment[key] for key in ("first", "count", "fps"))
    time_base = str(Fraction(1 / fps).limit_denominator(90000))
    start = int(np.searchsorted(track.source_frames, frame))
    rows = slice(start, start + count)
    positions, roots = camera_vertices(track, rows)
    if np.any(positions[..., 2] <= -alignment["plane_depth"]):
        raise ValueError("The source plane must be behind the tracked subject")
    mm = np.rint(positions * 1000)
    if np.abs(mm).max() > 32767:
        raise ValueError("The source motion exceeds the millimetre encoding")
    mm = mm.astype("<i2")
    mm[1:] = np.diff(mm, axis=0)
    with TemporaryDirectory(prefix=".aligned-", dir=destination.parent) as directory:
        clip = Path(directory) / "source.mp4"
        stage = Path(directory) / "motion.bin"
        result = subprocess.run(
            [ffmpeg, "-nostdin", "-v", "error", "-i", str(video), "-an", "-vf",
             f"trim=start_frame={frame}:end_frame={frame + count},settb={time_base},setpts=N",
             "-fps_mode", "passthrough", "-enc_time_base", time_base,
             "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-g", "15",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(clip)],
            capture_output=True,
        )
        if result.returncode:
            LOG.warning("Could not encode the aligned excerpt: %s", result.stderr.decode(errors="replace").strip())
            return None
        verify = cv2.VideoCapture(str(clip))
        try:
            if int(verify.get(cv2.CAP_PROP_FRAME_COUNT)) != count:
                raise ValueError("Encoded source and fitted pose counts differ")
            size = (int(verify.get(cv2.CAP_PROP_FRAME_WIDTH)), int(verify.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if size != (alignment["width"], alignment["height"]):
                raise ValueError("Encoded source and fitted camera sizes differ")
        finally:
            verify.release()
        with gzip.open(stage, "wb", compresslevel=4) as packed:
            packed.write(struct.pack("<I", count))
            packed.write(mm.tobytes())
            packed.write(roots.astype("<f4").tobytes())
            if pipeline:
                cameras, points = pipeline
                packed.write(cameras.tobytes())
                encoded_points = points.copy()
                encoded_points[1:] = np.diff(points, axis=0)
                packed.write(encoded_points.tobytes())
            with clip.open("rb") as movie:
                shutil.copyfileobj(movie, packed)
        stage.replace(destination)
    return destination
