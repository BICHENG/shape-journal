"""Detect and track people, predict pose, and fit SMPL BodyPacks."""

from bisect import bisect_left
from itertools import pairwise
from pathlib import Path
from typing import Callable

import torch
import numpy as np

from core.output_directory import new_output_directory
from core.smpl_fit import SMPLFitter
from .io import _file_sha256, _load_payload, _resolve_ref, _write_body, dump_body, dump_detection, dump_pose
from .loading import (
    MotionPackBody, MotionPackDetection, MotionPackPose, MotionPackSource, _read_body, _read_pose, load_detection,
)
from .model import (
    BODY_METADATA,
    POSE_METADATA,
    BodyTrack,
    PoseTrack,
    Track,
    SMPL_24_JOINT_SCHEMA,
    normalize_pose_track,
    validate_pose_pack_identity,
)



class Detection:
    """实例化即检测（三阶段：检测→双向跟踪→密集化）"""

    def __init__(self, video_path: str, device: str = 'cuda', threshold: float = 0.3,
                 bidirectional: bool = True, *, detector=None):
        """
        Args:
            video_path: 视频路径
            device: 设备
            threshold: 检测阈值
            bidirectional: 启用双向跟踪（前向+后向ByteTrack合并，提高首尾帧召回率）
        """
        from video_io import VideoReader
        from core.persondet.boxmot_compat import create_bytetrack

        self.video_path = str(video_path)
        with VideoReader(video_path, device=device) as reader:
            self.fps = reader.fps
            self.shape = (reader.height, reader.width)

            if detector is None:
                from core.persondet.detector_trt import YOLO26TRTDetector
                detector = YOLO26TRTDetector(device=device)
            dummy_frame = np.zeros((reader.height, reader.width, 3), dtype=np.uint8)

            per_frame_dets: dict[int, np.ndarray] = {}
            for fid, frame in reader:
                f = frame.float() / 255.0 if frame.dtype == torch.uint8 else frame
                boxes = detector.detect(f.unsqueeze(0), threshold=threshold)
                per_frame_dets[fid] = boxes

        self.n_frames = len(per_frame_dets)
        del detector
        torch.cuda.empty_cache()

        tracks = self._run_tracker(create_bytetrack(), per_frame_dets, dummy_frame)
        if bidirectional:
            backward = self._run_tracker(create_bytetrack(), per_frame_dets, dummy_frame, reverse=True)
            self._merge_bidirectional(tracks, backward)
        self.tracks = {tid: self._densify_track(tid, frames) for tid, frames in tracks.items()}

    @staticmethod
    def _run_tracker(tracker, detections, frame, reverse=False):
        tracks = {}
        for fid in sorted(detections, reverse=reverse):
            for row in tracker.update(detections[fid], frame):
                x1, y1, x2, y2, track_id, score = row[:6]
                track = tracks.setdefault(int(track_id), {})
                if fid in track:
                    raise ValueError(f"Tracker returned ID {int(track_id)} twice in frame {fid}")
                track[fid] = ([x1, y1, x2 - x1, y2 - y1], float(score))
        return tracks

    @staticmethod
    def _merge_bidirectional(fwd, bwd, iou_thresh=0.6, min_overlap_ratio=0.3, max_bbox_jump=0.5):
        def iou(a, b):
            x1, y1 = max(a[0], b[0]), max(a[1], b[1])
            x2 = min(a[0] + a[2], b[0] + b[2])
            y2 = min(a[1] + a[3], b[1] + b[3])
            overlap = max(0, x2 - x1) * max(0, y2 - y1)
            union = a[2] * a[3] + b[2] * b[3] - overlap
            return overlap / union if union > 0 else 0

        def center_jump(box, reference):
            cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
            rx, ry = reference[0] + reference[2] / 2, reference[1] + reference[3] / 2
            diagonal = (reference[2] ** 2 + reference[3] ** 2) ** 0.5
            return ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5 / max(diagonal, 1)

        pairs = []
        for forward_id, forward in fwd.items():
            for backward_id, backward in bwd.items():
                shared = forward.keys() & backward.keys()
                if len(shared) >= max(3, min(len(forward), len(backward)) * min_overlap_ratio):
                    average = sum(iou(forward[fid][0], backward[fid][0]) for fid in sorted(shared)) / len(shared)
                    if average >= iou_thresh:
                        pairs.append((average, forward_id, backward_id))

        matched_fwd, matched_bwd = set(), set()
        for _, forward_id, backward_id in sorted(pairs, reverse=True):
            if forward_id in matched_fwd or backward_id in matched_bwd:
                continue
            matched_fwd.add(forward_id)
            matched_bwd.add(backward_id)
            forward, backward = fwd[forward_id], bwd[backward_id]
            original_frames = sorted(forward)
            for fid, record in backward.items():
                if fid in forward:
                    continue
                index = bisect_left(original_frames, fid)
                neighbors = original_frames[max(0, index - 1):index + 1]
                nearest = min(neighbors, key=lambda candidate: abs(candidate - fid))
                if center_jump(record[0], forward[nearest][0]) > max_bbox_jump:
                    continue
                forward[fid] = record

    @staticmethod
    def _densify_track(track_id, known):
        frames = sorted(known)
        samples = []
        for left, right in pairwise(frames):
            left_box, left_score = known[left]
            right_box, right_score = known[right]
            samples.append(known[left])
            for frame in range(left + 1, right):
                fraction = (frame - left) / (right - left)
                box = [a + fraction * (b - a) for a, b in zip(left_box, right_box)]
                score = left_score * (1 - fraction) + right_score * fraction
                samples.append((box, score))
        samples.append(known[frames[-1]])
        boxes, scores = zip(*samples)
        start, end = frames[0], frames[-1]
        return Track(track_id, np.arange(start, end + 1, dtype=np.int32),
                     np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32))

    def save(self, path: str):
        dump_detection(self, path)

    @classmethod
    def load(cls, path: str) -> 'Detection':
        return cls._loaded(load_detection(path))

    @classmethod
    def _loaded(cls, pack: MotionPackDetection):
        result = cls.__new__(cls)
        result.video_path, result.shape, result.tracks = pack.video_path, pack.shape, pack.tracks
        result.fps, result.n_frames, result.video_sha256 = pack.source_fps, pack.total_frames, pack.video_sha256
        return result


# ==================== Pose ====================

class Pose:
    """实例化即推理"""

    def __init__(self, detection: Detection, track_ids: list[int] | None = None,
                 device: str = 'cuda', batch_size: int = 4, read_batch: int = 16,
                 nlf_model=None):
        self.detection = detection
        tids = track_ids or list(detection.tracks.keys())

        if nlf_model is None:
            from nlf_pt_inference import NLFSMPLInference

            nlf_model = NLFSMPLInference(device=device, trt_batch_size=batch_size)
        nlf = nlf_model
        self.producer_model_sha256 = getattr(nlf, "producer_model_sha256", None)
        self.query_layout_id = getattr(nlf, "query_layout_id", None)
        validate_pose_pack_identity(
            self.producer_model_sha256,
            self.query_layout_id,
            POSE_METADATA,
        )
        self.metadata = dict(POSE_METADATA)

        from video_io import VideoReader
        from nlf_pt_ops import compute_intrinsics, generate_crops, restore_camera_coordinates

        intr = compute_intrinsics(*detection.shape, device=device)
        self.tracks = {}
        with VideoReader(detection.video_path, device=device) as reader:
            for tid in tids:
                t = detection.tracks[tid]
                n = len(t.frames)

                all_p3d, all_p2d, all_unc = [], [], []

                for i in range(0, n, read_batch):
                    fids = t.frames[i:i+read_batch].tolist()
                    boxes = t.bboxes[i:i+read_batch]
                    frames = reader.get_batch(fids)

                    boxes = torch.as_tensor(boxes, device=device)
                    xyxy = torch.cat((boxes[:, :2], boxes[:, :2] + boxes[:, 2:4]), dim=1)
                    crops, ci, R = generate_crops(frames, list(xyxy.split(1)), intr.expand(len(fids), -1, -1))
                    del frames

                    for j in range(0, len(crops), batch_size):
                        c, cii, r = crops[j:j+batch_size], ci[j:j+batch_size], R[j:j+batch_size]
                        with torch.inference_mode():
                            res = nlf(c)
                            fin = restore_camera_coordinates(**res, intrinsics=cii, rotation=r)
                        all_p3d.append(fin['poses3d'].cpu().numpy())
                        all_p2d.append(res['coords2d'].cpu().numpy())
                        all_unc.append(fin['uncertainties'].cpu().numpy())
                    del crops, ci, R

                p3d = np.concatenate(all_p3d)
                self.tracks[tid] = PoseTrack(
                    tid,
                    p3d,
                    np.concatenate(all_p2d),
                    np.concatenate(all_unc),
                    np.asarray(t.frames),
                )

    def save(self, path: str, det_path: str = None):
        """保存Pose数据，det_path为Detection文件路径（引用存储）"""
        dump_pose(self, path, det_path)

    @classmethod
    def load(cls, path: str, detection: Detection = None) -> 'Pose':
        """加载Pose，可传入Detection或从引用路径加载"""
        path = Path(path)
        return cls._loaded(_read_pose(path, _load_payload(path, "pose"), detection))

    @classmethod
    def _loaded(cls, pack: MotionPackPose):
        detection = pack.detection
        if isinstance(detection, MotionPackDetection):
            detection = Detection._loaded(detection)
        if detection is None and pack.detection_ref:
            raise FileNotFoundError(pack.source.detection_ref)
        result = cls.__new__(cls)
        result.__dict__.update(vars(pack), detection=detection)
        return result


# ==================== Body ====================

class Body:
    """实例化即拟合"""

    def __init__(self, pose: Pose, track_ids: list[int] | None = None,
                 device: str = 'cuda', batch_size: int = 16,
                 *, fitter=None):
        self.pose = pose
        tids = track_ids or list(pose.tracks.keys())
        validate_pose_pack_identity(
            getattr(pose, "producer_model_sha256", None),
            getattr(pose, "query_layout_id", None),
            getattr(pose, "metadata", None),
        )
        fitter = SMPLFitter(device) if fitter is None else fitter
        self.metadata = dict(BODY_METADATA)
        self.body_model_asset_identity = fitter.asset_identity
        self.smpl_vertex_subset_sha256 = fitter.subset_sha256
        self.joint_schema = fitter.joint_schema
        joint_count = len(SMPL_24_JOINT_SCHEMA["names"])

        self.tracks = {}

        for tid in tids:
            pt = normalize_pose_track(pose.tracks[tid], require_v2=True)
            n = len(pt.poses3d)

            source_frames = pt.source_frames
            detection_track = getattr(getattr(pose, "detection", None), "tracks", {}).get(tid)
            if detection_track is not None and not np.array_equal(source_frames, detection_track.frames):
                raise ValueError(f"PoseTrack {tid} source_frames do not match Detection Track.frames")

            if not n or pt.poses3d.dtype != np.float32 or pt.uncertainties.dtype != np.float32:
                raise ValueError("Fitting requires non-empty float32 poses3d and uncertainties")
            source_joint_uncertainties_mm = pt.uncertainties[:, -joint_count:]

            p3d = torch.tensor(pt.poses3d, device=device)
            unc = torch.tensor(pt.uncertainties, device=device)

            all_betas, all_pose, all_joints = [], [], []
            for i in range(0, n, batch_size):
                res = fitter(p3d[i:i+batch_size], unc[i:i+batch_size])
                all_betas.append(res['betas'].cpu().numpy())
                all_pose.append(res['pose_rotvecs'].cpu().numpy())
                all_joints.append(res['joints3d'].cpu().numpy())

            frame_betas = np.concatenate(all_betas)
            betas = np.median(frame_betas, axis=0)
            self.tracks[tid] = BodyTrack(
                id=tid,
                betas=betas,
                pose_rotvecs=np.concatenate(all_pose).reshape(n, 24, 3),
                joints=np.concatenate(all_joints),
                frame_betas=frame_betas,
                source_frames=source_frames,
                source_joint_uncertainties_mm=source_joint_uncertainties_mm,
            )

    def save(self, path: str, pose_path: str = None):
        """保存Body数据，pose_path为Pose文件路径（引用存储）"""
        dump_body(self, path, pose_path)

    @classmethod
    def load(cls, path: str, pose: Pose = None) -> 'Body':
        """加载Body，可传入Pose或从引用路径加载"""
        pack, linked_pose = _read_body(Path(path), with_pose=pose is None)
        if pose is None:
            pose = Pose._loaded(linked_pose) if isinstance(linked_pose, MotionPackPose) else linked_pose
            if pose is None and pack.pose_ref:
                raise FileNotFoundError(_resolve_ref(pack.pose_ref, path))
        result = cls.__new__(cls)
        result.__dict__.update(vars(pack), pose=pose)
        return result


class NoPersonTrackError(ValueError):
    """Automatic detection did not find a track to fit."""


def _build_pack(
    video_path: str | Path,
    output_dir: str | Path,
    track_ids: list[int] | None,
    max_tracks: int,
    *,
    on_stage: Callable[[str], None] | None,
    device: str,
    detector,
    nlf_model,
) -> MotionPackBody:
    video = Path(video_path).resolve(strict=True)
    output = Path(output_dir).absolute()
    det, pose, body = ("detection.det", "pose.pose", "body.body")
    with new_output_directory(output) as staging:
        video_sha256 = _file_sha256(video)
        if on_stage:
            on_stage("detection")
        detection = Detection(str(video), device=device, detector=detector)
        detection.video_sha256 = video_sha256
        if track_ids is None:
            ranked = sorted(detection.tracks, key=lambda tid: -len(detection.tracks[tid].frames))
            track_ids = ranked[:max_tracks]
            if not track_ids:
                raise NoPersonTrackError(f"No matching person tracks in {video.name}")
        if not track_ids or any(tid not in detection.tracks for tid in track_ids):
            raise ValueError(f"No matching person tracks in {video.name}")
        detection.save(str(staging / det))

        if on_stage:
            on_stage("pose")
        fitted_pose = Pose(detection, track_ids=track_ids, device=device, nlf_model=nlf_model)
        fitted_pose.save(str(staging / pose), det_path=det)

        if on_stage:
            on_stage("body")
        fitted_body = Body(fitted_pose, track_ids, device=device)
        stored = _write_body(fitted_body, staging / body, pose)
        if _file_sha256(video) != video_sha256:
            raise ValueError(f"Source video changed during inference: {video.name}")

    source = MotionPackSource(
        video_path=str(video), total_frames=detection.n_frames, source_fps=detection.fps,
        detection_ref=str((output / det).resolve()), video_sha256=video_sha256,
    )
    return MotionPackBody(path=output / body, source=source, **stored)


def build_artifacts(
    video_path: str | Path,
    output_dir: str | Path,
    track_ids: list[int] | None = None,
    max_tracks: int = 2,
    *,
    on_stage: Callable[[str], None] | None = None,
    device: str = "cuda",
    detector=None,
    nlf_model=None,
) -> dict:
    """Build a fresh MotionPack and return its artifact paths."""
    pack = _build_pack(
        video_path, output_dir, track_ids, max_tracks,
        on_stage=on_stage, device=device, detector=detector, nlf_model=nlf_model,
    )
    return {
        "detection_path": str(pack.path.with_name(Path(pack.source.detection_ref).name)),
        "pose_path": str(pack.path.with_name(pack.pose_ref)),
        "body_path": str(pack.path),
        "track_ids": track_ids if track_ids is not None else list(pack.tracks),
    }


def reconstruct_subject(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    track_id: int | None = None,
    on_stage: Callable[[str], None] | None = None,
    device: str = "cuda",
    detector=None,
    nlf_model=None,
) -> MotionPackBody:
    """Fit one subject, publish its MotionPack, and return the saved BodyPack data."""
    return _build_pack(
        video_path, output_dir, [track_id] if track_id is not None else None, 1,
        on_stage=on_stage, device=device, detector=detector, nlf_model=nlf_model,
    )
