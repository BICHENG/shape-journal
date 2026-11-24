import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np

from core.motionpack import BodyTrack, PoseTrack, Track, load_artifacts, load_body
from core.motionpack.io import BODY_SCHEMA_VERSION, BODY_METADATA, PORTABLE_BODY_PREFIX, dump_body, dump_detection, dump_pose
from core.motionpack.loading import load_pose
from core.motionpack.model import (
    BODY_MODEL_ASSET_IDENTITY,
    POSE_METADATA,
    PRODUCTION_NLF_MODEL_SHA256,
    PRODUCTION_NLF_QUERY_LAYOUT_ID,
    SMPL_24_JOINT_SCHEMA,
    SMPL_24_JOINT_SIDES,
    SMPL_VERTEX_SUBSET_SHA256,
)


class MotionPackLoadingTest(unittest.TestCase):
    @staticmethod
    def _pose_pack(tracks, detection):
        return SimpleNamespace(
            tracks=tracks,
            detection=detection,
            producer_model_sha256=PRODUCTION_NLF_MODEL_SHA256,
            query_layout_id=PRODUCTION_NLF_QUERY_LAYOUT_ID,
            metadata=dict(POSE_METADATA),
        )

    def _capture(self, frames_by_track, **source):
        tracks, poses = {}, {}
        for tid, frame_ids in frames_by_track.items():
            frames = np.asarray(frame_ids, dtype=np.int32)
            count = len(frames)
            tracks[tid] = Track(tid, frames, np.zeros((count, 4), np.float32), np.ones(count, np.float32))
            poses[tid] = PoseTrack(tid, np.zeros((count, 1048, 3), np.float32),
                                   np.zeros((count, 1048, 2), np.float32), np.ones((count, 1048), np.float32), frames)
        return self._pose_pack(poses, SimpleNamespace(tracks=tracks, **source))

    @staticmethod
    def _body_pack(tracks):
        return SimpleNamespace(
            tracks=tracks,
            joint_schema=SMPL_24_JOINT_SCHEMA,
            metadata=dict(BODY_METADATA),
            body_model_asset_identity=dict(BODY_MODEL_ASSET_IDENTITY),
            smpl_vertex_subset_sha256=SMPL_VERTEX_SUBSET_SHA256,
        )

    @staticmethod
    def _body_track(track_id, frames):
        frames = np.asarray(frames, np.int32)
        n = len(frames)
        return BodyTrack(
            track_id,
            np.zeros(10, np.float32),
            np.zeros((n, 24, 3), np.float32),
            np.zeros((n, 24, 3), np.float32),
            source_frames=frames,
            source_joint_uncertainties_mm=np.ones((n, 24), np.float32),
        )

    def test_smpl_joint_sides_follow_joint_names(self):
        self.assertEqual(
            SMPL_24_JOINT_SIDES,
            tuple(
                "left" if name.startswith("left_") else "right" if name.startswith("right_") else "center"
                for name in SMPL_24_JOINT_SCHEMA["names"]
            ),
        )

    def test_pose_resolves_relative_detection_inside_its_pack(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            relative = Path(root.name) / "detection.det"
            pose_path = root / "pack" / "pose.pose"
            detection_path = pose_path.parent / relative
            detection_path.parent.mkdir(parents=True)
            detection = SimpleNamespace(
                video_path="right.mp4", fps=30.0, n_frames=10, shape=(720, 1280),
                tracks={}, video_sha256="a" * 64,
            )
            dump_detection(detection, str(detection_path))
            dump_detection(SimpleNamespace(**{**vars(detection), "video_path": "wrong.mp4"}), str(root / "detection.det"))
            dump_pose(self._pose_pack({}, detection), str(pose_path), det_path=str(relative))

            pose = load_pose(pose_path)
            self.assertEqual(pose.source.video_path, "right.mp4")
            self.assertEqual(pose.source.video_sha256, "a" * 64)

    def test_load_body_consumes_artifact_without_execution_stack(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("core.motionpack.builders", sys.modules)

        tracks = {
            1: self._body_track(1, [2, 4]),
            3: self._body_track(3, [3, 5, 7, 9]),
        }
        tracks[3].betas[:] = 1

        with tempfile.TemporaryDirectory() as tmp:
            body_path = Path(tmp) / "sample.body"
            det_path = Path(tmp) / "sample.det"
            pose_path = Path(tmp) / "sample.pose"
            pose = self._capture(
                {tid: track.source_frames for tid, track in tracks.items()},
                video_path="video.mp4", fps=29.97, shape=(640, 480), n_frames=42,
            )
            dump_detection(pose.detection, str(det_path))
            dump_pose(pose, str(pose_path), det_path="sample.det")
            dump_body(self._body_pack(tracks), str(body_path), pose_path="sample.pose")

            body = load_body(body_path)
            expected_pose_sha256 = hashlib.sha256(pose_path.read_bytes()).hexdigest()

        self.assertEqual(body.track_ids(), (1, 3))
        self.assertEqual(body.schema_version, BODY_SCHEMA_VERSION)
        self.assertEqual(body.body_model_asset_identity, BODY_MODEL_ASSET_IDENTITY)
        self.assertEqual(body.smpl_vertex_subset_sha256, SMPL_VERTEX_SUBSET_SHA256)
        self.assertEqual(body.joint_schema, SMPL_24_JOINT_SCHEMA)
        self.assertEqual(body.metadata, BODY_METADATA)
        self.assertEqual(body.pose_ref, "sample.pose")
        self.assertEqual(body.pose_sha256, expected_pose_sha256)
        summary = body.summary()
        self.assertEqual(summary["pose_sha256"], expected_pose_sha256)
        self.assertEqual(body.source.video_path, "video.mp4")
        self.assertEqual(body.source.total_frames, 42)
        self.assertAlmostEqual(body.source.source_fps, 29.97)
        self.assertEqual(body.select_track().id, 3)
        self.assertEqual(body.select_track(1).betas.shape, (10,))
        self.assertEqual(summary["source"]["total_frames"], 42)
        self.assertEqual(summary["metadata"]["pose_shape"], [24, 3])
        self.assertEqual(summary["metadata"]["units"], "mm")
        self.assertEqual(
            summary["metadata"]["coordinate_system"],
            "camera_cv_rh_x_right_y_down_z_forward",
        )
        self.assertEqual(summary["tracks"][3]["pose_rotvecs_shape"], [4, 24, 3])
        self.assertEqual(summary["tracks"][3]["joints_shape"], [4, 24, 3])
        self.assertEqual(summary["tracks"][3]["source_frames_shape"], [4])
        self.assertEqual(summary["tracks"][3]["source_joint_uncertainties_shape"], [4, 24])
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("core.motionpack.builders", sys.modules)

    def test_load_artifacts_returns_the_pose_verified_by_body(self):
        tracks = {2: self._body_track(2, [10, 11, 12])}
        pose = self._capture({2: [10, 11, 12]}, video_path="video.mp4", fps=30.0, shape=(640, 480), n_frames=13)
        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = Path(tmp)
            dump_detection(pose.detection, str(trial_dir / "detection.det"))
            linked_pose = (trial_dir / "subject.pose").resolve()
            dump_pose(pose, str(linked_pose), det_path="detection.det")
            dump_body(self._body_pack(tracks), str(trial_dir / "body.body"), pose_path=linked_pose.name)
            (trial_dir / "pose.pose").write_bytes(b"unrelated sibling")

            artifacts = load_artifacts(trial_dir)

        self.assertEqual(artifacts.pose.path, linked_pose)
        self.assertEqual(artifacts.body.source, artifacts.pose.source)
        self.assertEqual(artifacts.pose.tracks[2].poses3d.shape, (3, 1048, 3))
        self.assertEqual(artifacts.pose.producer_model_sha256, PRODUCTION_NLF_MODEL_SHA256)
        self.assertEqual(artifacts.pose.query_layout_id, PRODUCTION_NLF_QUERY_LAYOUT_ID)
        self.assertEqual(artifacts.pose.metadata, POSE_METADATA)
        self.assertEqual(artifacts.track_frame_ranges()[2], {"start_frame": 10, "end_frame": 12})
        self.assertNotIn("core.motionpack.builders", sys.modules)

    def test_legacy_script_pickles_are_remapped_to_data_models(self):
        original_module = BodyTrack.__module__
        legacy_module = ModuleType("scripts.run_to_build_motion_pack")
        with tempfile.TemporaryDirectory() as tmp:
            legacy_module.BodyTrack = BodyTrack
            sys.modules[legacy_module.__name__] = legacy_module
            BodyTrack.__module__ = legacy_module.__name__
            try:
                body_path = Path(tmp) / "legacy.body"
                track = BodyTrack(7, np.zeros(10), np.zeros((1, 72)), np.zeros((1, 24, 3)))
                joblib.dump({"tracks": {7: track}, "pose_ref": "missing.pose"}, body_path)
            finally:
                BodyTrack.__module__ = original_module
                del sys.modules[legacy_module.__name__]

            body = load_body(body_path)

        self.assertEqual(type(body.select_track()).__module__, "core.motionpack.model")
        self.assertEqual(body.schema_version, 0)
        self.assertIsNone(body.joint_schema)
        self.assertIsNone(body.metadata)
        self.assertEqual(body.select_track().pose_rotvecs.shape, (1, 24, 3))
        self.assertIsNone(body.select_track().source_frames)
        self.assertIsNone(body.select_track().source_joint_uncertainties_mm)
        self.assertNotIn("scripts.run_to_build_motion_pack", sys.modules)
        self.assertNotIn("core.motionpack.builders", sys.modules)

    def test_measurement_smoke_rejects_legacy_body_without_inventing_identity(self):
        from body_measurement.scripts.check_body_smoke import run_smoke

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body_path = root / "legacy.body"
            joblib.dump(
                {"tracks": {1: BodyTrack(1, np.zeros(10), np.zeros((1, 72)), np.zeros((1, 24, 3)))}},
                body_path,
            )

            self.assertEqual(load_body(body_path).schema_version, 0)
            with self.assertRaisesRegex(ValueError, "rebuild PosePack and BodyPack"):
                run_smoke(body_path, root / "output", 175.0)
            self.assertFalse((root / "output").exists())

    def test_body_v2_remains_readable_without_pose_digest(self):
        track = self._body_track(1, [3, 5])
        payload = {
            "kind": "body",
            "schema_version": 2,
            "body_model_asset_identity": BODY_MODEL_ASSET_IDENTITY,
            "smpl_vertex_subset_sha256": SMPL_VERTEX_SUBSET_SHA256,
            "joint_schema": SMPL_24_JOINT_SCHEMA,
            "metadata": BODY_METADATA,
            "tracks": {1: track},
            "pose_ref": "existing.pose",
        }
        with tempfile.TemporaryDirectory() as tmp:
            body_path = Path(tmp) / "v2.body"
            joblib.dump(payload, body_path)
            body = load_body(body_path)

        self.assertEqual(body.schema_version, 2)
        self.assertEqual(body.pose_ref, "existing.pose")
        self.assertIsNone(body.pose_sha256)
        np.testing.assert_array_equal(body.tracks[1].source_frames, track.source_frames)

    def test_portable_body_loads_bound_records_without_sidecars(self):
        capture = self._capture({1: [2]}, video_path="video.mp4", fps=30.0, shape=(640, 480), n_frames=3)
        capture.detection.tracks[1].bboxes[0, 2:] = [140, 300]
        track = self._body_track(1, [2])
        track.frame_betas = np.zeros((1, 10), np.float32)

        def encode(data):
            return json.dumps(data, default=lambda array: array.tolist()).encode("utf-8")

        detection = encode({
            "kind": "detection", "schema_version": 1, "video": "video.mp4", "video_sha256": "a" * 64,
            "fps": 30.0, "shape": [640, 480], "n_frames": 3, "tracks": {1: vars(capture.detection.tracks[1])},
        })
        pose = encode({
            "kind": "pose", "schema_version": 2,
            "det_sha256": hashlib.sha256(detection).hexdigest(),
            "producer_model_sha256": PRODUCTION_NLF_MODEL_SHA256, "query_layout_id": PRODUCTION_NLF_QUERY_LAYOUT_ID,
            **POSE_METADATA, "tracks": {1: vars(capture.tracks[1])},
        })
        body = encode({
            "kind": "body", "schema_version": BODY_SCHEMA_VERSION,
            "pose_ref": "pose.pose", "pose_sha256": hashlib.sha256(pose).hexdigest(),
            **vars(self._body_pack({1: vars(track)})),
        })

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "body.body"
            (root / "pose.pose").write_bytes(b"unrelated sibling")
            path.write_bytes(PORTABLE_BODY_PREFIX + gzip.compress(b"\n".join((body, detection, pose))))
            actual = load_artifacts(root)
            self.assertEqual(actual.body.source.video_sha256, "a" * 64)
            self.assertEqual(actual.pose.detection.shape, (640, 480))
            self.assertEqual(actual.track_frame_ranges(), {1: {"start_frame": 2, "end_frame": 2}})
            self.assertEqual(actual.pose.tracks[1].poses3d.dtype, np.float32)
            self.assertEqual(load_body(path).select_track().joints.dtype, np.float32)

            for records, field in (((body, detection, pose + b" "), "pose_sha256"),
                                   ((body, detection + b" ", pose), "det_sha256")):
                path.write_bytes(PORTABLE_BODY_PREFIX + gzip.compress(b"\n".join(records)))
                with self.assertRaisesRegex(ValueError, field):
                    load_body(path)

    def test_body_v3_rejects_changed_pose_content(self):
        pose = self._capture({1: [3]})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pose_path = root / "pose.pose"
            body_path = root / "body.body"
            dump_pose(pose, str(pose_path))
            dump_body(self._body_pack({1: self._body_track(1, [3])}), str(body_path), pose_path="pose.pose")
            pose_path.write_bytes(pose_path.read_bytes() + b"changed")

            with self.assertRaisesRegex(ValueError, "pose_sha256"):
                load_body(body_path)

    def test_body_v3_writer_rejects_pose_track_frame_mismatch(self):
        pose = self._capture({1: [3]})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dump_pose(pose, str(root / "pose.pose"))
            with self.assertRaisesRegex(ValueError, "source_frames do not match PosePack"):
                dump_body(
                    self._body_pack({1: self._body_track(1, [4])}),
                    str(root / "body.body"),
                    pose_path="pose.pose",
                )

    def test_legacy_nlf_motion_pack_v3_uses_data_models_and_validation(self):
        legacy_module = ModuleType("nlf.motion_pack_v3")
        classes = {
            name: type(name, (), {"__module__": legacy_module.__name__})
            for name in ("Pose", "Detection", "Track", "PoseTrack", "BodyTrack")
        }
        for name, cls in classes.items():
            setattr(legacy_module, name, cls)

        def legacy(name, **state):
            value = classes[name]()
            vars(value).update(state)
            return value

        detection = legacy(
            "Detection",
            video_path="legacy.mp4",
            fps=25.0,
            n_frames=9,
            shape=(480, 640),
            tracks={1: legacy("Track", id=1)},
        )
        pose = legacy(
            "Pose",
            detection=detection,
            tracks={1: legacy("PoseTrack", id=1)},
        )

        def body_track(rotvec_width):
            return legacy(
                "BodyTrack",
                id=1,
                betas=np.zeros(10, np.float32),
                pose_rotvecs=np.zeros((2, rotvec_width), np.float32),
                joints=np.zeros((2, 24, 3), np.float32),
            )

        with tempfile.TemporaryDirectory() as tmp:
            valid_path = Path(tmp) / "legacy-nlf.body"
            invalid_path = Path(tmp) / "invalid-legacy-nlf.body"
            sys.modules[legacy_module.__name__] = legacy_module
            try:
                joblib.dump({"tracks": {1: body_track(72)}, "pose": pose}, valid_path)
                joblib.dump({"tracks": {1: body_track(71)}, "pose": pose}, invalid_path)
            finally:
                del sys.modules[legacy_module.__name__]

            body = load_body(valid_path)
            self.assertEqual(type(body.select_track()), BodyTrack)
            self.assertEqual(body.select_track().pose_rotvecs.shape, (2, 24, 3))
            self.assertEqual(body.source.video_path, "legacy.mp4")
            self.assertEqual(body.source.total_frames, 9)
            self.assertEqual(body.source.source_fps, 25.0)
            with self.assertRaisesRegex(ValueError, "pose_rotvecs"):
                load_body(invalid_path)

        self.assertNotIn("nlf.motion_pack_v3", sys.modules)
        self.assertNotIn("core.motionpack.builders", sys.modules)

    def test_legacy_dict_track_is_normalized_without_inventing_provenance(self):
        track = {
            "id": 4,
            "betas": np.zeros(10),
            "pose_rotvecs": np.zeros((2, 72)),
            "joints": np.zeros((2, 24, 3)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            body_path = Path(tmp) / "legacy-dict.body"
            joblib.dump(
                {"schema_version": 1, "joint_schema": SMPL_24_JOINT_SCHEMA, "tracks": {4: track}},
                body_path,
            )
            body = load_body(body_path)

        self.assertEqual(body.schema_version, 1)
        self.assertEqual(body.select_track().pose_rotvecs.shape, (2, 24, 3))
        self.assertIsNone(body.joint_schema)
        self.assertIsNone(body.metadata)
        self.assertIsNone(body.select_track().source_frames)
        self.assertIsNone(body.select_track().source_joint_uncertainties_mm)

    def test_body_v2_rejects_invalid_schema_metadata_and_track_arrays(self):
        track = self._body_track(1, [1, 2])
        valid = dict(vars(self._body_pack({1: track})), kind="body", schema_version=2)
        bad_uncertainty = track.source_joint_uncertainties_mm.copy()
        bad_uncertainty[0, 0] = np.nan
        malformed_tracks = (
            replace(track, source_frames=None, source_joint_uncertainties_mm=None),
            replace(track, pose_rotvecs=np.zeros((2, 72))),
            replace(track, pose_rotvecs=np.zeros((2, 23, 3))),
            replace(track, source_frames=np.array([1], np.int32)),
            replace(track, source_frames=np.array([2, 1], np.uint32)),
            replace(track, frame_betas=np.zeros((2, 9))),
            replace(track, source_joint_uncertainties_mm=bad_uncertainty),
        )
        cases = [
            {**valid, "joint_schema": {**SMPL_24_JOINT_SCHEMA, "names": SMPL_24_JOINT_SCHEMA["names"][:-1]}},
            {**valid, "metadata": {**BODY_METADATA, "units": "m"}},
            {key: value for key, value in valid.items() if key != "metadata"},
            {**valid, "body_model_asset_identity": {**BODY_MODEL_ASSET_IDENTITY, "model_sha256": "0" * 64}},
            {**valid, "smpl_vertex_subset_sha256": "0" * 64},
            *({**valid, "tracks": {1: malformed}} for malformed in malformed_tracks),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for index, case in enumerate(cases):
                body_path = Path(tmp) / f"invalid-{index}.body"
                joblib.dump(case, body_path)
                with self.subTest(index=index), self.assertRaises((TypeError, ValueError)):
                    load_body(body_path)

    def test_pose_v2_rejects_wrong_identity_and_missing_source_frames(self):
        track = self._capture({1: [1, 2]}).tracks[1]
        base = {
            "kind": "pose",
            "schema_version": 2,
            "producer_model_sha256": PRODUCTION_NLF_MODEL_SHA256,
            "query_layout_id": PRODUCTION_NLF_QUERY_LAYOUT_ID,
            **POSE_METADATA,
        }
        cases = [
            {**base, "producer_model_sha256": "0" * 64, "tracks": {1: track}},
            {**base, "query_layout_id": "0" * 64, "tracks": {1: track}},
            {**base, "points2d_coordinate_system": "source_video_pixels", "tracks": {1: track}},
            {**base, "tracks": {1: replace(track, source_frames=None)}},
            {**base, "tracks": {1: replace(track, source_frames=np.array([2, 1], np.uint32))}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for index, case in enumerate(cases):
                path = Path(tmp) / f"invalid-{index}.pose"
                joblib.dump(case, path)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    load_pose(path)

    def test_legacy_pose_keeps_identity_and_source_frames_missing(self):
        track = PoseTrack(
            1,
            np.zeros((1, 1048, 3), np.float32),
            np.zeros((1, 1048, 2), np.float32),
            np.zeros((1, 1048), np.float32),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pose"
            joblib.dump({"kind": "pose", "schema_version": 1, "tracks": {1: track}}, path)
            pose = load_pose(path)

        self.assertIsNone(pose.producer_model_sha256)
        self.assertIsNone(pose.query_layout_id)
        self.assertIsNone(pose.metadata)
        self.assertIsNone(pose.tracks[1].source_frames)

    def test_writers_require_explicit_pack_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                dump_pose(SimpleNamespace(tracks={}, detection=None), str(Path(tmp) / "bad.pose"))
            track = PoseTrack(
                1,
                np.zeros((1, 1048, 3), np.float32),
                np.zeros((1, 1048, 2), np.float32),
                np.ones((1, 1048), np.float32),
                np.array([2], np.int32),
            )
            detection = SimpleNamespace(tracks={1: SimpleNamespace(frames=np.array([1], np.int32))})
            with self.assertRaisesRegex(ValueError, "Detection Track.frames"):
                dump_pose(self._pose_pack({1: track}, detection), str(Path(tmp) / "misaligned.pose"))
            with self.assertRaises(ValueError):
                dump_body(SimpleNamespace(tracks={}), str(Path(tmp) / "bad.body"))
            body = self._body_pack({1: self._body_track(1, [1, 2])})
            body.tracks[1].pose_rotvecs = np.zeros((2, 72))
            with self.assertRaises(ValueError):
                dump_body(body, str(Path(tmp) / "legacy-shape.body"))

    def test_z_body_builder_rejects_wrong_pose_identity_before_fitting(self):
        from core.motionpack.builders import Body

        pose = SimpleNamespace(
            tracks={},
            producer_model_sha256=PRODUCTION_NLF_MODEL_SHA256,
            query_layout_id="0" * 64,
            metadata=dict(POSE_METADATA),
        )
        with self.assertRaisesRegex(ValueError, "query_layout_id"):
            Body(pose, fitter=object())

    def test_z_body_builder_stores_smpl_joints_and_source_uncertainties(self):
        import torch

        from core.motionpack.builders import Body

        frames = np.array([4], np.int32)
        track = PoseTrack(
            1,
            np.zeros((1, 1048, 3), np.float32),
            np.zeros((1, 1048, 2), np.float32),
            np.arange(1, 1049, dtype=np.float32).reshape(1, 1048),
            frames,
        )
        pose = self._pose_pack(
            {1: track},
            SimpleNamespace(tracks={1: SimpleNamespace(frames=frames)}),
        )
        from unittest.mock import Mock

        fitter = Mock(
            asset_identity=BODY_MODEL_ASSET_IDENTITY,
            subset_sha256=SMPL_VERTEX_SUBSET_SHA256,
            joint_schema=SMPL_24_JOINT_SCHEMA,
            return_value={
                "betas": torch.zeros((1, 10)),
                "pose_rotvecs": torch.zeros((1, 24, 3)),
                "joints3d": torch.full((1, 24, 3), 42.0),
            },
        )
        body = Body(pose, device="cpu", fitter=fitter)

        points, uncertainties = fitter.call_args.args
        np.testing.assert_array_equal(points.numpy(), track.poses3d)
        np.testing.assert_array_equal(uncertainties.numpy(), track.uncertainties)
        result = body.tracks[1]
        np.testing.assert_array_equal(result.source_frames, [4])
        np.testing.assert_array_equal(result.source_joint_uncertainties_mm, np.arange(1025, 1049)[None])
        np.testing.assert_array_equal(result.joints, np.full((1, 24, 3), 42.0))

    def test_z_pose_builder_requires_verified_runtime_identity(self):
        from core.motionpack.builders import Pose

        detection = SimpleNamespace(tracks={})
        with self.assertRaisesRegex(ValueError, "producer_model_sha256"):
            Pose(detection, nlf_model=SimpleNamespace(model_type="main"))
        with self.assertRaisesRegex(ValueError, "query_layout_id"):
            Pose(detection, nlf_model=SimpleNamespace(
                model_type="main",
                producer_model_sha256=PRODUCTION_NLF_MODEL_SHA256,
            ))

    def test_z_runtime_artifacts_match_snapshot_and_reject_bad_digest(self):
        import copy
        import json

        from nlf_pt_inference import (
            MAIN_RUNTIME_WEIGHT_FILES,
            NLF_METADATA_SNAPSHOT_PATH,
            runtime_artifacts_metadata,
            verify_main_runtime_weights,
            verify_smpl_runtime_artifact,
        )

        root = Path(__file__).resolve().parents[1]
        weights = root / "weights" / "main"
        canonical_path = root / "output" / "canonical_smpl.npy"
        if not weights.exists() and not canonical_path.exists():
            self.skipTest("The model fingerprint check needs the local NLF assets.")
        snapshot, digest = verify_main_runtime_weights(weights)
        identity = verify_smpl_runtime_artifact(
            canonical_path,
            np.load(canonical_path, allow_pickle=False),
            snapshot,
        )
        self.assertEqual(identity, (
            PRODUCTION_NLF_MODEL_SHA256,
            PRODUCTION_NLF_QUERY_LAYOUT_ID,
            digest,
        ))

        bad_snapshot = copy.deepcopy(json.loads(NLF_METADATA_SNAPSHOT_PATH.read_text(encoding="utf-8")))
        bad_snapshot["runtime_artifacts"]["digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            verify_main_runtime_weights(root / "weights" / "main", bad_snapshot)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            weights_dir = tmp / "weights"
            weights_dir.mkdir()
            for name in MAIN_RUNTIME_WEIGHT_FILES:
                (weights_dir / name).write_bytes(name.encode("ascii"))
            canonical_path = tmp / "canonical_smpl.npy"
            np.save(canonical_path, np.zeros((1048, 3), np.float32))
            generated = {
                "runtime_artifacts": runtime_artifacts_metadata(weights_dir, canonical_path)
            }
            verify_main_runtime_weights(weights_dir, generated)
            (weights_dir / "field.pth").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "main runtime weights"):
                verify_main_runtime_weights(weights_dir, generated)


if __name__ == "__main__":
    unittest.main()
