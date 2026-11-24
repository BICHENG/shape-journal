import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from body_measurement.shape_tracking import ShapeObservation, compare_shapes, estimate_visit
from core.motionpack import BodyTrack, MotionPackBody, MotionPackSource
from core.motionpack.model import BODY_MODEL_ASSET_IDENTITY


def observation(source, beta, **fields):
    return replace(
        ShapeObservation(
            subject_id="person-1",
            visit_id="visit-1",
            protocol_id="setup-v1",
            video_sha256=source * 64,
            body_model_asset_sha256=BODY_MODEL_ASSET_IDENTITY["asset_sha256"],
            track_id=1,
            betas=(beta,) * 10,
        ),
        **fields,
    )


class ShapeTrackingTest(unittest.TestCase):
    def test_observation_roundtrip(self):
        original = observation("a", 0.5)
        record = json.loads(original.to_json())
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["betas"], [0.5] * 10)
        self.assertEqual(ShapeObservation.from_json(original.to_json()), original)
        record["betas"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            ShapeObservation.from_json(json.dumps(record))

    def test_bodypack_ingress_requires_unambiguous_provenance(self):
        tracks = {
            i: BodyTrack(i, np.zeros(10), np.zeros((1, 24, 3)), np.zeros((1, 24, 3)))
            for i in (1, 2)
        }
        body = MotionPackBody(
            path=Path("body.body"),
            tracks=tracks,
            schema_version=3,
            body_model_asset_identity=BODY_MODEL_ASSET_IDENTITY,
            smpl_vertex_subset_sha256=None,
            joint_schema=None,
            metadata=None,
            pose_ref=None,
            pose_sha256=None,
            source=MotionPackSource(),
        )
        context = dict(subject_id="person-1", visit_id="visit-1", protocol_id="setup-v1")
        with self.assertRaisesRegex(ValueError, "verified source"):
            ShapeObservation.from_body(body, track_id=1, **context)
        with self.assertRaisesRegex(ValueError, "select a track"):
            ShapeObservation.from_body(body, video_sha256="a" * 64, **context)
        record = ShapeObservation.from_body(body, track_id=2, video_sha256="a" * 64, **context)
        self.assertEqual(record.track_id, 2)
        self.assertEqual(record.video_sha256, "a" * 64)

    def test_visit_median_gives_each_distinct_recording_one_vote(self):
        first, second, outlier = observation("a", 0), observation("b", 2), observation("c", 9)
        estimate = estimate_visit([first, second, outlier, first, first])
        self.assertEqual(estimate.betas, (2.0,) * 10)
        self.assertEqual(estimate.video_sha256s, ("a" * 64, "b" * 64, "c" * 64))
        self.assertEqual(estimate_visit([second, first]).betas, (1.0,) * 10)
        with self.assertRaisesRegex(ValueError, "conflicting observations"):
            estimate_visit([first, observation("a", 9)])

    def test_visit_cannot_mix_subject_visit_protocol_or_model(self):
        first = observation("a", 0)
        for field, other in (
            ("subject_id", "person-2"),
            ("visit_id", "visit-2"),
            ("protocol_id", "setup-v2"),
            ("body_model_asset_sha256", "a" * 64),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "same subject"):
                estimate_visit([first, observation("b", 2, **{field: other})])

    def test_compare_rejects_a_reused_recording_or_different_subject(self):
        first = estimate_visit([observation("a", 0)])
        reused = estimate_visit([observation("a", 0, visit_id="visit-2")])
        other = estimate_visit([observation("b", 0, subject_id="person-2")])
        with self.assertRaisesRegex(ValueError, "share a recording"):
            compare_shapes(first, reused, height_cm=163)
        with self.assertRaisesRegex(ValueError, "same subject"):
            compare_shapes(first, other, height_cm=163)
