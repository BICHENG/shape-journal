import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


class VideoReaderTest(unittest.TestCase):
    def make_reader(self, read, *, count=3):
        from video_io import VideoReader

        class Decoder:
            def __len__(self):
                return count

            def get_width(self):
                return 2

            def get_height(self):
                return 2

            def get_avg_fps(self):
                return 30.0

            def __getitem__(self, index):
                return SimpleNamespace(asnumpy=lambda: read(index))

        decord = SimpleNamespace(
            VideoReader=lambda *_args, **_kwargs: Decoder(),
            cpu=lambda _: "cpu",
        )
        with patch.dict(sys.modules, {"decord": decord}):
            return VideoReader("example.mp4", device="cpu")

    def test_decode_error_preserves_cause_and_ends_iteration(self):
        from video_io import VideoDecodeError

        def read(index):
            if index == 1:
                raise OSError("truncated frame")
            return np.full((2, 2, 3), 7, dtype=np.uint8)

        with self.make_reader(read) as reader:
            frame_id, frame = next(reader)
            self.assertEqual(frame_id, 0)
            self.assertEqual(frame.tolist(), [[[7, 7], [7, 7]]] * 3)
            with self.assertRaises(VideoDecodeError) as failure:
                next(reader)
            self.assertIsInstance(failure.exception.__cause__, OSError)
            self.assertEqual(str(failure.exception.__cause__), "truncated frame")
            with self.assertRaises(StopIteration):
                next(reader)

    def test_opencv_batch_reads_the_last_frame_when_seeking_fails(self):
        import cv2

        from video_io import VideoReader

        captures = []

        def open_clip(_):
            capture = Mock()
            capture.get.side_effect = {cv2.CAP_PROP_FRAME_WIDTH: 2, cv2.CAP_PROP_FRAME_HEIGHT: 2,
                                       cv2.CAP_PROP_FPS: 30.0, cv2.CAP_PROP_FRAME_COUNT: 3}.__getitem__
            capture.set.return_value = False
            capture.read.return_value = (False, None)
            capture.grab.side_effect = [True, True, True, False]
            capture.retrieve.side_effect = [
                (True, np.full((2, 2, 3), value, np.uint8)) for value in (1, 3)
            ]
            captures.append(capture)
            return capture

        with patch.dict(sys.modules, {"decord": None}), patch.object(cv2, "VideoCapture", open_clip):
            with self.assertWarns(RuntimeWarning), VideoReader("example.mp4", device="cpu") as reader:
                self.assertEqual(reader.get_batch([2, 0, 2])[:, 0, 0, 0].tolist(), [3, 1, 3])
                frame_id, frame = next(reader)
                self.assertEqual((frame_id, frame[0, 0, 0].item()), (0, 1))
        self.assertEqual(len(captures), 3)
        for capture in captures:
            capture.release.assert_called_once()
        with self.assertRaises(StopIteration):
            next(reader)
        with self.assertRaisesRegex(ValueError, "已关闭"):
            reader.get_frame(0)


if __name__ == "__main__":
    unittest.main()
