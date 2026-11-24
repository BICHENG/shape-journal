import subprocess
import sys
import unittest
from textwrap import dedent


class CameraFrameTest(unittest.TestCase):
    def test_vertical_view_keeps_a_finite_camera_frame(self):
        script = dedent("""
            import torch
            from nlf.pt.ptu3d import lookat_matrix

            forward = torch.tensor([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
            torch.testing.assert_close(lookat_matrix(forward, forward), torch.tensor([
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            ]))
        """)
        subprocess.run([sys.executable, "-c", script], check=True)
