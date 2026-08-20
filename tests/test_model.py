import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import create_model, load_trained_model  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(REPO_ROOT, "weights", "svamitva_refined_v2_e5.pth")


class TestModel(unittest.TestCase):
    def test_output_shape(self):
        model = create_model()
        model.eval()
        x = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(y.shape, (1, 5, 64, 64))

    def test_default_class_count_matches_checkpoint(self):
        model = create_model()
        head = model.segmentation_head[0]
        self.assertEqual(head.out_channels, 5)

    @unittest.skipUnless(os.path.exists(WEIGHTS_PATH), "trained weights not present")
    def test_checkpoint_loads_strict(self):
        model = load_trained_model(WEIGHTS_PATH)
        x = torch.zeros(1, 3, 512, 512)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(y.shape, (1, 5, 512, 512))


if __name__ == "__main__":
    unittest.main()
