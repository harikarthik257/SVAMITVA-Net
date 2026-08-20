import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audit_palette import get_palette  # noqa: E402


class TestPalette(unittest.TestCase):
    def test_five_classes_indexed_zero_to_four(self):
        palette = get_palette()
        self.assertEqual(set(palette.keys()), {0, 1, 2, 3, 4})

    def test_background_is_class_zero(self):
        palette = get_palette()
        self.assertEqual(palette[0]["label"], "Background")

    def test_every_entry_has_a_valid_rgb_triple(self):
        palette = get_palette()
        for class_id, info in palette.items():
            rgb = info["rgb"]
            self.assertEqual(len(rgb), 3)
            for channel in rgb:
                self.assertTrue(0 <= channel <= 255)


if __name__ == "__main__":
    unittest.main()
