"""Site data contract tests, independent of the live research experiment."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_site_data import pearson, read_checkpoints, ensure


class SiteDataContractTests(unittest.TestCase):
    def test_perfect_and_reversed_correlation(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [1, 2, 3]), 1)
        self.assertAlmostEqual(pearson([1, 2, 3], [3, 2, 1]), -1)

    def test_mismatched_inputs_rejected(self):
        with self.assertRaises(ValueError):
            pearson([1, 2, 3], [1, 2])

    def test_no_synthetic_intermediate_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoints.csv"
            path.write_text("0,1,2,3\n50,3,2,1\n", encoding="utf-8")
            rows = read_checkpoints(path, 3, [1, 2, 3])
            self.assertEqual([r["epoch"] for r in rows], [0, 50])
            self.assertAlmostEqual(rows[-1]["correlation"], -1)

    def test_fails_on_bad_checkpoint_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoints.csv"
            path.write_text("0,1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_checkpoints(path, 3, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
