import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from demo import synthetic_record
from reproduce import recompute
from run_eswa_task_evidence import probability, read_jsonl, render_webarena
from check_earlier_audit import check


class SnapshotTests(unittest.TestCase):
    def test_renderer_preserves_repeated_current_pages_and_missing_input(self):
        row = render_webarena(synthetic_record())
        self.assertEqual(row["snapshot_messages_removed"], 1)
        self.assertEqual(row["missing_input_steps"], 1)
        self.assertEqual(sum(e["source"] == "environment_observation_before_action" for e in row["events"]), 2)
        self.assertTrue(any(e["source"] == "missing_input_in_source" for e in row["events"]))

    def test_probability_requires_a_valid_explicit_score(self):
        self.assertEqual(probability("<p_success>0.25</p_success>"), 0.25)
        for malformed in ("0.25", "<p_success>1.2</p_success>", "<p_success>-0.1</p_success>"):
            self.assertIsNone(probability(malformed))

    def test_retained_analysis_reproduces(self):
        rows = read_jsonl(ROOT / "data/development_pool_utilities.jsonl")
        reference = json.loads((ROOT / "data/development_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(recompute(rows, reference)["status"], "PASS")

    def test_missing_task_arm_fails_validation(self):
        rows = read_jsonl(ROOT / "data/development_pool_utilities.jsonl")
        reference = json.loads((ROOT / "data/development_summary.json").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "denominators"):
            recompute(rows[:-1], reference)

    def test_earlier_histograms_reconcile_and_reject_missing_pools(self):
        summary = json.loads((ROOT / "data/earlier_audit_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(check(summary)["serialization"]["same_winner_pools"], 21)
        summary["serialization_winner_histogram"][0]["pool_count"] -= 1
        with self.assertRaisesRegex(ValueError, "all 37 pools"):
            check(summary)


if __name__ == "__main__":
    unittest.main()
