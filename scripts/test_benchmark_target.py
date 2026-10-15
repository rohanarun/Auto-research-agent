import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from benchmark_target import discovery_outcome


class DiscoveryEndpointTests(unittest.TestCase):
    def test_baseline_reformatting_and_recompression_never_count_as_discovery(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, candidate = root / "baseline.gz", root / "candidate.gz"
            baseline.write_bytes(gzip.compress(b'{"R":4,"ops":[[1,2,3]]}', mtime=0))
            candidate.write_bytes(gzip.compress(b'{ "ops": [[1,2,3]], "R": 4 }', mtime=123))
            self.assertNotEqual(baseline.read_bytes(), candidate.read_bytes())
            result = discovery_outcome({"milestones": [{"id": "target"}]},
                                       {"axis": baseline}, {"axis": candidate},
                                       {"valid": True, "target_reached": True},
                                       {"valid": True, "target_reached": True})
            self.assertEqual(result["classification"], "baseline_reproduction")
            self.assertFalse(result["research_success"])

    def test_valid_but_unimproved_partial_invalid_and_unrevalidated_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, candidate = root / "baseline.gz", root / "candidate.gz"
            baseline.write_bytes(gzip.compress(json.dumps({"R": 4}).encode()))
            candidate.write_bytes(gzip.compress(json.dumps({"R": 3}).encode()))
            protocol = {"milestones": [{"id": "first"}, {"id": "final"}]}
            good = {"valid": True, "target_reached": True, "bit": {},
                    "finite_bridge": {}, "replay": ["fixture-independent-replay"],
                    "milestones": [{"id": "first", "target_reached": True},
                                   {"id": "final", "target_reached": True}]}
            def classify(original, replay):
                return discovery_outcome(protocol, {"axis": baseline}, {"axis": candidate}, original, replay)
            self.assertTrue(classify(good, copy.deepcopy(good))["research_success"])
            partial = copy.deepcopy(good)
            partial["milestones"][-1]["target_reached"] = False
            partial["target_reached"] = False
            self.assertFalse(classify(partial, partial)["research_success"])
            invalid = copy.deepcopy(good); invalid["valid"] = False
            self.assertFalse(classify(invalid, invalid)["research_success"])
            self.assertFalse(classify(good, {})["research_success"])
            different = copy.deepcopy(good); different["bit"] = {"different": True}
            self.assertFalse(classify(good, different)["research_success"])
            omitted = copy.deepcopy(good); omitted["milestones"].pop()
            self.assertFalse(classify(omitted, omitted)["research_success"])
            # An aggregate flag alone must not bypass any individual target.
            partial["target_reached"] = True
            self.assertFalse(classify(partial, partial)["research_success"])


if __name__ == "__main__":
    unittest.main()
