import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from backend.climb_path.schema import SUPPORTED_GRADES
from backend.data_processing.build_phase1_dataset import DEFAULT_INPUTS, build_dataset
from backend.evaluation.nearest_route_baseline import NearestRouteBaseline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = PROJECT_ROOT / "data_contract" / "v1"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Phase1ContractTests(unittest.TestCase):
    def test_manifest_hashes_and_split_invariants(self):
        manifest = json.loads((CONTRACT_DIR / "manifest.json").read_text(encoding="utf-8"))
        board = json.loads((CONTRACT_DIR / "board_holds.json").read_text(encoding="utf-8"))
        valid_holds = set(board["holds"])
        self.assertEqual(len(valid_holds), 141)

        seen: set[str] = set()
        multi_start = False
        multi_end = False
        for split in ("train", "validation", "test", "ood"):
            records = read_jsonl(CONTRACT_DIR / f"{split}.jsonl")
            fingerprints = {record["fingerprint"] for record in records}
            self.assertFalse(seen & fingerprints)
            seen.update(fingerprints)
            for record in records:
                holds = (
                    record["start_holds"]
                    + record["middle_holds"]
                    + record["end_holds"]
                )
                self.assertTrue(set(holds) <= valid_holds)
                multi_start = multi_start or len(record["start_holds"]) > 1
                multi_end = multi_end or len(record["end_holds"]) > 1
                if split == "ood":
                    self.assertNotIn(record["grade"], SUPPORTED_GRADES)
                else:
                    self.assertIn(record["grade"], SUPPORTED_GRADES)

        self.assertTrue(multi_start)
        self.assertTrue(multi_end)
        self.assertEqual(manifest["invariants"]["cross_split_fingerprint_overlap"], 0)
        for filename, metadata in manifest["files"].items():
            self.assertEqual(sha256_file(CONTRACT_DIR / filename), metadata["sha256"])

    def test_frozen_evaluation_references_only_test_routes(self):
        train_ids = {record["route_id"] for record in read_jsonl(CONTRACT_DIR / "train.jsonl")}
        test_ids = {record["route_id"] for record in read_jsonl(CONTRACT_DIR / "test.jsonl")}
        prompts = read_jsonl(CONTRACT_DIR / "frozen_evaluation.jsonl")
        reference_ids = {prompt["reference_route_id"] for prompt in prompts}
        self.assertEqual(len(prompts), 128)
        self.assertTrue(reference_ids <= test_ids)
        self.assertFalse(reference_ids & train_ids)

    def test_builder_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "v1"
            build_dataset(
                [PROJECT_ROOT / path for path in DEFAULT_INPUTS],
                output_dir,
            )
            expected_manifest = (CONTRACT_DIR / "manifest.json").read_bytes()
            self.assertEqual((output_dir / "manifest.json").read_bytes(), expected_manifest)

    def test_nearest_route_baseline_prefers_grade_then_endpoints(self):
        baseline = NearestRouteBaseline(
            [
                {
                    "route_id": "route_a",
                    "grade": "7A",
                    "start_holds": ["A4"],
                    "middle_holds": ["C8"],
                    "end_holds": ["I17"],
                },
                {
                    "route_id": "route_b",
                    "grade": "7A+",
                    "start_holds": ["F4"],
                    "middle_holds": ["F10"],
                    "end_holds": ["F17"],
                },
            ]
        )
        prediction = baseline.predict(
            grade="7A",
            start_holds=["F4"],
            end_holds=["F17"],
        )
        self.assertEqual(prediction.source_route_id, "route_a")
        self.assertEqual(prediction.grade_distance, 0)


if __name__ == "__main__":
    unittest.main()
