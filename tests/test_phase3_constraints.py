import json
from pathlib import Path
import unittest

from backend.climb_path.constraints import (
    GradeProfile,
    build_grade_profiles,
    diagnose_route,
)
from backend.climb_path.schema import Hold, Route, SUPPORTED_GRADES
from backend.climb_path.structured_inference import (
    GeneratedMiddle,
    Phase3RouteService,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = PROJECT_ROOT / "data_contract" / "v1"


class FakeGenerator:
    def __init__(self, candidates):
        self.candidates = candidates

    def generate_middle_candidates(self, **_kwargs):
        return self.candidates


class Phase3ConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training_records = [
            json.loads(line)
            for line in (CONTRACT_DIR / "train.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        cls.board_holds = json.loads(
            (CONTRACT_DIR / "board_holds.json").read_text(encoding="utf-8")
        )["holds"]

    def test_profiles_are_training_derived_and_ordered(self):
        profiles = build_grade_profiles(self.training_records)

        self.assertEqual(set(profiles), set(SUPPORTED_GRADES))
        for profile in profiles.values():
            self.assertLessEqual(
                profile.minimum_middle_holds, profile.target_middle_holds
            )
            self.assertLessEqual(
                profile.target_middle_holds, profile.maximum_middle_holds
            )
            self.assertLessEqual(profile.soft_move_limit, profile.hard_move_limit)

    def test_metadata_exposes_role_specific_endpoint_catalogs(self):
        service = Phase3RouteService(
            generator=FakeGenerator([]),
            training_records=self.training_records,
            board_holds=self.board_holds,
            model_version="test-model",
            data_version="test-data",
        )

        metadata = service.metadata()
        self.assertIn("F4", metadata["endpoint_holds"]["start"])
        self.assertEqual(
            {label[1:] for label in metadata["endpoint_holds"]["end"]}, {"17"}
        )
        self.assertEqual(metadata["display_row_offset"], 1)

    def test_service_rejects_holds_in_the_wrong_endpoint_role(self):
        service = Phase3RouteService(
            generator=FakeGenerator([]),
            training_records=self.training_records,
            board_holds=self.board_holds,
            model_version="test-model",
            data_version="test-data",
        )

        with self.assertRaisesRegex(ValueError, "start positions"):
            service.generate(
                grade="7A",
                start_holds=["D8"],
                end_holds=["G17"],
                candidate_count=1,
                temperature=0.8,
                top_p=0.95,
                seed=42,
            )

    def test_diagnostics_reject_severe_moves(self):
        route = Route(
            grade="7A",
            start_holds=(Hold.from_label("A4"),),
            middle_holds=(Hold.from_label("K15"),),
            end_holds=(Hold.from_label("G17"),),
        )
        diagnostics = diagnose_route(
            route,
            requested_grade="7A",
            requested_starts=route.start_holds,
            requested_ends=route.end_holds,
            valid_hold_labels={"A4", "K15", "G17"},
            profile=GradeProfile("7A", 1, 1, 3, 3.0, 5.0),
            training_fingerprints=set(),
        )

        self.assertFalse(diagnostics.valid)
        self.assertIn("severe_reachability_violation", diagnostics.hard_rejections)

    def test_service_uses_retrieval_when_model_candidates_are_rejected(self):
        reference = self.training_records[0]
        invalid_middle = (reference["start_holds"][0],)
        service = Phase3RouteService(
            generator=FakeGenerator([GeneratedMiddle(invalid_middle, -1.0)]),
            training_records=self.training_records,
            board_holds=self.board_holds,
            model_version="test-model",
            data_version="test-data",
        )

        result = service.generate(
            grade=reference["grade"],
            start_holds=reference["start_holds"],
            end_holds=reference["end_holds"],
            candidate_count=1,
            temperature=0.8,
            top_p=0.95,
            seed=42,
        )

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.candidates[0].source, "retrieval_fallback")
        self.assertTrue(result.candidates[0].diagnostics.valid)

    def test_service_keeps_valid_retrieval_first_and_model_as_alternative(self):
        reference = self.training_records[0]
        generated_middle = tuple(reference["middle_holds"][:1])
        if not generated_middle:
            generated_middle = ("D8",)
        service = Phase3RouteService(
            generator=FakeGenerator([GeneratedMiddle(generated_middle, -1.0)]),
            training_records=self.training_records,
            board_holds=self.board_holds,
            model_version="test-model",
            data_version="test-data",
        )

        result = service.generate(
            grade=reference["grade"],
            start_holds=reference["start_holds"],
            end_holds=reference["end_holds"],
            candidate_count=2,
            temperature=0.8,
            top_p=0.95,
            seed=42,
        )

        self.assertEqual(result.candidates[0].source, "retrieval_fallback")
        self.assertTrue(result.used_fallback)


if __name__ == "__main__":
    unittest.main()
