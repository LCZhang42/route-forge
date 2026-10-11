import unittest

from backend.climb_path.inference import (
    build_prompt,
    normalize_generated_route,
    parse_generated_text,
    validate_route,
)
from backend.climb_path.schema import Hold


class InferenceFormatTests(unittest.TestCase):
    def test_prompt_and_parser_round_trip(self):
        prompt = build_prompt("7A", Hold(0, 4), Hold(8, 17))
        route = parse_generated_text(prompt + " (1,7) (3,11) (5,13)")
        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.grade, "7A")
        self.assertEqual(route.as_pairs(), [[0, 4], [1, 7], [3, 11], [5, 13], [8, 17]])

    def test_validation_reports_duplicates_and_invalid_holds(self):
        route = parse_generated_text(
            "GRADE: 7A | START: (0,4) | END: (8,17) | MID: (1,7) (1,7)"
        )
        assert route is not None
        warnings = validate_route(
            route,
            requested_grade="7A",
            requested_start=Hold(0, 4),
            requested_end=Hold(8, 17),
            valid_holds={(0, 4), (8, 17)},
        )
        self.assertTrue(any("duplicate" in warning for warning in warnings))
        self.assertTrue(any("invalid board" in warning for warning in warnings))

    def test_normalization_stops_at_end_and_removes_noisy_holds(self):
        route = parse_generated_text(
            "GRADE: 7A | START: (0,4) | END: (8,17) | "
            "MID: (2,7) (2,7) (5,9) (8,17) (4,15)"
        )
        assert route is not None

        normalized, warnings = normalize_generated_route(
            route,
            valid_holds={(0, 4), (2, 7), (8, 17)},
        )

        self.assertEqual(normalized.as_pairs(), [[0, 4], [2, 7], [8, 17]])
        self.assertTrue(any("truncated" in warning for warning in warnings))
        self.assertTrue(any("duplicate" in warning for warning in warnings))
        self.assertTrue(any("invalid" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
