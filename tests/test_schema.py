import unittest

from backend.climb_path.schema import Hold, Route


class HoldTests(unittest.TestCase):
    def test_round_trip_label(self):
        hold = Hold.from_label("K17")
        self.assertEqual((hold.x, hold.y), (10, 17))
        self.assertEqual(hold.label, "K17")

    def test_rejects_out_of_bounds_hold(self):
        with self.assertRaises(ValueError):
            Hold(11, 4)


class RouteTests(unittest.TestCase):
    def test_preserves_multiple_endpoint_semantics(self):
        route = Route.from_pairs(
            grade="7A",
            start_holds=[[0, 4], [1, 4]],
            middle_holds=[[3, 8]],
            end_holds=[[6, 17]],
        )
        self.assertEqual(route.as_pairs(), [[0, 4], [1, 4], [3, 8], [6, 17]])


if __name__ == "__main__":
    unittest.main()
