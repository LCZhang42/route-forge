import unittest

from fastapi.testclient import TestClient

from backend.api.phase3_server import SlidingWindowRateLimiter, create_phase3_app
from backend.climb_path.constraints import RouteDiagnostics
from backend.climb_path.schema import Hold, Route
from backend.climb_path.structured_inference import (
    GenerationResult,
    RankedRouteCandidate,
)


class FakePhase3Service:
    model_version = "fake-v1"
    data_version = "data-v1"

    def metadata(self):
        return {
            "model_version": self.model_version,
            "data_version": self.data_version,
            "supported_grades": ["7A"],
        }

    def generate(self, **kwargs):
        route = Route(
            grade=kwargs["grade"],
            start_holds=tuple(
                Hold.from_label(label) for label in kwargs["start_holds"]
            ),
            middle_holds=(Hold.from_label("D8"), Hold.from_label("F12")),
            end_holds=tuple(Hold.from_label(label) for label in kwargs["end_holds"]),
        )
        diagnostics = RouteDiagnostics(
            hard_rejections=(),
            warnings=(),
            middle_hold_count=2,
            target_middle_holds=2,
            length_delta=0,
            move_distances=(4.0, 4.472, 5.099),
            maximum_move_distance=5.099,
            reachability_violations=0,
            downward_moves=0,
            novel_route=True,
        )
        return GenerationResult(
            candidates=(
                RankedRouteCandidate(
                    route=route,
                    source="model",
                    score=-1.25,
                    model_log_probability=-1.25,
                    diagnostics=diagnostics,
                ),
            ),
            seed=kwargs["seed"],
            generated_attempts=4,
            rejected_attempts=3,
            used_fallback=False,
        )


class FailingPhase3Service(FakePhase3Service):
    def generate(self, **_kwargs):
        raise RuntimeError("private internal failure detail")


class Phase3ApiTests(unittest.TestCase):
    def setUp(self):
        self.app = create_phase3_app(service=FakePhase3Service())

    def test_generation_contract_and_reproducibility_fields(self):
        with TestClient(self.app) as client:
            response = client.post(
                "/api/v1/routes/generate",
                json={
                    "grade": "7A",
                    "start_holds": ["A4", "F4"],
                    "end_holds": ["G17"],
                    "candidate_count": 1,
                    "seed": 123,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["seed"], 123)
        self.assertEqual(payload["model_version"], "fake-v1")
        self.assertEqual(payload["candidates"][0]["start_holds"], ["A4", "F4"])
        self.assertTrue(payload["candidates"][0]["diagnostics"]["valid"])
        self.assertNotIn("traceback", response.text.lower())

    def test_readiness_and_model_metadata(self):
        with TestClient(self.app) as client:
            ready = client.get("/api/v1/health/ready")
            model = client.get("/api/v1/model")

        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json(), {"status": "ready"})
        self.assertEqual(model.status_code, 200)
        self.assertEqual(model.json()["model_version"], "fake-v1")

    def test_legacy_coordinate_endpoints_remain_supported(self):
        with TestClient(self.app) as client:
            response = client.post(
                "/api/v1/routes/generate",
                json={
                    "grade": "7A",
                    "start_hold": [0, 4],
                    "end_hold": [6, 17],
                    "candidate_count": 1,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["candidates"][0]["start_holds"], ["A4"])

    def test_rejects_endpoint_overlap_and_excessive_candidates(self):
        with TestClient(self.app) as client:
            overlap = client.post(
                "/api/v1/routes/generate",
                json={"grade": "7A", "start_holds": ["A4"], "end_holds": ["A4"]},
            )
            too_many = client.post(
                "/api/v1/routes/generate",
                json={
                    "grade": "7A",
                    "start_holds": ["A4"],
                    "end_holds": ["G17"],
                    "candidate_count": 4,
                },
            )

        self.assertEqual(overlap.status_code, 422)
        self.assertEqual(too_many.status_code, 422)

    def test_internal_errors_are_sanitized(self):
        app = create_phase3_app(service=FailingPhase3Service())
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/routes/generate",
                json={
                    "grade": "7A",
                    "start_holds": ["A4"],
                    "end_holds": ["G17"],
                },
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Route generation failed")
        self.assertNotIn("private internal failure detail", response.text)

    def test_rate_limiter_uses_a_sliding_window(self):
        limiter = SlidingWindowRateLimiter(requests_per_minute=2)

        self.assertTrue(limiter.allow("client", now=0.0))
        self.assertTrue(limiter.allow("client", now=1.0))
        self.assertFalse(limiter.allow("client", now=59.0))
        self.assertTrue(limiter.allow("client", now=60.0))


if __name__ == "__main__":
    unittest.main()
