import unittest

from fastapi.testclient import TestClient

from backend.api.server import create_app


class ApiSmokeTests(unittest.TestCase):
    def test_health_and_missing_model_behavior(self):
        app = create_app(model_dir="Z:/definitely-missing-climb-model")
        with TestClient(app) as client:
            self.assertEqual(client.get("/api/health/live").status_code, 200)
            self.assertEqual(client.get("/api/health/ready").status_code, 503)
            response = client.post(
                "/api/v1/routes/generate",
                json={
                    "grade": "7A",
                    "start_hold": [0, 4],
                    "end_hold": [8, 17],
                },
            )
            self.assertEqual(response.status_code, 503)

    def test_rejects_untrained_grade(self):
        app = create_app(model_dir="Z:/definitely-missing-climb-model")
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/routes/generate",
                json={
                    "grade": "8A",
                    "start_hold": [0, 4],
                    "end_hold": [8, 17],
                },
            )
            self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
