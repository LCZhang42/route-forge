"""Opt-in smoke test for a deployed Phase 3 Modal URL."""

import json
import os
import unittest
from urllib.request import Request, urlopen


PHASE3_URL = os.environ.get("CLIMB_PHASE3_API_URL")


@unittest.skipUnless(PHASE3_URL, "set CLIMB_PHASE3_API_URL to test the deployment")
class Phase3DeploymentTests(unittest.TestCase):
    def test_deployed_generation_contract(self):
        request = Request(
            f"{PHASE3_URL.rstrip('/')}/api/v1/routes/generate",
            data=json.dumps(
                {
                    "grade": "6B+",
                    "start_holds": ["A4"],
                    "end_holds": ["G17"],
                    "candidate_count": 1,
                    "seed": 42,
                    "generation_mode": "balanced",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=120) as response:
            payload = json.load(response)

        self.assertEqual(response.status, 200)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["generation_mode"], "balanced")
        self.assertTrue(payload["candidates"])
        self.assertTrue(payload["candidates"][0]["diagnostics"]["valid"])
        self.assertGreaterEqual(
            len(payload["candidates"][0]["diagnostics"]["plausible_path"]), 2
        )


if __name__ == "__main__":
    unittest.main()
