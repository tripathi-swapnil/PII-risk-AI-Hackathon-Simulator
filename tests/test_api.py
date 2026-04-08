import unittest

from fastapi.testclient import TestClient

from app import app


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_required_endpoints_respond(self) -> None:
        self.assertEqual(self.client.get("/health").status_code, 200)

        reset_resp = self.client.post("/reset", json={"task_type": "easy"})
        self.assertEqual(reset_resp.status_code, 200)

        step_resp = self.client.post(
            "/step",
            json={"action": {"action_type": "detect", "spans": [], "confidence": 0.3}},
        )
        self.assertEqual(step_resp.status_code, 200)
        payload = step_resp.json()
        self.assertIn("observation", payload)
        self.assertIn("reward", payload)
        self.assertIn("done", payload)
        self.assertIn("info", payload)

        tasks_resp = self.client.get("/tasks")
        self.assertEqual(tasks_resp.status_code, 200)
        self.assertEqual(len(tasks_resp.json()), 3)

        grader_resp = self.client.post(
            "/grader",
            json={"task_type": "easy", "pred_entities": []},
        )
        self.assertEqual(grader_resp.status_code, 200)
        self.assertIn("score", grader_resp.json())

    def test_baseline_endpoint_returns_scores(self) -> None:
        response = self.client.post("/baseline", json={"use_ai": False})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("scores", payload)
        self.assertIn("easy", payload["scores"])
        self.assertIn("medium", payload["scores"])
        self.assertIn("hard", payload["scores"])


if __name__ == "__main__":
    unittest.main()
