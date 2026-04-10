import importlib
import os
import unittest
from unittest.mock import patch


class FakeOpenAI:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key


class TestInferenceProxyConfig(unittest.TestCase):
    def _reload_inference(self):
        import inference

        return importlib.reload(inference)

    def test_create_client_requires_proxy_credentials(self) -> None:
        with patch.dict(
            os.environ,
            {
                "API_BASE_URL": "",
                "API_KEY": "",
            },
            clear=False,
        ):
            inference = self._reload_inference()
            client, error = inference.create_client()
            self.assertIsNone(client)
            self.assertEqual(error, "missing_proxy_credentials")

    def test_create_client_uses_api_base_url_and_api_key(self) -> None:
        with patch.dict(
            os.environ,
            {
                "API_BASE_URL": "https://proxy.example/v1",
                "API_KEY": "proxy-key",
            },
            clear=False,
        ):
            inference = self._reload_inference()
            inference.OpenAI = FakeOpenAI

            client, error = inference.create_client()
            self.assertIsNone(error)
            self.assertIsNotNone(client)
            self.assertEqual(client.base_url, "https://proxy.example/v1")
            self.assertEqual(client.api_key, "proxy-key")

    def test_model_action_http_fallback_uses_proxy_headers(self) -> None:
        with patch.dict(
            os.environ,
            {
                "API_BASE_URL": "https://proxy.example/v1",
                "API_KEY": "proxy-key",
                "MODEL_NAME": "test-model",
            },
            clear=False,
        ):
            inference = self._reload_inference()

            captured: dict[str, object] = {}

            def fake_http_post_json(url, payload, timeout=30.0, headers=None):
                captured["url"] = url
                captured["payload"] = payload
                captured["timeout"] = timeout
                captured["headers"] = headers or {}
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"action_type":"finalize","confidence":1.0}'
                            }
                        }
                    ]
                }

            inference._http_post_json = fake_http_post_json
            action = inference.model_action(None, "easy", "Email me at a@b.com", 1)

            self.assertEqual(captured["url"], "https://proxy.example/v1/chat/completions")
            self.assertEqual(captured["headers"], {"Authorization": "Bearer proxy-key"})
            self.assertEqual(action.get("action_type"), "finalize")

    def test_strict_task_score_bounds_are_exclusive(self) -> None:
        inference = self._reload_inference()
        self.assertEqual(inference._strict_task_score(0.0), 0.01)
        self.assertEqual(inference._strict_task_score(-1.0), 0.01)
        self.assertEqual(inference._strict_task_score(1.0), 0.99)
        self.assertEqual(inference._strict_task_score(2.0), 0.99)

    def test_strict_task_score_keeps_interior_value(self) -> None:
        inference = self._reload_inference()
        self.assertAlmostEqual(inference._strict_task_score(0.42), 0.42)


if __name__ == "__main__":
    unittest.main()
