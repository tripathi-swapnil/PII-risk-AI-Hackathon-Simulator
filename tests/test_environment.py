import unittest

from env import Action, ActionType, SafePIIEnvironment, TaskType


class TestSafePIIEnvironment(unittest.TestCase):
    def setUp(self) -> None:
        self.env = SafePIIEnvironment()

    def test_reset_returns_valid_observation(self) -> None:
        obs = self.env.reset(TaskType.EASY)
        self.assertEqual(obs.task_type, TaskType.EASY)
        self.assertFalse(obs.done)
        self.assertEqual(obs.step_count, 0)
        self.assertIsInstance(obs.document_text, str)
        self.assertGreater(len(obs.document_text), 0)

    def test_step_returns_reward_done_info(self) -> None:
        self.env.reset(TaskType.EASY)
        action = Action(action_type=ActionType.DETECT, spans=[], confidence=0.5)
        obs, reward, done, info = self.env.step(action)

        self.assertIsInstance(reward, float)
        self.assertGreaterEqual(reward, 0.0)
        self.assertLessEqual(reward, 1.0)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertIn("gold_risk", info)
        self.assertIn("missed_critical", info)
        self.assertIn("max_steps", info)
        self.assertEqual(obs.step_count, 1)

    def test_disallowed_action_ends_episode(self) -> None:
        self.env.reset(TaskType.EASY)
        action = Action(
            action_type=ActionType.CLASSIFY,
            classification="low",
            confidence=0.3,
        )
        obs, reward, done, _ = self.env.step(action)

        self.assertTrue(done)
        self.assertTrue(obs.constraint_violated)
        self.assertEqual(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
