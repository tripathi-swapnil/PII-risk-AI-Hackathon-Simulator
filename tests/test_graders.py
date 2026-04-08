import unittest

from env import Entity
from graders import grade_easy, grade_hard, grade_medium


class TestGraders(unittest.TestCase):
    def setUp(self) -> None:
        self.gold = [
            Entity(text="John Doe", label="NAME", start=0, end=8, confidence=1.0),
            Entity(text="123-45-6789", label="SSN", start=20, end=31, confidence=1.0),
        ]

    def test_easy_grade_is_bounded_and_deterministic(self) -> None:
        pred = list(self.gold)
        score1 = grade_easy(pred, self.gold)
        score2 = grade_easy(pred, self.gold)
        self.assertGreaterEqual(score1, 0.0)
        self.assertLessEqual(score1, 1.0)
        self.assertEqual(score1, score2)

    def test_medium_grade_is_bounded_and_deterministic(self) -> None:
        pred = list(self.gold)
        score1 = grade_medium(pred, self.gold, "high", "high")
        score2 = grade_medium(pred, self.gold, "high", "high")
        self.assertGreaterEqual(score1, 0.0)
        self.assertLessEqual(score1, 1.0)
        self.assertEqual(score1, score2)

    def test_hard_grade_applies_critical_cap(self) -> None:
        pred_missing_critical = [
            Entity(text="John Doe", label="NAME", start=0, end=8, confidence=1.0)
        ]
        score = grade_hard(
            pred_missing_critical,
            self.gold,
            "high",
            "high",
            "John Doe and SSN 123-45-6789",
            "[REDACTED_NAME] and SSN 123-45-6789",
            3,
        )
        self.assertLessEqual(score, 0.4)


if __name__ == "__main__":
    unittest.main()
