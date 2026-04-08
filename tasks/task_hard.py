from typing import Any

from .common import load_dataset


def load_hard_tasks() -> list[dict[str, Any]]:
    return load_dataset("hard.json")
