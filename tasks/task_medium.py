from typing import Any

from .common import load_dataset


def load_medium_tasks() -> list[dict[str, Any]]:
    return load_dataset("medium.json")
