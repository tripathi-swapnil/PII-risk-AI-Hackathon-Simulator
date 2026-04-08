from typing import Any

from .common import load_dataset


def load_easy_tasks() -> list[dict[str, Any]]:
    return load_dataset("easy.json")
