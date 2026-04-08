import json
from pathlib import Path
from typing import Any


DATASET_DIR = Path(__file__).resolve().parent / "dataset"


def load_dataset(file_name: str) -> list[dict[str, Any]]:
    dataset_path = DATASET_DIR / file_name
    with dataset_path.open("r", encoding="utf-8") as f:
        return json.load(f)
