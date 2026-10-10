"""Result persistence for VulDuet experiments with structured logging."""

import json
from pathlib import Path

from src.util.logger import get_logger


logger = get_logger()


class ResultSaver:
    def __init__(self, save_path: str, auto_save_every: int = 20, resume: bool = True):
        self.save_path = Path(save_path)
        self.auto_save_every = auto_save_every
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.finished_ids = set()

        if resume and self.save_path.exists():
            with self.save_path.open("r", encoding="utf-8") as file:
                self.results = json.load(file)
            self.finished_ids = {item["id"] for item in self.results}
            logger.info("Resumed %s saved samples", len(self.results))

    def contains(self, sample_id) -> bool:
        return sample_id in self.finished_ids

    def append(self, result) -> bool:
        sample_id = result["id"]
        if sample_id in self.finished_ids:
            logger.info("Duplicate sample ignored: %s", sample_id)
            return False

        self.finished_ids.add(sample_id)
        self.results.append(result)
        if len(self.results) % self.auto_save_every == 0:
            self.save()
        return True

    def save(self) -> None:
        # Write beside the destination and atomically replace it. Interrupting
        # a long JSON serialization can no longer truncate the last checkpoint.
        temporary = self.save_path.with_suffix(self.save_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(self.results, file, indent=2, ensure_ascii=False)
        temporary.replace(self.save_path)
        logger.info("Saved %s samples to %s", len(self.results), self.save_path)

    def close(self) -> None:
        self.save()
