"""Validated loader for alternating before/after PrimeVul JSONL rows."""

import json

from src.util.logger import get_logger


logger = get_logger()


class PrimeVulPairLoader:
    def __init__(self, path: str):
        self.path = path

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as file:
            pair_number = 0
            while True:
                before_line = file.readline()
                if not before_line:
                    break
                pair_number += 1
                after_line = file.readline()
                if not after_line:
                    raise ValueError(f"PrimeVul pair {pair_number} is missing its second row")

                before = json.loads(before_line)
                after = json.loads(after_line)
                if before.get("commit_id") != after.get("commit_id"):
                    logger.error(
                        "[Skip invalid pair] pair=%s before_id=%s after_id=%s "
                        "before_commit=%s after_commit=%s reason=mismatched_commit_ids",
                        pair_number,
                        before.get("idx"),
                        after.get("idx"),
                        before.get("commit_id"),
                        after.get("commit_id"),
                    )
                    continue
                if before.get("target") == after.get("target"):
                    logger.error(
                        "[Skip invalid pair] pair=%s before_id=%s after_id=%s "
                        "before_label=%s after_label=%s reason=labels_not_opposite",
                        pair_number,
                        before.get("idx"),
                        after.get("idx"),
                        before.get("target"),
                        after.get("target"),
                    )
                    continue

                yield {
                    "id": before["idx"],
                    "commit_id": before["commit_id"],
                    "cwe": before.get("cwe", []),
                    "before": {"code": before["func"], "label": before["target"]},
                    "after": {"code": after["func"], "label": after["target"]},
                }
