"""Evaluate final predictions, path ablations, agreement, and conflicts."""

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score


def _binary(decision):
    return 1 if decision == "vulnerable" else 0


def _metrics(labels, predictions):
    return {"accuracy": accuracy_score(labels, predictions),
            "balanced_accuracy": balanced_accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, zero_division=0),
            "recall": recall_score(labels, predictions, zero_division=0),
            "f1": f1_score(labels, predictions, zero_division=0),
            "mcc": matthews_corrcoef(labels, predictions)}


def _pair_metrics(rows, prediction_getter):
    """Compute paired vulnerable/patched discrimination metrics.

    A pair is correct only when its vulnerable member is predicted vulnerable
    and its patched member is predicted benign. Labels, rather than the
    before/after field names, determine which member is vulnerable so that an
    incorrectly ordered record cannot silently corrupt PairAcc.
    """
    counts = Counter()
    for index, row in enumerate(rows):
        labels = {version: int(row["labels"][version]) for version in ("before", "after")}
        if sorted(labels.values()) != [0, 1]:
            pair_id = row.get("pair_id", row.get("id", index))
            raise ValueError(
                f"Pair {pair_id!r} must contain exactly one vulnerable label (1) "
                f"and one patched label (0); got {labels}"
            )

        vulnerable_version = next(version for version, label in labels.items() if label == 1)
        patched_version = next(version for version, label in labels.items() if label == 0)
        vulnerable_correct = prediction_getter(row[vulnerable_version]) == 1
        patched_correct = prediction_getter(row[patched_version]) == 0

        if vulnerable_correct and patched_correct:
            counts["both_correct"] += 1
        elif vulnerable_correct:
            counts["vulnerable_only_correct"] += 1
        elif patched_correct:
            counts["patched_only_correct"] += 1
        else:
            counts["both_wrong"] += 1

    total = len(rows)
    correct = counts["both_correct"]
    return {
        "total_pairs": total,
        "correct_pairs": correct,
        "pair_accuracy": correct / total if total else 0.0,
        "both_correct": correct,
        "vulnerable_only_correct": counts["vulnerable_only_correct"],
        "patched_only_correct": counts["patched_only_correct"],
        "both_wrong": counts["both_wrong"],
    }


def evaluate(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8")); labels = []; final = []; rule = []; knowledge = []
    modes = Counter(); agreements = 0; conflicts = 0
    for row in rows:
        for version in ("before", "after"):
            result = row[version]; labels.append(int(row["labels"][version]))
            final.append(int(result["final_result"]["is_vulnerable"]))
            r = result["path_decisions"]["rule"]; k = result["path_decisions"]["knowledge"]
            rule.append(_binary(r)); knowledge.append(_binary(k)); modes[result["final_result"]["decision_mode"]] += 1
            agreements += r == k and r != "unknown"; conflicts += {r, k} == {"vulnerable", "benign"}
    n = len(labels)
    return {"result_path": str(path.resolve()), "pairs": len(rows), "functions": n,
            "final_metrics": _metrics(labels, final), "rule_path_metrics": _metrics(labels, rule),
            "knowledge_path_metrics": _metrics(labels, knowledge),
            "final_pair_metrics": _pair_metrics(
                rows, lambda result: int(result["final_result"]["is_vulnerable"])),
            "rule_path_pair_metrics": _pair_metrics(
                rows, lambda result: _binary(result["path_decisions"]["rule"])),
            "knowledge_path_pair_metrics": _pair_metrics(
                rows, lambda result: _binary(result["path_decisions"]["knowledge"])),
            "path_agreement_rate": agreements / n if n else 0.0,
            "path_conflict_rate": conflicts / n if n else 0.0, "arbiter_decision_modes": dict(modes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("result", type=Path); parser.add_argument("--save-json", type=Path)
    args = parser.parse_args(); metrics = evaluate(args.result); print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
