"""Deterministic evidence weighting for retrieved CERT rules."""

from typing import Any, Dict, List


class EvidenceAggregator:
    POLARITY = {"Yes": 1.0, "No": -1.0, "Unknown": 0.0}

    def __init__(
        self,
        minimum_similarity: float = 0.0,
        minimum_confidence: float = 0.5,
    ):
        self.minimum_similarity = minimum_similarity
        self.minimum_confidence = minimum_confidence

    def aggregate(
        self,
        retrieval_description: str,
        retrieved_rules: List[Dict[str, Any]],
        reasoning_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # Associate results by rule_id instead of positional zip().
        results_by_id = {item["rule_id"]: item for item in reasoning_results}
        rule_results = []

        for rule in retrieved_rules:
            rule_id = rule["rule_id"]
            result = results_by_id.get(
                rule_id,
                {
                    "rule_id": rule_id,
                    "violation": "Unknown",
                    "confidence": 0.0,
                    "analysis": "Missing reasoning result.",
                    "evidence": {},
                    "status": "missing",
                },
            )

            similarity = float(rule.get("similarity", 0.0))
            confidence = float(result.get("confidence", 0.0))
            violation = result.get("violation", "Unknown")
            eligible = (
                similarity >= self.minimum_similarity
                and confidence >= self.minimum_confidence
                and result.get("status") == "success"
            )
            weighted_score = (
                similarity * confidence * self.POLARITY.get(violation, 0.0)
                if eligible
                else 0.0
            )

            rule_results.append(
                {
                    "rule_id": rule_id,
                    "rule_document": rule.get("document", ""),
                    "similarity": similarity,
                    "violation": violation,
                    "confidence": confidence,
                    "weighted_score": weighted_score,
                    "eligible": eligible,
                    "status": result.get("status", "unknown"),
                    "analysis": result.get("analysis", ""),
                    "evidence": result.get("evidence", {}),
                }
            )

        rule_results.sort(key=lambda item: abs(item["weighted_score"]), reverse=True)
        supported = [
            item
            for item in rule_results
            if item["eligible"] and item["violation"] == "Yes"
        ]
        opposed = [
            item
            for item in rule_results
            if item["eligible"] and item["violation"] == "No"
        ]

        return {
            "retrieval_description": retrieval_description,
            "rule_results": rule_results,
            "supported_results": supported,
            "opposed_results": opposed,
            "statistics": {
                "retrieved": len(retrieved_rules),
                "reasoning_success": sum(
                    item["status"] == "success" for item in rule_results
                ),
                "yes": sum(item["violation"] == "Yes" for item in rule_results),
                "no": sum(item["violation"] == "No" for item in rule_results),
                "unknown": sum(
                    item["violation"] == "Unknown" for item in rule_results
                ),
                "support_score": sum(item["weighted_score"] for item in supported),
                "opposition_score": abs(
                    sum(item["weighted_score"] for item in opposed)
                ),
            },
        }

