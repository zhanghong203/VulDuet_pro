"""Deterministic aggregation and normalization for the two evidence paths."""

from typing import Any, Dict, List


class VulnerabilityEvidenceAggregator:
    POLARITY = {"Supports": 1.0, "Opposes": -1.0, "Unknown": 0.0}

    def __init__(self, minimum_similarity=0.0, minimum_scenario_match=0.3,
                 minimum_confidence=0.5, decision_margin=0.05):
        self.minimum_similarity = minimum_similarity
        self.minimum_scenario_match = minimum_scenario_match
        self.minimum_confidence = minimum_confidence
        self.decision_margin = decision_margin

    def aggregate(self, description: str, retrieved: List[Dict[str, Any]], reasoning: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_id = {x["knowledge_id"]: x for x in reasoning}
        results = []
        for item in retrieved:
            kid = item["knowledge_id"]
            result = by_id.get(kid, {"conclusion": "Unknown", "confidence": 0.0, "scenario_match": 0.0, "status": "missing", "evidence": {}})
            similarity = float(item.get("similarity", 0.0))
            confidence = float(result.get("confidence", 0.0))
            scenario_match = float(result.get("scenario_match", 0.0))
            conclusion = result.get("conclusion", "Unknown")
            fix_present = result.get("fix_present", "Unknown")
            result_evidence = result.get("evidence", {})
            # A historical scenario remains applicable when the current code
            # contains its effective fix. Permit a slightly lower match score for
            # concrete fixes, but never let an unrelated (zero-match) scenario
            # become benign evidence merely because the model said "fixed".
            fixed_scenario_floor = self.minimum_scenario_match * (2.0 / 3.0)
            fixed_applicable = (
                conclusion == "Opposes"
                and fix_present == "Yes"
                and scenario_match >= fixed_scenario_floor
                and bool(result_evidence.get("existing_fixes"))
            )
            applicable = (
                scenario_match >= self.minimum_scenario_match
                or fixed_applicable
            )
            eligible = (
                similarity >= self.minimum_similarity
                and applicable
                and confidence >= self.minimum_confidence
                and conclusion != "Unknown"
                and result.get("status") == "success"
            )
            effective_match = scenario_match if eligible else 0.0
            score = (
                similarity * effective_match * confidence
                * self.POLARITY.get(conclusion, 0.0)
                if eligible else 0.0
            )
            results.append({
                "knowledge_id": kid, "knowledge": item, "similarity": similarity,
                "scenario_match": scenario_match, "conclusion": conclusion,
                "cause_present": result.get("cause_present", "Unknown"),
                "trigger_present": result.get("trigger_present", "Unknown"),
                "fix_present": fix_present,
                "confidence": confidence, "weighted_score": score, "eligible": eligible,
                "status": result.get("status", "unknown"), "analysis": result.get("analysis", ""),
                "evidence": result_evidence,
            })
        results.sort(key=lambda x: abs(x["weighted_score"]), reverse=True)
        supported = [x for x in results if x["eligible"] and x["conclusion"] == "Supports"]
        opposed = [x for x in results if x["eligible"] and x["conclusion"] == "Opposes"]
        best_support = max((x["weighted_score"] for x in supported), default=0.0)
        best_opposition = max((-x["weighted_score"] for x in opposed), default=0.0)
        if best_support > best_opposition + self.decision_margin:
            decision = "vulnerable"
        elif best_opposition > best_support + self.decision_margin:
            decision = "benign"
        else:
            decision = "unknown"
        return {"retrieval_description": description, "knowledge_results": results,
                "supported_results": supported, "opposed_results": opposed,
                "path_decision": decision,
                "statistics": {"retrieved": len(retrieved), "eligible": sum(x["eligible"] for x in results),
                               "support_score": sum(x["weighted_score"] for x in supported),
                               "opposition_score": abs(sum(x["weighted_score"] for x in opposed)),
                               "best_support_score": best_support,
                               "best_opposition_score": best_opposition,
                               "decision_margin": self.decision_margin}}


def normalize_dual_path_evidence(rule_summary: Dict[str, Any], knowledge_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence = []
    for item in rule_summary["rule_results"]:
        claim = {"Yes": "supports_vulnerable", "No": "supports_benign"}.get(item["violation"], "unknown")
        evidence.append({"source_path": "rule", "evidence_id": item["rule_id"], "claim": claim,
                         "applicability": item["similarity"], "confidence": item["confidence"],
                         "weighted_score": item["weighted_score"], "eligible": item["eligible"],
                         "cwe_ids": [], "code_evidence": item.get("evidence", {}), "reason": item.get("analysis", "")})
    for item in knowledge_summary["knowledge_results"]:
        claim = {"Supports": "supports_vulnerable", "Opposes": "supports_benign"}.get(item["conclusion"], "unknown")
        knowledge = item.get("knowledge", {})
        evidence.append({"source_path": "knowledge", "evidence_id": item["knowledge_id"], "claim": claim,
                         "applicability": item["scenario_match"], "retrieval_similarity": item["similarity"],
                         "confidence": item["confidence"], "weighted_score": item["weighted_score"],
                         "eligible": item["eligible"], "cwe_ids": knowledge.get("cwe_ids", []),
                         "code_evidence": item.get("evidence", {}), "reason": item.get("analysis", ""),
                         "provenance": knowledge.get("provenance", {})})
    return evidence
