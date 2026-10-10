"""Evidence-closed final arbiter for VulDuet."""

import json
import re
import time
from typing import Any, Dict, List


class ArbiterAgent:
    MODES = {"cross_path_agreement", "rule_only", "knowledge_only", "conflict_resolved", "insufficient_evidence"}

    def __init__(self, llm, retry_times: int = 3):
        self.llm = llm
        self.retry_times = retry_times

    def build_prompt(self, code: str, evidence: List[Dict[str, Any]], path_decisions: Dict[str, str]) -> str:
        eligible = [x for x in evidence if x["eligible"] and x["claim"] != "unknown"]
        return f"""
You are a constrained security evidence arbiter. Decide using ONLY the eligible
evidence below. Do not invent evidence IDs, rules, CWE IDs, or code facts.

Source code:
```c
{code}
```
Independent path decisions: {json.dumps(path_decisions)}
Eligible evidence: {json.dumps(eligible, ensure_ascii=False, indent=2)}

Return only valid JSON:
{{"is_vulnerable": true, "confidence": 0.0,
 "decision_mode": "cross_path_agreement | rule_only | knowledge_only | conflict_resolved | insufficient_evidence",
 "primary_evidence": [{{"source_path": "rule | knowledge", "evidence_id": ""}}],
 "conflicts": [], "violated_rules": [], "related_cwes": [],
 "summary": "", "final_reason": ""}}
Any vulnerable decision must cite eligible supports_vulnerable evidence. A safe
decision in the presence of positive evidence must be conflict_resolved and cite
eligible supports_benign evidence. Confidence is from 0.0 to 1.0.
"""

    @staticmethod
    def _parse(text: str) -> Dict[str, Any]:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", cleaned, re.S)
        return json.loads(match.group() if match else cleaned)

    def _validate(self, result: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(result.get("is_vulnerable"), bool):
            raise ValueError("invalid decision")
        confidence = result.get("confidence")
        is_percentage = False
        if isinstance(confidence, str):
            is_percentage = confidence.strip().endswith("%")
            confidence = float(confidence.strip().rstrip("%"))
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("invalid confidence")
        confidence = abs(float(confidence))
        if is_percentage or 1 < confidence <= 100:
            confidence /= 100
        if not 0 <= confidence <= 1:
            raise ValueError("invalid confidence")
        result["confidence"] = confidence
        eligible = {(x["source_path"], x["evidence_id"]): x for x in evidence if x["eligible"] and x["claim"] != "unknown"}
        refs = result.get("primary_evidence")
        if not isinstance(refs, list):
            raise ValueError("primary_evidence must be a list")
        keys = []
        for ref in refs:
            key = (ref.get("source_path"), ref.get("evidence_id")) if isinstance(ref, dict) else None
            if key not in eligible:
                raise ValueError("primary_evidence contains invented or ineligible evidence")
            keys.append(key)
        cited = [eligible[key] for key in keys]
        positives = [x for x in eligible.values() if x["claim"] == "supports_vulnerable"]
        negatives = [x for x in eligible.values() if x["claim"] == "supports_benign"]
        if result["is_vulnerable"] and not any(x["claim"] == "supports_vulnerable" for x in cited):
            raise ValueError("vulnerable decision lacks positive evidence")
        if not result["is_vulnerable"] and positives:
            if not negatives or not any(x["claim"] == "supports_benign" for x in cited):
                raise ValueError("safe conflict decision lacks opposing evidence")
        cited_paths = {x["source_path"] for x in cited}
        expected_claim = "supports_vulnerable" if result["is_vulnerable"] else "supports_benign"
        if eligible and not any(x["claim"] == expected_claim for x in cited):
            raise ValueError("decision lacks cited evidence in the selected direction")
        if not eligible and result["is_vulnerable"]:
            raise ValueError("vulnerable decision cannot use insufficient evidence")

        cross_path_conflicts = [
            {"left": {"source_path": left["source_path"], "evidence_id": left["evidence_id"], "claim": left["claim"]},
             "right": {"source_path": right["source_path"], "evidence_id": right["evidence_id"], "claim": right["claim"]}}
            for left in eligible.values() for right in eligible.values()
            if left["source_path"] == "rule" and right["source_path"] == "knowledge"
            and left["claim"] != right["claim"]
        ]

        # decision_mode is deterministic metadata, not a model-authored fact.
        # Deriving it here prevents semantically correct decisions from failing
        # because the model chose the wrong descriptive enum.
        if not eligible:
            mode = "insufficient_evidence"
        elif cross_path_conflicts:
            mode = "conflict_resolved"
        elif cited_paths == {"rule", "knowledge"}:
            if any(x["claim"] != expected_claim for x in cited):
                raise ValueError("cross-path agreement cites contradictory evidence")
            mode = "cross_path_agreement"
        elif cited_paths == {"rule"}:
            mode = "rule_only"
        elif cited_paths == {"knowledge"}:
            mode = "knowledge_only"
        else:
            raise ValueError("unable to derive decision_mode from cited evidence")
        result["decision_mode"] = mode
        # Conflict records are derived, not authored by the model.
        result["conflicts"] = cross_path_conflicts
        allowed_rules = {x["evidence_id"] for x in evidence if x["source_path"] == "rule" and x["eligible"] and x["claim"] == "supports_vulnerable"}
        if not isinstance(result.get("violated_rules"), list) or not set(result["violated_rules"]).issubset(allowed_rules):
            raise ValueError("invalid violated_rules")
        allowed_cwes = {cwe for x in evidence if x["eligible"] for cwe in x.get("cwe_ids", [])}
        if not isinstance(result.get("related_cwes"), list) or not set(result["related_cwes"]).issubset(allowed_cwes):
            raise ValueError("invalid related_cwes")
        return result

    def judge(self, code: str, evidence: List[Dict[str, Any]], path_decisions: Dict[str, str]) -> Dict[str, Any]:
        errors = []
        for attempt in range(1, self.retry_times + 1):
            try:
                result = self._validate(self._parse(self.llm.generate(self.build_prompt(code, evidence, path_decisions))), evidence)
                result["attempts"] = attempt
                return result
            except Exception as exc:
                errors.append(str(exc))
                if attempt < self.retry_times:
                    time.sleep(1)
        raise RuntimeError(f"arbiter failed validation: {errors}")
