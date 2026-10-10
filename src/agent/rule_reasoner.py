"""Rule-constrained reasoning with validation and explicit failure results."""

import json
import re
import time
from typing import Any, Dict

from src.util.logger import get_detail_logger, get_logger


logger = get_logger()
detail_logger = get_detail_logger()


class RuleReasoner:
    ALLOWED_VIOLATIONS = {"Yes", "No", "Unknown"}

    def __init__(self, llm, retry_times: int = 3):
        self.llm = llm
        self.retry_times = retry_times

    def build_prompt(self, code: str, rule: Dict[str, Any]) -> str:
        return f"""
You are an experienced C/C++ security auditor.

Judge the source code only against the supplied SEI CERT C rule. Do not infer
unrelated vulnerabilities. Quote concrete code evidence. If the available
function is insufficient for a conclusion, return Unknown.

SEI CERT rule ID: {rule['rule_id']}
SEI CERT rule:
{rule['document']}

Source code:
```c
{code}
```

Return only valid JSON:
{{
  "rule_id": "{rule['rule_id']}",
  "rule_summary": "",
  "evidence": {{
    "relevant_operations": [],
    "related_code": [],
    "observations": []
  }},
  "analysis": "",
  "violation": "Yes | No | Unknown",
  "confidence": 0.0
}}

confidence must be a non-negative decimal number from 0.0 to 1.0. The
violation field already expresses direction, so confidence must not be signed.
Do not return a percentage such as 80, 95, "80%", or "95%".
"""

    @staticmethod
    def _extract_json(response: str) -> Dict[str, Any]:
        cleaned = response.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", cleaned, re.S)
        return json.loads(match.group() if match else cleaned)

    def _validate(self, result: Dict[str, Any], expected_rule_id: str) -> Dict[str, Any]:
        if result.get("rule_id") != expected_rule_id:
            raise ValueError("The returned rule_id does not match the requested rule")

        violation = result.get("violation")
        if violation not in self.ALLOWED_VIOLATIONS:
            raise ValueError(f"Invalid violation value: {violation}")

        confidence = self._normalize_confidence(result.get("confidence"))
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

        evidence = result.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("evidence must be an object")

        for field in ("relevant_operations", "related_code", "observations"):
            if not isinstance(evidence.get(field), list):
                raise ValueError(f"evidence.{field} must be a list")

        result["confidence"] = confidence
        result["status"] = "success"
        return result

    @staticmethod
    def _normalize_confidence(value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError("confidence must be numeric")

        is_explicit_percentage = False
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.endswith("%"):
                is_explicit_percentage = True
                cleaned = cleaned[:-1].strip()
            try:
                value = float(cleaned)
            except ValueError as exc:
                raise ValueError("confidence must be numeric") from exc

        if not isinstance(value, (int, float)):
            raise ValueError("confidence must be numeric")

        # Some compatible models emit a signed certainty score (for example
        # -0.95 for a confident "No"). Direction already belongs to the
        # violation field, so confidence is the magnitude of that score.
        confidence = abs(float(value))
        if is_explicit_percentage or 1 < confidence <= 100:
            confidence /= 100.0
        return confidence

    def reason(self, code: str, rule: Dict[str, Any]) -> Dict[str, Any]:
        rule_id = rule["rule_id"]
        prompt = self.build_prompt(code, rule)
        errors = []

        for attempt in range(1, self.retry_times + 1):
            response = None
            try:
                response = self.llm.generate(prompt)
                result = self._extract_json(response)
                result = self._validate(result, rule_id)
                result["attempts"] = attempt
                return result
            except Exception as exc:
                errors.append(str(exc))
                logger.warning(
                    "Rule %s attempt %s/%s failed: %s",
                    rule_id,
                    attempt,
                    self.retry_times,
                    exc,
                )
                if response is not None:
                    detail_logger.debug(
                        "Rule %s invalid raw response (attempt %s): %s",
                        rule_id,
                        attempt,
                        response,
                    )
                if attempt < self.retry_times:
                    time.sleep(1)

        # Keep the failed rule in the result list so rule/result alignment is stable.
        logger.error("Rule %s skipped after %s failed attempts", rule_id, self.retry_times)
        return {
            "rule_id": rule_id,
            "rule_summary": "",
            "evidence": {
                "relevant_operations": [],
                "related_code": [],
                "observations": [],
            },
            "analysis": "Rule reasoning failed after retries.",
            "violation": "Unknown",
            "confidence": 0.0,
            "status": "failed",
            "attempts": self.retry_times,
            "errors": errors,
        }
