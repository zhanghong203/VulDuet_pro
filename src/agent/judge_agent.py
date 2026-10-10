"""Final judges that consume the actual rule evidence produced by RAG."""

import json
import re
import time
from typing import Any, Dict, List

from src.util.logger import get_detail_logger, get_logger


logger = get_logger()
detail_logger = get_detail_logger()


class JudgeAgent:
    def __init__(self, llm, retry_times: int = 3):
        self.llm = llm
        self.retry_times = retry_times

    @staticmethod
    def _evidence_json(summary: Dict[str, Any]) -> str:
        return json.dumps(
            {
                "statistics": summary["statistics"],
                "rule_results": summary["rule_results"],
            },
            ensure_ascii=False,
            indent=2,
        )

    def build_single_prompt(self, code: str, summary: Dict[str, Any]) -> str:
        evidence = self._evidence_json(summary)
        candidate_ids = [item["rule_id"] for item in summary["rule_results"]]
        return f"""
You are the final C/C++ security judge in a rule-grounded RAG pipeline.

Source code:
```c
{code}
```

Security retrieval description:
{summary['retrieval_description']}

Retrieved rule analyses and deterministic evidence scores:
{evidence}

Candidate rule IDs: {json.dumps(candidate_ids)}

Base the decision on concrete rule evidence. Do not invent rule IDs. A failed,
missing, low-confidence, or ineligible analysis is not positive evidence. If
evidence is insufficient, prefer a lower confidence rather than speculation.

Return only valid JSON:
{{
  "is_vulnerable": true,
  "confidence": 0.0,
  "violated_rules": [],
  "summary": "",
  "final_reason": ""
}}

confidence must be a non-negative decimal number from 0.0 to 1.0. Do not use
a signed score. Do not return a percentage such as 80, 95, "80%", or "95%".
"""

    def build_pair_prompt(
        self,
        before_code: str,
        after_code: str,
        before_summary: Dict[str, Any],
        after_summary: Dict[str, Any],
    ) -> str:
        before_evidence = self._evidence_json(before_summary)
        after_evidence = self._evidence_json(after_summary)
        allowed_ids = sorted(
            {
                item["rule_id"]
                for item in before_summary["rule_results"]
                + after_summary["rule_results"]
            }
        )
        return f"""
You are evaluating a vulnerable function and its patched version as a pair.
Determine whether concrete security weaknesses in the before version were
removed by the patch. Similar operations alone do not imply the same weakness.

Before code:
```c
{before_code}
```

After code:
```c
{after_code}
```

Before rule evidence:
{before_evidence}

After rule evidence:
{after_evidence}

Allowed rule IDs: {json.dumps(allowed_ids)}

Return only valid JSON:
{{
  "before_is_vulnerable": true,
  "after_is_vulnerable": false,
  "patch_status": "fixed | partially_fixed | not_fixed | inconclusive",
  "fixed_rules": [],
  "remaining_violated_rules": [],
  "new_violated_rules": [],
  "confidence": 0.0,
  "summary": "",
  "final_reason": ""
}}

confidence must be a non-negative decimal number from 0.0 to 1.0. Do not use
a signed score. Do not return a percentage such as 80, 95, "80%", or "95%".
"""

    @staticmethod
    def _parse(response: str) -> Dict[str, Any]:
        cleaned = response.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", cleaned, re.S)
        return json.loads(match.group() if match else cleaned)

    @staticmethod
    def _validate_confidence(result: Dict[str, Any]) -> None:
        value = result.get("confidence")
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

        confidence = abs(float(value))
        if is_explicit_percentage or 1 < confidence <= 100:
            confidence /= 100.0
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        result["confidence"] = confidence

    def _validate_single(
        self, result: Dict[str, Any], allowed_ids: List[str]
    ) -> Dict[str, Any]:
        if not isinstance(result.get("is_vulnerable"), bool):
            raise ValueError("is_vulnerable must be a boolean")
        violated = result.get("violated_rules")
        if not isinstance(violated, list) or not set(violated).issubset(allowed_ids):
            raise ValueError("violated_rules contains an unknown rule ID")
        self._validate_confidence(result)
        return result

    def _validate_pair(
        self, result: Dict[str, Any], allowed_ids: List[str]
    ) -> Dict[str, Any]:
        for field in ("before_is_vulnerable", "after_is_vulnerable"):
            if not isinstance(result.get(field), bool):
                raise ValueError(f"{field} must be a boolean")
        if result.get("patch_status") not in {
            "fixed",
            "partially_fixed",
            "not_fixed",
            "inconclusive",
        }:
            raise ValueError("invalid patch_status")
        for field in (
            "fixed_rules",
            "remaining_violated_rules",
            "new_violated_rules",
        ):
            values = result.get(field)
            if not isinstance(values, list) or not set(values).issubset(allowed_ids):
                raise ValueError(f"{field} contains an unknown rule ID")
        self._validate_confidence(result)
        return result

    def _generate_and_validate(self, prompt, validator):
        errors = []
        for attempt in range(1, self.retry_times + 1):
            response = None
            try:
                response = self.llm.generate(prompt)
                result = validator(self._parse(response))
                result["attempts"] = attempt
                return result
            except Exception as exc:
                errors.append(str(exc))
                logger.warning(
                    "Final judge attempt %s/%s failed validation: %s",
                    attempt,
                    self.retry_times,
                    exc,
                )
                if response is not None:
                    detail_logger.debug(
                        "Final judge invalid raw response (attempt %s): %s",
                        attempt,
                        response,
                    )
                if attempt < self.retry_times:
                    time.sleep(1)
        raise RuntimeError(f"Final judge failed validation: {errors}")

    def judge(self, code: str, summary: Dict[str, Any]) -> Dict[str, Any]:
        allowed_ids = [item["rule_id"] for item in summary["rule_results"]]
        return self._generate_and_validate(
            self.build_single_prompt(code, summary),
            lambda result: self._validate_single(result, allowed_ids),
        )

    def judge_pair(
        self,
        before_code: str,
        after_code: str,
        before_summary: Dict[str, Any],
        after_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        allowed_ids = list(
            {
                item["rule_id"]
                for item in before_summary["rule_results"]
                + after_summary["rule_results"]
            }
        )
        return self._generate_and_validate(
            self.build_pair_prompt(
                before_code, after_code, before_summary, after_summary
            ),
            lambda result: self._validate_pair(result, allowed_ids),
        )
