"""Generate a label-free query specialized for historical vulnerability retrieval."""

import json
import re
import time


class HistoricalQueryAgent:
    FIELDS = (
        "functional_scenario",
        "processed_objects",
        "external_inputs",
        "security_sensitive_operations",
        "possible_failure_conditions",
        "existing_protections",
    )

    def __init__(self, llm, retry_times=3):
        self.llm = llm
        self.retry_times = retry_times

    @staticmethod
    def build_prompt(code):
        return f"""
Generate a retrieval query for a historical C/C++ vulnerability knowledge base.
Analyze only the supplied function. Describe its functional scenario, processed
objects, externally influenced inputs, security-sensitive operations, possible
failure conditions visible from the code, and existing checks or mitigations.

Do not decide whether the function is vulnerable. Do not mention CWE, CVE,
vulnerability labels, patch status, before/after identity, or paired code.

Source code:
```c
{code}
```

Return only valid JSON:
{{
  "functional_scenario": "",
  "processed_objects": [],
  "external_inputs": [],
  "security_sensitive_operations": [],
  "possible_failure_conditions": [],
  "existing_protections": []
}}
"""

    @staticmethod
    def _parse(response):
        cleaned = response.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", cleaned, re.S)
        return json.loads(match.group() if match else cleaned)

    @classmethod
    def _validate(cls, value):
        if not isinstance(value, dict):
            raise ValueError("historical query must be an object")
        if not isinstance(value.get("functional_scenario"), str):
            raise ValueError("functional_scenario must be a string")
        for field in cls.FIELDS[1:]:
            if not isinstance(value.get(field), list) or not all(
                isinstance(item, str) for item in value[field]
            ):
                raise ValueError(f"{field} must be a list of strings")
        return {field: value[field] for field in cls.FIELDS}

    @staticmethod
    def render_query(value):
        labels = {
            "functional_scenario": "Functional scenario",
            "processed_objects": "Processed objects",
            "external_inputs": "External inputs",
            "security_sensitive_operations": "Security-sensitive operations",
            "possible_failure_conditions": "Possible failure conditions",
            "existing_protections": "Existing protections",
        }
        lines = []
        for field in HistoricalQueryAgent.FIELDS:
            content = value[field]
            if isinstance(content, list):
                content = "; ".join(content) if content else "Not visible"
            lines.append(f"{labels[field]}: {content}")
        return "\n".join(lines)

    def analyze(self, code):
        errors = []
        for attempt in range(1, self.retry_times + 1):
            try:
                value = self._validate(self._parse(self.llm.generate(self.build_prompt(code))))
                return {
                    "query": value,
                    "query_text": self.render_query(value),
                    "status": "success",
                    "attempts": attempt,
                }
            except Exception as exc:
                errors.append(str(exc))
                if attempt < self.retry_times:
                    time.sleep(1)
        raise RuntimeError(f"Historical query generation failed: {errors}")

