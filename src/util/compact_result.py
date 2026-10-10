"""Create compact, evaluation-ready records from full VulDuet outputs."""


def _select(value, fields):
    if not isinstance(value, dict):
        return {}
    return {field: value[field] for field in fields if field in value}


def _compact_rule_path(path):
    summary = path.get("summary", {})
    retrieved = []
    for item in path.get("retrieved_rules", []):
        retrieved.append(_select(item, (
            "rule_id", "title", "similarity", "section", "category"
        )))
    reasoning = []
    for item in path.get("reasoning_results", []):
        row = _select(item, (
            "rule_id", "violation", "confidence", "status", "attempts"
        ))
        row["evidence"] = item.get("evidence", {})
        reasoning.append(row)
    return {
        "path_decision": summary.get("path_decision"),
        "statistics": summary.get("statistics", {}),
        "retrieved_rules": retrieved,
        "reasoning_results": reasoning,
    }


def _compact_candidate(item):
    return {
        **_select(item, (
            "knowledge_id", "source_id", "cve_id", "cwe_ids", "similarity"
        )),
        "retrieval": item.get("retrieval", {}),
        "rerank": item.get("rerank", {}),
    }


def _compact_knowledge_path(path):
    summary = path.get("summary", {})
    reasoning = []
    for item in path.get("reasoning_results", []):
        row = _select(item, (
            "knowledge_id", "scenario_match", "cause_present",
            "trigger_present", "fix_present", "conclusion", "confidence",
            "status", "attempts"
        ))
        row["evidence"] = item.get("evidence", {})
        reasoning.append(row)
    return {
        "historical_query": path.get("historical_query", {}).get("query", {}),
        "retrieval_candidates": [
            _compact_candidate(item)
            for item in path.get("retrieval_candidates", [])
        ],
        "selected_knowledge": [
            _compact_candidate(item)
            for item in path.get("selected_knowledge", [])
        ],
        "reasoning_results": reasoning,
        "path_decision": summary.get("path_decision"),
        "statistics": summary.get("statistics", {}),
        "reranker": path.get("reranker", {}),
    }


def compact_detection(result):
    final = _select(result.get("final_result", {}), (
        "is_vulnerable", "confidence", "decision_mode", "primary_evidence",
        "conflicts", "violated_rules", "related_cwes", "attempts"
    ))
    return {
        "code_id": result.get("code_id"),
        "path_decisions": result.get("path_decisions", {}),
        "final_result": final,
        "rule_path": _compact_rule_path(result.get("rule_path", {})),
        "knowledge_path": _compact_knowledge_path(
            result.get("knowledge_path", {})
        ),
        "timing_seconds": result.get("timing_seconds", {}),
        "metadata": result.get("metadata", {}),
    }


def compact_pair(result):
    return {
        **_select(result, (
            "id", "commit_id", "cwe", "labels", "model", "experiment"
        )),
        "before": compact_detection(result.get("before", {})),
        "after": compact_detection(result.get("after", {})),
    }
