"""Build the self-contained historical knowledge base used by retrieval.

The extracted knowledge documents contain the abstract cause/trigger/repair
information used by the reasoner.  PrimeVul's cleaned training records contain
the vulnerable and patched functions.  ``source_id`` links the two sources.
This command performs a strict join and writes a portable knowledge file so a
deployed experiment does not depend on the original training dataset.
"""

import argparse
import json
from pathlib import Path


def _normalized_id(value):
    return str(value).strip()


def build_knowledge(documents_path, sources_path, output_path):
    documents = json.loads(Path(documents_path).read_text(encoding="utf-8"))
    sources = json.loads(Path(sources_path).read_text(encoding="utf-8"))
    sources_by_id = {_normalized_id(item["id"]): item for item in sources}

    if len(sources_by_id) != len(sources):
        raise ValueError("training source ids are not unique")

    enhanced = []
    missing = []
    mismatches = []
    for document in documents:
        source_id = _normalized_id(document.get("source_id", ""))
        source = sources_by_id.get(source_id)
        if source is None:
            missing.append(source_id)
            continue

        expected = {
            "cve_id": document.get("cve_id"),
            "commit_id": document.get("provenance", {}).get("commit_id"),
        }
        observed = {
            "cve_id": source.get("cve_id"),
            "commit_id": source.get("commit_id"),
        }
        if any(expected[key] and expected[key] != observed[key] for key in expected):
            mismatches.append({
                "knowledge_id": document.get("knowledge_id"),
                "source_id": source_id,
                "expected": expected,
                "observed": observed,
            })
            continue

        functional = document.get("functional_semantics", {})
        item = dict(document)
        item["retrieval_views"] = {
            "purpose": functional.get("purpose", ""),
            "function": functional.get("security_relevant_behavior", ""),
            "vulnerable_code": source.get("code_before_change", ""),
        }
        # Retain both sides for later positive/negative retrieval experiments;
        # the current Vul-RAG retriever indexes only vulnerable_code.
        item["historical_code"] = {
            "vulnerable": source.get("code_before_change", ""),
            "patched": source.get("code_after_change", ""),
        }
        item["source_metadata"] = {
            "project": source.get("project"),
            "source_row_ids": source.get("source_row_ids", []),
        }
        enhanced.append(item)

    if missing or mismatches:
        raise ValueError(
            f"knowledge join failed: missing={len(missing)}, "
            f"mismatches={len(mismatches)}; "
            f"examples={{'missing': {missing[:3]}, 'mismatches': {mismatches[:3]}}}"
        )
    if len(enhanced) != len(documents):
        raise ValueError("enhanced knowledge count differs from document count")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(enhanced, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "documents": len(documents),
        "sources": len(sources),
        "written": len(enhanced),
        "output": str(output.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents")
    parser.add_argument("sources")
    parser.add_argument("output")
    args = parser.parse_args()
    print(json.dumps(build_knowledge(
        args.documents, args.sources, args.output
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
