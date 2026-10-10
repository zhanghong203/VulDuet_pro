"""Vul-RAG-style multi-view BM25 retrieval for historical vulnerabilities."""

import json
import math
import re
from collections import Counter
from pathlib import Path


class _BM25:
    """Dependency-free BM25Okapi implementation for a fixed corpus."""

    TOKEN_PATTERN = re.compile(
        r"[A-Za-z_][A-Za-z0-9_]*|0x[0-9A-Fa-f]+|\d+|[^\W\d_]",
        re.UNICODE,
    )

    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = [self.tokenize(text) for text in corpus]
        self.lengths = [len(document) for document in self.documents]
        self.average_length = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        )
        self.term_frequencies = [Counter(document) for document in self.documents]
        document_frequency = Counter()
        for document in self.term_frequencies:
            document_frequency.update(document.keys())
        count = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    @classmethod
    def tokenize(cls, text):
        return [token.lower() for token in cls.TOKEN_PATTERN.findall(text or "")]

    def search(self, query, top_k):
        terms = self.tokenize(query)
        if not terms or not self.documents:
            return []
        scores = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            length = self.lengths[index]
            normalization = 1.0 - self.b
            if self.average_length:
                normalization += self.b * length / self.average_length
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * normalization
                score += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1.0) / denominator
                )
            scores.append((index, score))
        scores.sort(key=lambda value: (-value[1], value[0]))
        return scores[: min(top_k, len(scores))]


class HistoricalRetriever:
    """Retrieve cases through Vul-RAG's purpose/function/code views."""

    VIEW_NAMES = ("purpose", "function", "code")

    def __init__(self, document_path, top_k=5, view_top_k=20, **_ignored):
        self.top_k = top_k
        self.view_top_k = max(view_top_k, top_k)
        self.items = json.loads(Path(document_path).read_text(encoding="utf-8"))
        if not self.items:
            raise ValueError("historical knowledge base is empty")
        self.indexes = {
            "purpose": _BM25([
                item.get("retrieval_views", {}).get("purpose", "")
                for item in self.items
            ]),
            "function": _BM25([
                item.get("retrieval_views", {}).get("function", "")
                for item in self.items
            ]),
            "code": _BM25([
                item.get("retrieval_views", {}).get("vulnerable_code", "")
                for item in self.items
            ]),
        }

    @staticmethod
    def _queries(query, code):
        if not isinstance(query, dict):
            return {"purpose": str(query), "function": str(query), "code": code}
        function_parts = []
        for field in (
            "processed_objects",
            "external_inputs",
            "security_sensitive_operations",
            "possible_failure_conditions",
            "existing_protections",
        ):
            value = query.get(field, [])
            if isinstance(value, list):
                function_parts.extend(str(item) for item in value)
            elif value:
                function_parts.append(str(value))
        return {
            "purpose": str(query.get("functional_scenario", "")),
            "function": "\n".join(function_parts),
            "code": code,
        }

    def retrieve(self, query, code=""):
        queries = self._queries(query, code)
        results = {
            view: self.indexes[view].search(queries[view], self.view_top_k)
            for view in self.VIEW_NAMES
        }
        candidates = set()
        ranks = {}
        raw_scores = {}
        for view, rows in results.items():
            ranks[view] = {index: rank for rank, (index, _) in enumerate(rows)}
            raw_scores[view] = {index: score for index, score in rows}
            candidates.update(index for index, _ in rows)

        fused = []
        missing_rank = self.view_top_k
        for index in candidates:
            view_ranks = {
                view: ranks[view].get(index, missing_rank)
                for view in self.VIEW_NAMES
            }
            rank_sum = sum(view_ranks.values())
            fused.append((index, rank_sum, view_ranks))
        fused.sort(key=lambda value: (value[1], value[0]))

        output = []
        for final_rank, (index, rank_sum, view_ranks) in enumerate(
            fused[: self.top_k], 1
        ):
            item = dict(self.items[index])
            maximum_rank_sum = len(self.VIEW_NAMES) * self.view_top_k
            item["similarity"] = max(0.0, 1.0 - rank_sum / maximum_rank_sum)
            item["retrieval"] = {
                "backend": "vulrag_bm25",
                "rank": final_rank,
                "rank_sum": rank_sum,
                "view_ranks": {
                    view: rank + 1 if rank < missing_rank else None
                    for view, rank in view_ranks.items()
                },
                "view_scores": {
                    view: raw_scores[view].get(index, 0.0)
                    for view in self.VIEW_NAMES
                },
            }
            output.append(item)
        return output
