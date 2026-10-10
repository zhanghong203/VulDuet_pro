"""Shared implementation of the independent rule and historical paths."""

import time
from concurrent.futures import ThreadPoolExecutor

from src.agent.arbiter_agent import ArbiterAgent
from src.agent.rule_detector import RuleDetector
from src.agent.vulnerability_reasoner import VulnerabilityReasoner
from src.retrieval.historical_retriever import HistoricalRetriever
from src.util.dual_path_evidence import VulnerabilityEvidenceAggregator, normalize_dual_path_evidence


class DualPathCore:
    def __init__(self, llm, rule_index_path, rule_document_path,
                 knowledge_document_path, rule_top_k=5, knowledge_top_k=3,
                 knowledge_view_top_k=20,
                 minimum_similarity=0.0, minimum_confidence=0.5,
                 minimum_scenario_match=0.3, reasoning_workers=5):
        self.rule_path = RuleDetector(llm, rule_index_path, rule_document_path, rule_top_k,
                                    minimum_similarity, minimum_confidence)
        self.knowledge_retriever = HistoricalRetriever(
            knowledge_document_path,
            top_k=knowledge_top_k,
            view_top_k=knowledge_view_top_k,
        )
        self.knowledge_reasoner = VulnerabilityReasoner(llm)
        self.knowledge_aggregator = VulnerabilityEvidenceAggregator(minimum_similarity, minimum_scenario_match, minimum_confidence)
        self.arbiter = ArbiterAgent(llm)
        self.knowledge_top_k = knowledge_top_k
        self.reasoning_workers = reasoning_workers

    def _reason_in_parallel(self, code, rules, knowledge):
        """Run both paths' independent evidence tasks in one bounded pool."""
        with ThreadPoolExecutor(
            max_workers=self.reasoning_workers,
            thread_name_prefix="DualPathReasoning",
        ) as pool:
            rule_futures = [
                pool.submit(self.rule_path.reasoner.reason, code, rule)
                for rule in rules
            ]
            knowledge_futures = [
                pool.submit(self.knowledge_reasoner.reason, code, item)
                for item in knowledge
            ]
            # Reading futures in submission order keeps result files stable;
            # the tasks themselves have already been running concurrently.
            rule_results = [future.result() for future in rule_futures]
            knowledge_results = [future.result() for future in knowledge_futures]
        return rule_results, knowledge_results

    def detect(self, code: str, code_id=None):
        started = time.perf_counter()
        t = time.perf_counter()
        analysis = self.rule_path.analysis_agent.analyze(code)
        analysis_seconds = round(time.perf_counter() - t, 4)
        description = analysis["security_description"]

        t = time.perf_counter()
        retrieved_rules = self.rule_path.retriever.retrieve(description)
        retrieved_rules = [
            rule for rule in retrieved_rules
            if float(rule.get("similarity", 0.0)) >= self.rule_path.minimum_similarity
        ]
        rule_retrieval_seconds = round(time.perf_counter() - t, 4)
        t = time.perf_counter()
        retrieved = self.knowledge_retriever.retrieve(description)
        knowledge_retrieval_seconds = round(time.perf_counter() - t, 4)

        t = time.perf_counter()
        rule_reasoning, knowledge_reasoning = self._reason_in_parallel(
            code, retrieved_rules, retrieved
        )
        parallel_reasoning_seconds = round(time.perf_counter() - t, 4)

        rule_summary = self.rule_path.aggregator.aggregate(
            description, retrieved_rules, rule_reasoning
        )
        knowledge_summary = self.knowledge_aggregator.aggregate(
            description, retrieved, knowledge_reasoning
        )
        rule_summary["path_decision"] = "vulnerable" if rule_summary["supported_results"] else ("benign" if rule_summary["opposed_results"] else "unknown")
        rule_result = {
            "code_id": code_id,
            "analysis_result": analysis,
            "retrieved_rules": retrieved_rules,
            "reasoning_results": rule_reasoning,
            "summary": rule_summary,
            "timing_seconds": {
                "analysis": analysis_seconds,
                "retrieval": rule_retrieval_seconds,
                "parallel_reasoning_shared": parallel_reasoning_seconds,
            },
            "metadata": {
                "top_k": self.rule_path.top_k,
                "minimum_similarity": self.rule_path.minimum_similarity,
                "reasoning_workers": self.reasoning_workers,
            },
        }
        unified = normalize_dual_path_evidence(rule_summary, knowledge_summary)
        path_decisions = {"rule": rule_summary["path_decision"], "knowledge": knowledge_summary["path_decision"]}
        t = time.perf_counter()
        final = self.arbiter.judge(code, unified, path_decisions)
        arbiter_seconds = round(time.perf_counter() - t, 4)
        return {"code_id": code_id, "analysis_result": rule_result["analysis_result"],
                "rule_path": rule_result, "knowledge_path": {"retrieved_knowledge": retrieved,
                "reasoning_results": knowledge_reasoning, "summary": knowledge_summary},
                "unified_evidence": unified, "path_decisions": path_decisions,
                "final_result": final, "timing_seconds": {"analysis": analysis_seconds,
                "rule_retrieval": rule_retrieval_seconds,
                "knowledge_retrieval": knowledge_retrieval_seconds,
                "parallel_reasoning": parallel_reasoning_seconds, "arbiter": arbiter_seconds,
                "total": round(time.perf_counter() - started, 4)},
                "metadata": {"architecture": "dual_path", "knowledge_top_k": self.knowledge_top_k,
                "reasoning_workers": self.reasoning_workers}}
