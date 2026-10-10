"""Evidence-preserving detector with single-function and patch-pair modes."""

import time
from typing import Any, Dict

from src.agent.analysis_agent import AnalysisAgent
from src.agent.judge_agent import JudgeAgent
from src.agent.rule_reasoner import RuleReasoner
from src.retrieval.rule_retriever import RuleRetriever
from src.util.evidence_aggregator import EvidenceAggregator


class RuleDetector:
    def __init__(
        self,
        llm,
        index_path: str,
        document_path: str,
        top_k: int = 5,
        minimum_similarity: float = 0.0,
        minimum_confidence: float = 0.5,
    ):
        self.top_k = top_k
        self.minimum_similarity = minimum_similarity
        self.analysis_agent = AnalysisAgent(llm)
        self.reasoner = RuleReasoner(llm)
        self.judge = JudgeAgent(llm)
        self.aggregator = EvidenceAggregator(
            minimum_similarity=minimum_similarity,
            minimum_confidence=minimum_confidence,
        )
        self.retriever = RuleRetriever(
            index_path=index_path,
            document_path=document_path,
            top_k=top_k,
        )

    @staticmethod
    def _run_timed(function, *args, **kwargs):
        started = time.perf_counter()
        value = function(*args, **kwargs)
        return value, round(time.perf_counter() - started, 4)

    def analyze_with_rules(self, code: str, code_id=None) -> Dict[str, Any]:
        analysis, analysis_seconds = self._run_timed(
            self.analysis_agent.analyze, code
        )
        # 安全语义
        description = analysis["security_description"]
        retrieved, retrieval_seconds = self._run_timed(
            self.retriever.retrieve, description
        )
        retained = [
            rule
            for rule in retrieved
            if float(rule.get("similarity", 0.0)) >= self.minimum_similarity
        ]

        started = time.perf_counter()
        reasoning = [self.reasoner.reason(code, rule) for rule in retained]
        reasoning_seconds = round(time.perf_counter() - started, 4)

        summary = self.aggregator.aggregate(description, retained, reasoning)
        return {
            "code_id": code_id,
            "analysis_result": analysis,
            "retrieved_rules": retained,
            "reasoning_results": reasoning,
            "summary": summary,
            "timing_seconds": {
                "analysis": analysis_seconds,
                "retrieval": retrieval_seconds,
                "reasoning": reasoning_seconds,
            },
            "metadata": {
                "top_k": self.top_k,
                "minimum_similarity": self.minimum_similarity,
            },
        }

    def detect(self, code: str, code_id=None) -> Dict[str, Any]:
        result = self.analyze_with_rules(code, code_id)
        final_result, judge_seconds = self._run_timed(
            self.judge.judge, code, result["summary"]
        )
        result["final_result"] = final_result
        result["timing_seconds"]["judge"] = judge_seconds
        result["timing_seconds"]["total"] = round(
            sum(result["timing_seconds"].values()), 4
        )
        return result

    def detect_pair(
        self,
        before_code: str,
        after_code: str,
        code_id=None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        before = self.analyze_with_rules(before_code, code_id)
        after = self.analyze_with_rules(after_code, code_id)
        pair_result, pair_judge_seconds = self._run_timed(
            self.judge.judge_pair,
            before_code,
            after_code,
            before["summary"],
            after["summary"],
        )
        return {
            "code_id": code_id,
            "before": before,
            "after": after,
            "pair_result": pair_result,
            "timing_seconds": {
                "pair_judge": pair_judge_seconds,
                "total": round(time.perf_counter() - started, 4),
            },
        }
