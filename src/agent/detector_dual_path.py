"""VulDuet detector with Vul-RAG-style historical retrieval."""

import time

from src.agent.dual_path_core import DualPathCore
from src.agent.historical_query_agent import HistoricalQueryAgent
from src.agent.vulnerability_reranker import VulnerabilityReranker
from src.util.dual_path_evidence import normalize_dual_path_evidence


class DualPathDetector(DualPathCore):
    def __init__(self, llm, rule_index_path, rule_document_path,
                 knowledge_document_path, rule_top_k=5, knowledge_retrieval_top_k=5,
                 knowledge_view_top_k=20,
                 knowledge_reasoning_top_k=3, minimum_similarity=0.0,
                 minimum_confidence=0.5, minimum_scenario_match=0.3,
                 reasoning_workers=5):
        if knowledge_retrieval_top_k < knowledge_reasoning_top_k:
            raise ValueError("knowledge_retrieval_top_k must be >= knowledge_reasoning_top_k")
        super().__init__(
            llm, rule_index_path, rule_document_path,
            knowledge_document_path, rule_top_k=rule_top_k,
            knowledge_top_k=knowledge_retrieval_top_k,
            knowledge_view_top_k=knowledge_view_top_k,
            minimum_similarity=minimum_similarity,
            minimum_confidence=minimum_confidence,
            minimum_scenario_match=minimum_scenario_match,
            reasoning_workers=reasoning_workers,
        )
        self.historical_query_agent = HistoricalQueryAgent(llm)
        self.knowledge_reranker = VulnerabilityReranker(
            llm, reasoning_top_k=knowledge_reasoning_top_k
        )
        self.knowledge_retrieval_top_k = knowledge_retrieval_top_k
        self.knowledge_view_top_k = knowledge_view_top_k
        self.knowledge_reasoning_top_k = knowledge_reasoning_top_k

    def detect(self, code, code_id=None):
        started = time.perf_counter()

        t = time.perf_counter()
        analysis = self.rule_path.analysis_agent.analyze(code)
        analysis_seconds = round(time.perf_counter() - t, 4)
        # 检索关键词 规则描述
        rule_description = analysis["security_description"]

        t = time.perf_counter()
        # 检索关键词 漏洞描述
        historical_query = self.historical_query_agent.analyze(code)
        query_seconds = round(time.perf_counter() - t, 4)

        t = time.perf_counter()
        # 安全路径检索
        retrieved_rules = self.rule_path.retriever.retrieve(rule_description)
        retrieved_rules = [
            item for item in retrieved_rules
            if float(item.get("similarity", 0.0)) >= self.rule_path.minimum_similarity
        ]
        rule_retrieval_seconds = round(time.perf_counter() - t, 4)

        t = time.perf_counter()
        # 检索top-k条历史漏洞案例
        candidates = self.knowledge_retriever.retrieve(
            historical_query["query"], code=code
        )
        knowledge_retrieval_seconds = round(time.perf_counter() - t, 4)

        t = time.perf_counter()
        # 历史漏洞排序
        selected, ranked, rerank_diagnostics = self.knowledge_reranker.select(
            historical_query["query_text"], candidates
        )
        rerank_seconds = round(time.perf_counter() - t, 4)

        t = time.perf_counter()
        rule_reasoning, knowledge_reasoning = self._reason_in_parallel(
            code, retrieved_rules, selected
        )
        parallel_reasoning_seconds = round(time.perf_counter() - t, 4)

        rule_summary = self.rule_path.aggregator.aggregate(
            rule_description, retrieved_rules, rule_reasoning
        )
        knowledge_summary = self.knowledge_aggregator.aggregate(
            historical_query["query_text"], selected, knowledge_reasoning
        )
        # 路径仲裁 vulnerable benign unknown
        rule_summary["path_decision"] = (
            "vulnerable" if rule_summary["supported_results"] else
            ("benign" if rule_summary["opposed_results"] else "unknown")
        )
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
        # 统一证据格式
        unified = normalize_dual_path_evidence(rule_summary, knowledge_summary)
        path_decisions = {
            "rule": rule_summary["path_decision"],
            "knowledge": knowledge_summary["path_decision"],
        }
        t = time.perf_counter()
        final = self.arbiter.judge(code, unified, path_decisions)
        arbiter_seconds = round(time.perf_counter() - t, 4)

        selected_ids = {item["knowledge_id"] for item in selected}
        retrieval_rank = {
            item["knowledge_id"]: rank for rank, item in enumerate(candidates, 1)
        }
        retrieval_diagnostics = []
        for row in ranked:
            retrieval_diagnostics.append({
                **row,
                "retrieval_rank": retrieval_rank[row["knowledge_id"]],
                "retrieval_score": float(
                    next(item["similarity"] for item in candidates
                         if item["knowledge_id"] == row["knowledge_id"])
                ),
                "selected_for_reasoning": row["knowledge_id"] in selected_ids,
            })

        return {
            "code_id": code_id,
            "analysis_result": rule_result["analysis_result"],
            "rule_path": rule_result,
            "knowledge_path": {
                "historical_query": historical_query,
                "retrieval_candidates": candidates,
                "reranked_candidates": retrieval_diagnostics,
                "selected_knowledge": selected,
                "reasoning_results": knowledge_reasoning,
                "summary": knowledge_summary,
                "reranker": rerank_diagnostics,
            },
            "unified_evidence": unified,
            "path_decisions": path_decisions,
            "final_result": final,
            "timing_seconds": {
                "analysis": analysis_seconds,
                "historical_query": query_seconds,
                "rule_retrieval": rule_retrieval_seconds,
                "knowledge_retrieval": knowledge_retrieval_seconds,
                "knowledge_rerank": rerank_seconds,
                "parallel_reasoning": parallel_reasoning_seconds,
                "arbiter": arbiter_seconds,
                "total": round(time.perf_counter() - started, 4),
            },
            "metadata": {
                "architecture": "dual_path_vulrag_retrieval",
                "historical_retriever": "vulrag_bm25",
                "knowledge_view_top_k": self.knowledge_view_top_k,
                "knowledge_retrieval_top_k": self.knowledge_retrieval_top_k,
                "knowledge_reasoning_top_k": self.knowledge_reasoning_top_k,
                "reasoning_workers": self.reasoning_workers,
                "arbiter": "unchanged",
            },
        }

