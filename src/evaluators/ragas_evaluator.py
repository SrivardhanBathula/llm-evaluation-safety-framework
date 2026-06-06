"""
RAGAS Evaluation Suite
Full RAG quality evaluation: faithfulness, answer relevancy,
context precision, context recall with statistical aggregation.
Includes offline benchmarks and regression suite.
"""

import json
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Baseline thresholds — what "good" looks like
QUALITY_THRESHOLDS = {
    "faithfulness": 0.85,        # Min: answer must be grounded in context
    "answer_relevancy": 0.80,    # Min: answer must address the question
    "context_precision": 0.75,   # Min: retrieved context must be relevant
    "context_recall": 0.75,      # Min: context must cover ground truth
}

# Regression baselines — alert if scores drop below these
REGRESSION_BASELINES = {
    "faithfulness": 0.90,
    "answer_relevancy": 0.88,
    "context_precision": 0.85,
    "context_recall": 0.83,
}


@dataclass
class RAGASResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    passed: bool = False


@dataclass
class RAGASReport:
    total_samples: int
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_precision: float
    mean_context_recall: float
    pass_rate: float
    regression_failures: list[str] = field(default_factory=list)
    results: list[RAGASResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def passed_regression(self) -> bool:
        return len(self.regression_failures) == 0


class RAGASEvaluator:
    """
    RAGAS-based RAG quality evaluator with regression suite.
    Tracks metric drift across versions and alerts on regressions.
    """

    def __init__(
        self,
        thresholds: Optional[dict] = None,
        regression_baselines: Optional[dict] = None,
        results_dir: str = "eval_results",
    ):
        self.thresholds = thresholds or QUALITY_THRESHOLDS
        self.regression_baselines = regression_baselines or REGRESSION_BASELINES
        self.results_dir = Path(results_dir)

    def evaluate(self, samples: list[dict]) -> RAGASReport:
        """
        Run full RAGAS evaluation.
        samples: [{"question": ..., "answer": ..., "contexts": [...], "ground_truth": ...}]
        """
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness, answer_relevancy,
                context_precision, context_recall,
            )
            from datasets import Dataset
        except ImportError:
            logger.error("Install: pip install ragas datasets")
            return self._empty_report()

        logger.info(f"Running RAGAS on {len(samples)} samples...")

        dataset = Dataset.from_dict({
            "question": [s["question"] for s in samples],
            "answer": [s["answer"] for s in samples],
            "contexts": [s["contexts"] for s in samples],
            "ground_truth": [s.get("ground_truth", "") for s in samples],
        })

        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

        results = []
        for i, sample in enumerate(samples):
            r = RAGASResult(
                question=sample["question"],
                answer=sample["answer"],
                contexts=sample["contexts"],
                ground_truth=sample.get("ground_truth", ""),
                faithfulness=float(scores["faithfulness"][i]),
                answer_relevancy=float(scores["answer_relevancy"][i]),
                context_precision=float(scores["context_precision"][i]),
                context_recall=float(scores["context_recall"][i]),
            )
            r.passed = all([
                r.faithfulness >= self.thresholds["faithfulness"],
                r.answer_relevancy >= self.thresholds["answer_relevancy"],
                r.context_precision >= self.thresholds["context_precision"],
                r.context_recall >= self.thresholds["context_recall"],
            ])
            results.append(r)

        report = self._build_report(results)
        self._check_regression(report)
        self._save_results(report)
        return report

    def _build_report(self, results: list[RAGASResult]) -> RAGASReport:
        return RAGASReport(
            total_samples=len(results),
            mean_faithfulness=round(statistics.mean(r.faithfulness for r in results), 4),
            mean_answer_relevancy=round(statistics.mean(r.answer_relevancy for r in results), 4),
            mean_context_precision=round(statistics.mean(r.context_precision for r in results), 4),
            mean_context_recall=round(statistics.mean(r.context_recall for r in results), 4),
            pass_rate=round(sum(1 for r in results if r.passed) / len(results), 4),
            results=results,
        )

    def _check_regression(self, report: RAGASReport):
        """Compare against baselines and flag regressions."""
        checks = {
            "faithfulness": report.mean_faithfulness,
            "answer_relevancy": report.mean_answer_relevancy,
            "context_precision": report.mean_context_precision,
            "context_recall": report.mean_context_recall,
        }
        for metric, value in checks.items():
            baseline = self.regression_baselines.get(metric, 0)
            if value < baseline:
                msg = f"{metric}: {value:.4f} < baseline {baseline:.4f}"
                report.regression_failures.append(msg)
                logger.warning(f"REGRESSION DETECTED: {msg}")

        if report.passed_regression:
            logger.info("Regression suite: PASSED")
        else:
            logger.error(f"Regression suite: FAILED ({len(report.regression_failures)} failures)")

    def _save_results(self, report: RAGASReport):
        """Persist results for trend tracking."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        path = self.results_dir / f"ragas_{report.timestamp[:10]}.json"
        summary = {
            "timestamp": report.timestamp,
            "total_samples": report.total_samples,
            "faithfulness": report.mean_faithfulness,
            "answer_relevancy": report.mean_answer_relevancy,
            "context_precision": report.mean_context_precision,
            "context_recall": report.mean_context_recall,
            "pass_rate": report.pass_rate,
            "regression_passed": report.passed_regression,
            "regression_failures": report.regression_failures,
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved to {path}")

    def _empty_report(self) -> RAGASReport:
        return RAGASReport(
            total_samples=0,
            mean_faithfulness=0.0,
            mean_answer_relevancy=0.0,
            mean_context_precision=0.0,
            mean_context_recall=0.0,
            pass_rate=0.0,
        )

    def print_report(self, report: RAGASReport):
        print("\n" + "=" * 55)
        print("RAGAS EVALUATION REPORT")
        print("=" * 55)
        print(f"  Samples:             {report.total_samples}")
        print(f"  Pass Rate:           {report.pass_rate:.1%}")
        print(f"  Faithfulness:        {report.mean_faithfulness:.4f}  (threshold: {self.thresholds['faithfulness']})")
        print(f"  Answer Relevancy:    {report.mean_answer_relevancy:.4f}  (threshold: {self.thresholds['answer_relevancy']})")
        print(f"  Context Precision:   {report.mean_context_precision:.4f}  (threshold: {self.thresholds['context_precision']})")
        print(f"  Context Recall:      {report.mean_context_recall:.4f}  (threshold: {self.thresholds['context_recall']})")
        print(f"  Regression:          {'PASSED' if report.passed_regression else 'FAILED'}")
        if report.regression_failures:
            for f in report.regression_failures:
                print(f"    - {f}")
        print("=" * 55 + "\n")
