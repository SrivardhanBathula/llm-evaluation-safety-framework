from deepeval import evaluate
from deepeval.metrics import HallucinationMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """Detects LLM hallucinations using DeepEval + GPT-4o judge."""

    def __init__(self, threshold: float = 0.7, model: str = "gpt-4o"):
        self.threshold = threshold
        self.hallucination_metric = HallucinationMetric(threshold=threshold, model=model)
        self.faithfulness_metric = FaithfulnessMetric(threshold=threshold, model=model)

    def evaluate_single(self, input_text: str, actual_output: str,
                       context: List[str]) -> Dict[str, Any]:
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            context=context
        )
        self.hallucination_metric.measure(test_case)
        self.faithfulness_metric.measure(test_case)
        return {
            "hallucination_score": self.hallucination_metric.score,
            "hallucination_passed": self.hallucination_metric.is_successful(),
            "faithfulness_score": self.faithfulness_metric.score,
            "faithfulness_passed": self.faithfulness_metric.is_successful(),
            "reason": self.hallucination_metric.reason
        }

    def batch_evaluate(self, samples: List[Dict]) -> List[Dict]:
        results = []
        for sample in samples:
            result = self.evaluate_single(
                sample["input"], sample["output"], sample.get("context", [])
            )
            results.append(result)
        pass_rate = sum(1 for r in results if r["hallucination_passed"]) / len(results)
        logger.info(f"Hallucination evaluation: {pass_rate:.1%} pass rate on {len(results)} samples")
        return results
