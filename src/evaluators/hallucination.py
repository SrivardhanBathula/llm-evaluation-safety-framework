"""
Hallucination Evaluator
Faithfulness scoring + factual consistency checking using LLM-as-a-Judge (GPT-4o).
Core evaluator for RAG and generative AI systems in production.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

FAITHFULNESS_JUDGE_PROMPT = """You are a strict factual auditor evaluating whether an AI-generated answer is faithful to the provided context.

Your task: For each claim in the answer, determine whether it is directly supported by the context.

Context:
{context}

Question:
{question}

Answer:
{answer}

Evaluate each claim and respond ONLY with valid JSON in this exact format:
{{
  "faithful": true or false,
  "faithfulness_score": float between 0.0 and 1.0,
  "total_claims": integer,
  "supported_claims": integer,
  "unsupported_claims": ["list of unsupported claim strings"],
  "reasoning": "brief explanation"
}}"""

FACTUAL_CONSISTENCY_PROMPT = """You are evaluating factual consistency between two texts.

Reference Text (ground truth):
{reference}

Generated Text:
{generated}

Assess whether the generated text contains factual contradictions or additions not present in the reference.
Respond ONLY with valid JSON:
{{
  "consistent": true or false,
  "consistency_score": float 0.0-1.0,
  "contradictions": ["list of contradictions found"],
  "hallucinated_facts": ["facts in generated not in reference"]
}}"""


@dataclass
class HallucinationResult:
    question: str
    answer: str
    context: str
    faithful: bool
    faithfulness_score: float
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: list[str] = field(default_factory=list)
    reasoning: str = ""
    passed: bool = False  # True if faithfulness_score >= threshold


@dataclass
class ConsistencyResult:
    consistent: bool
    consistency_score: float
    contradictions: list[str] = field(default_factory=list)
    hallucinated_facts: list[str] = field(default_factory=list)


@dataclass
class EvaluationSummary:
    total_samples: int
    passed: int
    failed: int
    pass_rate: float
    mean_faithfulness_score: float
    min_faithfulness_score: float
    mean_unsupported_claims: float
    results: list[HallucinationResult] = field(default_factory=list)


class HallucinationEvaluator:
    """
    LLM-as-a-Judge hallucination evaluator.
    Uses GPT-4o to assess faithfulness of model outputs against retrieved context.
    """

    def __init__(
        self,
        judge_model: str = "gpt-4o",
        faithfulness_threshold: float = 0.85,
        temperature: float = 0.0,
    ):
        self.client = OpenAI()
        self.judge_model = judge_model
        self.faithfulness_threshold = faithfulness_threshold
        self.temperature = temperature

    def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> HallucinationResult:
        """
        Evaluate whether an answer is grounded in the provided context.
        Returns per-claim faithfulness assessment.
        """
        prompt = FAITHFULNESS_JUDGE_PROMPT.format(
            context=context[:4000],
            question=question,
            answer=answer,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                temperature=self.temperature,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            data = json.loads(match.group()) if match else {}

            score = float(data.get("faithfulness_score", 0.0))
            result = HallucinationResult(
                question=question,
                answer=answer,
                context=context,
                faithful=bool(data.get("faithful", False)),
                faithfulness_score=score,
                total_claims=int(data.get("total_claims", 0)),
                supported_claims=int(data.get("supported_claims", 0)),
                unsupported_claims=data.get("unsupported_claims", []),
                reasoning=data.get("reasoning", ""),
                passed=score >= self.faithfulness_threshold,
            )

            log_fn = logger.info if result.passed else logger.warning
            log_fn(f"Faithfulness: {score:.3f} ({'PASS' if result.passed else 'FAIL'}) | "
                   f"Unsupported claims: {len(result.unsupported_claims)}")
            return result

        except Exception as e:
            logger.error(f"Faithfulness evaluation error: {e}")
            return HallucinationResult(
                question=question, answer=answer, context=context,
                faithful=False, faithfulness_score=0.0, passed=False,
                reasoning=f"Evaluation error: {e}",
            )

    def evaluate_factual_consistency(self, reference: str, generated: str) -> ConsistencyResult:
        """
        Check generated text for factual contradictions against a reference.
        Useful for summarization and knowledge-grounded generation evaluation.
        """
        prompt = FACTUAL_CONSISTENCY_PROMPT.format(
            reference=reference[:3000],
            generated=generated,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                temperature=self.temperature,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            data = json.loads(match.group()) if match else {}
            return ConsistencyResult(
                consistent=bool(data.get("consistent", False)),
                consistency_score=float(data.get("consistency_score", 0.0)),
                contradictions=data.get("contradictions", []),
                hallucinated_facts=data.get("hallucinated_facts", []),
            )
        except Exception as e:
            logger.error(f"Consistency evaluation error: {e}")
            return ConsistencyResult(consistent=False, consistency_score=0.0)

    def evaluate_batch(
        self,
        samples: list[dict],
        show_progress: bool = True,
    ) -> EvaluationSummary:
        """
        Evaluate a batch of QA samples.
        Each sample: {"question": ..., "answer": ..., "context": ...}
        """
        results = []
        for i, sample in enumerate(samples):
            if show_progress:
                logger.info(f"Evaluating sample {i+1}/{len(samples)}...")
            result = self.evaluate_faithfulness(
                question=sample["question"],
                answer=sample["answer"],
                context=sample["context"],
            )
            results.append(result)

        scores = [r.faithfulness_score for r in results]
        passed = sum(1 for r in results if r.passed)

        summary = EvaluationSummary(
            total_samples=len(results),
            passed=passed,
            failed=len(results) - passed,
            pass_rate=passed / len(results) if results else 0.0,
            mean_faithfulness_score=sum(scores) / len(scores) if scores else 0.0,
            min_faithfulness_score=min(scores) if scores else 0.0,
            mean_unsupported_claims=sum(len(r.unsupported_claims) for r in results) / len(results) if results else 0.0,
            results=results,
        )

        logger.info(
            f"Batch evaluation complete | "
            f"Pass rate: {summary.pass_rate:.1%} | "
            f"Mean faithfulness: {summary.mean_faithfulness_score:.3f}"
        )
        return summary

    def generate_report(self, summary: EvaluationSummary) -> dict:
        """Generate structured evaluation report for CI/CD or dashboards."""
        failures = [r for r in summary.results if not r.passed]
        return {
            "summary": {
                "total": summary.total_samples,
                "passed": summary.passed,
                "failed": summary.failed,
                "pass_rate": round(summary.pass_rate, 4),
                "mean_faithfulness_score": round(summary.mean_faithfulness_score, 4),
                "threshold": self.faithfulness_threshold,
                "judge_model": self.judge_model,
            },
            "failure_cases": [
                {
                    "question": r.question,
                    "faithfulness_score": r.faithfulness_score,
                    "unsupported_claims": r.unsupported_claims,
                    "reasoning": r.reasoning,
                }
                for r in failures
            ],
        }
