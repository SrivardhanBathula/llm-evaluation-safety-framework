"""
LLM-as-a-Judge Scoring Framework
GPT-4o judges LLM outputs on configurable quality dimensions:
helpfulness, clarity, accuracy, completeness, and tone.
Supports custom rubrics and batch evaluation with statistical aggregation.
"""

import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_RUBRIC = {
    "helpfulness": {
        "description": "Does the response fully address what the user asked for?",
        "scale": "1 (not helpful) to 5 (extremely helpful)",
    },
    "accuracy": {
        "description": "Are the facts and information in the response correct?",
        "scale": "1 (many errors) to 5 (fully accurate)",
    },
    "clarity": {
        "description": "Is the response clear, well-structured, and easy to understand?",
        "scale": "1 (very unclear) to 5 (perfectly clear)",
    },
    "completeness": {
        "description": "Does the response cover all relevant aspects of the question?",
        "scale": "1 (very incomplete) to 5 (comprehensive)",
    },
    "conciseness": {
        "description": "Is the response appropriately concise without unnecessary padding?",
        "scale": "1 (very verbose/too short) to 5 (perfectly sized)",
    },
}

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing AI language model responses.
You evaluate responses objectively based on the provided rubric.
Be strict, fair, and consistent. Always provide specific reasoning for your scores.
Respond ONLY with valid JSON — no preamble, no markdown."""

JUDGE_PROMPT_TEMPLATE = """Evaluate this AI response on the following dimensions:

Question: {question}
AI Response: {response}
{reference_section}

Scoring Rubric:
{rubric}

Provide scores and reasoning for EACH dimension. Respond ONLY with JSON:
{{
  "scores": {{
    "helpfulness": {{"score": 1-5, "reasoning": "..."}},
    "accuracy": {{"score": 1-5, "reasoning": "..."}},
    "clarity": {{"score": 1-5, "reasoning": "..."}},
    "completeness": {{"score": 1-5, "reasoning": "..."}},
    "conciseness": {{"score": 1-5, "reasoning": "..."}}
  }},
  "overall_score": float (1.0-5.0, weighted average),
  "overall_reasoning": "overall assessment",
  "recommendation": "pass|review|fail"
}}"""


@dataclass
class DimensionScore:
    score: float
    reasoning: str


@dataclass
class JudgeResult:
    question: str
    response: str
    scores: dict[str, DimensionScore]
    overall_score: float
    overall_reasoning: str
    recommendation: str   # pass | review | fail
    reference: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.recommendation == "pass"


@dataclass
class BatchJudgeReport:
    total: int
    passed: int
    review: int
    failed: int
    pass_rate: float
    mean_overall_score: float
    dimension_means: dict[str, float]
    results: list[JudgeResult] = field(default_factory=list)


class LLMJudge:
    """
    LLM-as-a-Judge evaluation framework using GPT-4o.
    Configurable rubrics, reference-based and reference-free modes,
    and batch evaluation with statistical reporting.
    """

    def __init__(
        self,
        judge_model: str = "gpt-4o",
        rubric: Optional[dict] = None,
        pass_threshold: float = 3.5,
        temperature: float = 0.0,
    ):
        self.client = OpenAI()
        self.judge_model = judge_model
        self.rubric = rubric or DEFAULT_RUBRIC
        self.pass_threshold = pass_threshold
        self.temperature = temperature

    def _format_rubric(self) -> str:
        lines = []
        for dim, info in self.rubric.items():
            lines.append(f"- {dim.upper()}: {info['description']} ({info['scale']})")
        return "\n".join(lines)

    def score(
        self,
        question: str,
        response: str,
        reference: Optional[str] = None,
    ) -> JudgeResult:
        """
        Score a single LLM response across all rubric dimensions.
        Optionally provide a reference answer for comparison.
        """
        ref_section = f"\nReference Answer: {reference}" if reference else ""
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            response=response,
            reference_section=ref_section,
            rubric=self._format_rubric(),
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.judge_model,
                temperature=self.temperature,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            data = json.loads(match.group()) if match else {}

            scores = {}
            for dim, val in data.get("scores", {}).items():
                scores[dim] = DimensionScore(
                    score=float(val.get("score", 0)),
                    reasoning=val.get("reasoning", ""),
                )

            overall = float(data.get("overall_score", sum(s.score for s in scores.values()) / len(scores) if scores else 0))

            # Determine recommendation if not provided
            rec = data.get("recommendation", "pass" if overall >= self.pass_threshold else "fail")

            result = JudgeResult(
                question=question,
                response=response,
                scores=scores,
                overall_score=overall,
                overall_reasoning=data.get("overall_reasoning", ""),
                recommendation=rec,
                reference=reference,
            )
            logger.info(f"Judge score: {overall:.2f}/5.0 → {rec.upper()}")
            return result

        except Exception as e:
            logger.error(f"Judge error: {e}")
            return JudgeResult(
                question=question, response=response, scores={},
                overall_score=0.0, overall_reasoning=f"Error: {e}",
                recommendation="review",
            )

    def score_batch(self, samples: list[dict]) -> BatchJudgeReport:
        """
        Score a batch of samples.
        Each sample: {"question": ..., "response": ..., "reference": ... (optional)}
        """
        results = []
        for i, sample in enumerate(samples):
            logger.info(f"Judging sample {i+1}/{len(samples)}...")
            result = self.score(
                question=sample["question"],
                response=sample["response"],
                reference=sample.get("reference"),
            )
            results.append(result)

        overall_scores = [r.overall_score for r in results]

        # Per-dimension means
        dimension_scores: dict[str, list[float]] = {}
        for result in results:
            for dim, ds in result.scores.items():
                dimension_scores.setdefault(dim, []).append(ds.score)
        dimension_means = {dim: round(statistics.mean(scores), 3) for dim, scores in dimension_scores.items()}

        passed = sum(1 for r in results if r.recommendation == "pass")
        review = sum(1 for r in results if r.recommendation == "review")
        failed = sum(1 for r in results if r.recommendation == "fail")

        report = BatchJudgeReport(
            total=len(results),
            passed=passed,
            review=review,
            failed=failed,
            pass_rate=passed / len(results) if results else 0.0,
            mean_overall_score=statistics.mean(overall_scores) if overall_scores else 0.0,
            dimension_means=dimension_means,
            results=results,
        )

        logger.info(
            f"Batch judging complete | Pass: {report.pass_rate:.1%} | "
            f"Mean score: {report.mean_overall_score:.2f}/5.0"
        )
        return report

    def compare_models(
        self,
        question: str,
        responses: dict[str, str],
        reference: Optional[str] = None,
    ) -> dict[str, JudgeResult]:
        """
        Compare multiple model responses side-by-side on the same question.
        Returns dict of model_name → JudgeResult, sorted by overall score.
        """
        scored = {
            name: self.score(question, resp, reference)
            for name, resp in responses.items()
        }
        return dict(sorted(scored.items(), key=lambda x: x[1].overall_score, reverse=True))
