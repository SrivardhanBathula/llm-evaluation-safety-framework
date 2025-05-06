# 🛡️ LLM Evaluation & Safety Framework

> **Comprehensive framework for evaluating LLM quality, safety, and reliability — covering hallucination detection, prompt injection resistance, bias auditing, and LLM-as-a-Judge scoring using DeepEval and Phoenix.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![DeepEval](https://img.shields.io/badge/DeepEval-0.21+-green)](https://deepeval.com)
[![Phoenix](https://img.shields.io/badge/Arize_Phoenix-Tracing-purple)](https://phoenix.arize.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-purple?logo=openai)](https://openai.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

A production-grade LLM evaluation and safety framework that gives AI teams confidence before deploying language models. Covers the full evaluation lifecycle: output quality, factual faithfulness, safety vulnerability testing, algorithmic bias auditing, and continuous monitoring.

Built around real-world needs from financial services and healthcare AI — domains where LLM failures carry regulatory and operational risk.

### 🏆 What This Framework Evaluates

| Dimension | Metrics |
|---|---|
| **Hallucination** | Faithfulness score, factual consistency, source grounding |
| **Answer Quality** | Relevancy, completeness, coherence, fluency |
| **Safety** | Prompt injection resistance, jailbreak detection, toxic output |
| **Bias** | Demographic parity, stereotype detection, disparate impact |
| **RAG Quality** | Context precision, context recall, answer relevancy (RAGAS) |
| **Latency & Cost** | Token usage, p50/p95/p99 latency, cost per query |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  EVALUATION PIPELINE                      │
│                                                          │
│  Test Dataset  →  LLM Under Test  →  Evaluation Suite   │
│                                                          │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐  │
│  │Hallucination│  │   Safety    │  │   Bias Audit     │  │
│  │ Evaluator  │  │  Evaluator  │  │   Evaluator      │  │
│  └────────────┘  └─────────────┘  └──────────────────┘  │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐  │
│  │   RAGAS    │  │  LLM-Judge  │  │ Latency / Cost   │  │
│  │ Evaluator  │  │  Scorer     │  │   Profiler       │  │
│  └────────────┘  └─────────────┘  └──────────────────┘  │
│                                                          │
│              ↓  Aggregated Report  ↓                     │
│   HTML Dashboard │ JSON Results │ Phoenix Traces         │
└──────────────────────────────────────────────────────────┘
```

---

## 🧠 Core Components

### 1. Hallucination Evaluator (`src/evaluators/hallucination.py`)
- **Faithfulness scoring** — measures if model output is grounded in provided context
- **Factual consistency** — cross-references claims against source documents
- **LLM-as-a-Judge** with GPT-4o for nuanced semantic faithfulness assessment
- Threshold-based pass/fail with per-claim breakdown

### 2. Safety Evaluator (`src/safety/`)
- **Prompt injection detection** — tests 50+ injection patterns and jailbreak attempts
- **Toxic output screening** — Perspective API + custom classifier
- **PII leakage detection** — ensures models don't regurgitate training data
- **Refusal rate measurement** — validates model refuses harmful requests correctly

### 3. Bias Auditor (`src/evaluators/bias_auditor.py`)
- **Demographic parity testing** — evaluates consistency across protected attributes
- **Stereotype detection** — flags gendered, racial, or occupational stereotypes
- **Disparate impact scoring** — measures outcome differences across population subgroups
- Generates structured bias report with per-attribute breakdown

### 4. LLM-as-a-Judge (`src/judges/llm_judge.py`)
- GPT-4o judge for subjective quality dimensions (helpfulness, clarity, tone)
- Configurable rubrics and scoring criteria
- Calibration against human preference data
- Batch evaluation with statistical aggregation

### 5. RAGAS Integration (`src/evaluators/ragas_evaluator.py`)
- Full RAGAS suite: faithfulness, answer relevancy, context precision, context recall
- Per-query and aggregate scores
- Integrated with Phoenix for trace-level observability

### 6. Reporting Dashboard (`src/reporting/`)
- HTML evaluation report with charts and per-metric breakdowns
- JSON results export for CI/CD integration
- Phoenix trace visualization for debugging failure cases
- Trend tracking across model versions

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Evaluation Frameworks** | DeepEval, RAGAS, LangChain Evaluation |
| **Observability** | Arize Phoenix, LangSmith |
| **LLM Judge** | OpenAI GPT-4o, Claude 3 Opus |
| **Safety** | Perspective API, custom classifiers |
| **Reporting** | Plotly, Jinja2 HTML reports |
| **CI/CD Integration** | pytest, GitHub Actions |

---

## 📁 Project Structure

```
llm-evaluation-safety-framework/
├── src/
│   ├── evaluators/
│   │   ├── hallucination.py        # Faithfulness + factual consistency
│   │   ├── ragas_evaluator.py      # Full RAGAS evaluation suite
│   │   ├── bias_auditor.py         # Demographic bias + stereotype detection
│   │   └── latency_profiler.py     # Token usage + cost + latency
│   ├── judges/
│   │   └── llm_judge.py            # LLM-as-a-Judge scoring framework
│   ├── safety/
│   │   ├── prompt_injection.py     # Injection + jailbreak resistance tests
│   │   ├── toxicity_detector.py    # Toxic output screening
│   │   └── pii_detector.py         # PII leakage detection
│   ├── reporting/
│   │   ├── report_generator.py     # HTML + JSON report generation
│   │   └── templates/              # Jinja2 HTML templates
│   └── utils/
│       └── dataset_loader.py       # Evaluation dataset utilities
├── tests/
├── configs/
│   └── eval_config.yaml
├── notebooks/
│   ├── 01_Hallucination_Evaluation.ipynb
│   ├── 02_Safety_Red_Teaming.ipynb
│   └── 03_Bias_Audit_Demo.ipynb
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/SrivardhanBathula/llm-evaluation-safety-framework.git
cd llm-evaluation-safety-framework

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY=your_key_here

# Run full evaluation suite
python -m src.evaluators.hallucination --model gpt-4o --dataset configs/eval_dataset.json

# Run safety red-teaming
python -m src.safety.prompt_injection --model gpt-4o

# Run bias audit
python -m src.evaluators.bias_auditor --model gpt-4o --attributes gender,race,age

# Generate HTML report
python -m src.reporting.report_generator --results results/latest.json
```

---

## 📊 Sample Evaluation Report

| Model | Faithfulness | Answer Relevancy | Injection Resistance | Bias Score | Pass |
|---|---|---|---|---|---|
| GPT-4o | 0.94 | 0.91 | 96% | 0.88 | ✅ |
| GPT-3.5-turbo | 0.81 | 0.87 | 89% | 0.82 | ⚠️ |
| Llama-3-8B | 0.76 | 0.83 | 78% | 0.79 | ❌ |

---

## 🔗 CI/CD Integration

```yaml
# .github/workflows/llm_eval.yml
- name: Run LLM Evaluation Suite
  run: pytest tests/eval/ --eval-threshold 0.85
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

---

## 👤 Author

**Srivardhan Bathula** — AI/ML Engineer
📧 Srivardhan.Bathula1@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/srivardhan-bathula) | [GitHub](https://github.com/SrivardhanBathula)
