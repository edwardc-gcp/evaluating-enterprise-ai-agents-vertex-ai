# Evaluating Enterprise AI Agents with Vertex AI and LLM-as-a-Judge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Google Cloud Vertex AI](https://img.shields.io/badge/Google%20Cloud-Vertex%20AI-4285F4.svg)](https://cloud.google.com/vertex-ai)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the complete, production-ready implementation and evaluation assets for the Google Developer Codelab: **[Evaluating Enterprise AI Agents with Vertex AI and LLM-as-a-Judge](https://codelabs.developers.google.com/)**.

---

## 📖 Overview

Enterprise AI Agents execute multi-turn, multi-step actions across backend databases and third-party APIs. Evaluating them requires more than simple string matching:
1. **Deterministic Trajectory Matching**: Validates tool-calling sequence, parameter accuracy, and business workflow integrity.
2. **Vertex AI LLM-as-a-Judge**: Evaluates semantic groundedness, tone, question-answering quality, and strict corporate compliance (e.g., 30-day return policy validation) with explainable reasoning (Chain-of-Thought).

```
+-------------------------------------------------------------------------------+
|                       Golden Evaluation Dataset (JSONL)                       |
|           Prompt + Expected Trajectory (Tools) + Ground Truth Context         |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                       Customer Service Agent Under Test                       |
|                 Generates: Actual Response + Actual Trajectory                |
+-------------------------------------------------------------------------------+
                                        |
                   +--------------------+--------------------+
                   |                                         |
                   v                                         v
+------------------------------------+   +--------------------------------------+
|   Deterministic Trajectory Engine  |   |    Vertex AI Gen AI Evaluation SDK   |
|  * Exact Match                     |   |  * Groundedness (LLM Judge)          |
|  * In-Order Match                  |   |  * QA Quality (LLM Judge)            |
|  * Precision & Recall              |   |  * Custom Policy Rubric (5-Point)    |
+------------------------------------+   +--------------------------------------+
                   |                                         |
                   +--------------------+--------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                  Automated Pytest CI/CD Quality Gate & Console                |
+-------------------------------------------------------------------------------+
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/edwardc-gcp/evaluating-enterprise-ai-agents-vertex-ai.git
cd evaluating-enterprise-ai-agents-vertex-ai
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Authenticate with Google Cloud & Vertex AI
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

---

## 📂 Repository Structure

```text
├── data/
│   └── eval_dataset.jsonl          # 6 golden test cases (prompts, tool trajectories, contexts)
├── src/
│   ├── __init__.py
│   ├── agent.py                    # Customer Service AI Agent mock under test
│   ├── metrics_config.py           # Deterministic trajectory & Pointwise LLM-as-a-Judge rubrics
│   └── run_evaluation.py           # End-to-end evaluation runner using Vertex AI EvalTask
├── tests/
│   ├── __init__.py
│   └── test_agent_vertex_eval.py   # Automated Pytest CI/CD quality gate assertions
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🧪 Running the Evaluation

Execute the evaluation task against the golden dataset:
```bash
python3 src/run_evaluation.py
```

### Sample Output
```text
🚀 [1/3] Loading Golden Evaluation Dataset from .../data/eval_dataset.jsonl...
🤖 [2/3] Executing Customer Service Agent against test cases...
⚖️  [3/3] Running Vertex AI EvalTask with LLM-as-a-Judge...

================================================================================
📊 EVALUATION SUMMARY METRICS
================================================================================
+----------------------------------+--------------+
| Metric Name                      |   Mean Score |
+==================================+==============+
| trajectory_exact_match/mean      |       1.0000 |
| trajectory_in_order_match/mean   |       1.0000 |
| trajectory_precision/mean        |       1.0000 |
| groundedness/mean                |       5.0000 |
| question_answering_quality/mean  |       5.0000 |
| refund_policy_compliance/mean    |       5.0000 |
+----------------------------------+--------------+
```

---

## 🛡️ Automated CI/CD Quality Gates

Run the test suite in your CI/CD pipeline (Cloud Build, GitHub Actions, GitLab CI):
```bash
pytest -v -s tests/test_agent_vertex_eval.py
```

---

## 📚 Related Codelabs & Resources

* **Part 1 Codelab**: [Evaluating Agents with Agent Development Kit (ADK)](https://codelabs.developers.google.com/adk-eval/instructions)
* **Vertex AI Evaluation Documentation**: [Vertex AI Gen AI Evaluation Service Overview](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview)

---

## 📄 License
Apache License 2.0. See [LICENSE](LICENSE) for details.
