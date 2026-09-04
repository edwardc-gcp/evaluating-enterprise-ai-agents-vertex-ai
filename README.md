# Advanced ADK Evaluation with LLM-as-a-Judge Method

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Google Cloud Gemini Enterprise](https://img.shields.io/badge/Google%20Cloud-Gemini%20Enterprise-4285F4.svg)](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/evaluation-overview)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the complete, production-ready implementation and evaluation assets for the Google Developer Codelab: **[Advanced ADK Evaluation with LLM-as-a-Judge Method](https://codelabs.developers.google.com/)**.

---

## 📖 Overview

Enterprise AI Agents built with the **Agent Development Kit (ADK)** execute multi-turn, multi-step actions across backend databases and third-party APIs. Evaluating them requires more than simple string matching:
1. **Deterministic Trajectory Matching (The Math)**: Validates tool-calling sequence, parameter accuracy, and business workflow integrity ($0 cost, <10ms).
2. **Gemini Enterprise Agent Platform (LLM-as-a-Judge - The Essay & The Law)**: Evaluates semantic groundedness, tone, question-answering quality, and strict corporate compliance (e.g., 30-day return policy validation) with explainable reasoning (Chain-of-Thought).

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
|   Deterministic Trajectory Engine  |   | Gemini Enterprise Agent Platform SDK |
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

### 2. Set Up Virtual Environment & Dependencies (with uv)
```bash
# Cloud Shell has uv pre-installed. (On local machines: curl -LsSf https://astral.sh/uv/install.sh | sh)
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Initialize Google Cloud & Vertex AI Environment
Run the automated setup script to detect/set your project, enable APIs, and generate `.env`:
```bash
./init.sh
```
*(Or run `source ./set_env.sh` anytime to load the environment variables into your current terminal).*


---

## 📂 Repository Structure

```text
├── data/
│   └── eval_dataset.json           # 6 golden test cases (prompts, tool trajectories, contexts)
├── src/
│   ├── __init__.py
│   ├── agent.py                    # Customer Service AI Agent supporting v1 (Baseline) & v2 (Challenger)
│   ├── metrics_config.py           # Deterministic trajectory, Pointwise, and Pairwise LLM-as-a-Judge rubrics
│   ├── run_evaluation.py           # Pointwise evaluation runner using EvalTask
│   └── run_pairwise_eval.py        # Pairwise A/B comparative evaluation runner (v1 vs. v2)
├── tests/
│   ├── __init__.py
│   └── test_agent_eval.py          # Automated Pytest CI/CD quality gate assertions
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🧪 Running the Evaluations

### 1. Pointwise Agent Evaluation
Execute the pointwise evaluation task against the golden dataset:
```bash
python3 src/run_evaluation.py
```

### 2. Pairwise A/B Comparative Evaluation (Model Upgrade Benchmarking)
Benchmark **Candidate A (Agent v2 Challenger)** against **Candidate B (Agent v1 Baseline)**:
```bash
python3 src/run_pairwise_eval.py
```

### Sample Output
```text
================================================================================
🏆 PAIRWISE A/B EVALUATION SUMMARY
================================================================================
+----------------------------------------------------+------------------------+
| Pairwise Metric                                    | Distribution / Score   |
+====================================================+========================+
| agent_pairwise_comparison/candidate_A_win_rate     | 83.33%                 |
| agent_pairwise_comparison/candidate_B_win_rate     | 0.00%                  |
| agent_pairwise_comparison/tie_rate                 | 16.67%                 |
+----------------------------------------------------+------------------------+
```

---

## 🛡️ Automated CI/CD Quality Gates

Run the test suite in your CI/CD pipeline (Cloud Build, GitHub Actions, GitLab CI):
```bash
pytest -v -s tests/test_agent_eval.py
```

---

## 📚 Related Codelabs & Resources

* **Part 1 Codelab**: [Evaluating Agents with Agent Development Kit (ADK)](https://codelabs.developers.google.com/adk-eval/instructions)
* **Gemini Enterprise Agent Platform**: [Evaluation Overview Documentation](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/evaluation-overview)

---

## 📄 License
Apache License 2.0. See [LICENSE](LICENSE) for details.
