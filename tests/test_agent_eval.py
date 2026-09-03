import os
import sys
import pytest
import pandas as pd

# Add src/ to sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from agent import CustomerServiceAgent, ACTIVE_AGENT_VERSION
from metrics_config import all_metrics
from vertexai.evaluation import EvalTask

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "eval_dataset.jsonl")

@pytest.mark.asyncio
async def test_agent_quality_and_trajectory_gates():
    """CI/CD Quality Gate: Validates agent trajectory compliance and groundedness."""
    # 1. Ingest evaluation dataset
    eval_df = pd.read_json(DATASET_PATH, lines=True)
    
    # 2. Run agent (uses ACTIVE_AGENT_VERSION or AGENT_VERSION override)
    agent_ver = os.environ.get("AGENT_VERSION", ACTIVE_AGENT_VERSION)
    agent = CustomerServiceAgent(model_version=agent_ver)
    eval_df["response"] = [agent.run(p)["response"] for p in eval_df["prompt"]]

    eval_df["trajectory"] = [agent.run(p)["trajectory"] for p in eval_df["prompt"]]
    
    # 3. Evaluate via Vertex AI EvalTask
    eval_task = EvalTask(
        dataset=eval_df,
        metrics=all_metrics,
        experiment="ci-cd-build-validation"
    )
    eval_result = eval_task.evaluate()
    
    # 4. Assert enterprise quality thresholds
    trajectory_score = eval_result.summary_metrics.get("trajectory_in_order_match/mean", 0.0)
    groundedness_score = eval_result.summary_metrics.get("groundedness/mean", 0.0)
    policy_score = eval_result.summary_metrics.get("refund_policy_compliance/mean", 0.0)
    
    print(f"\n[CI/CD Metrics] Trajectory: {trajectory_score:.2f}, Groundedness: {groundedness_score:.2f}, Policy: {policy_score:.2f}")
    
    assert trajectory_score >= 0.8, f"❌ Trajectory matching score too low: {trajectory_score}"
    assert groundedness_score >= 4.0, f"❌ Groundedness score too low: {groundedness_score}"
    assert policy_score >= 4.0, f"❌ Policy compliance score too low: {policy_score}"
    print("✅ All Enterprise AI Agent Quality Gates Passed!")
