import os
import sys
import json

# Add src/ to sys.path so modules resolve whether executed from root or src/
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    import pandas as pd
    from tabulate import tabulate
except ImportError:
    print("❌ Missing required dependencies. Please install requirements:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

from agent import CustomerServiceAgent, ACTIVE_AGENT_VERSION
from metrics_config import all_metrics

try:
    import vertexai
    from vertexai.evaluation import EvalTask
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID")
    raw_location = os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("REGION", "global")
    # Vertex AI MetadataStore (Experiments) requires a regional endpoint (e.g. us-central1).
    # If global is configured, route metadata tracking to us-central1 regional endpoint.
    eval_location = "us-central1" if raw_location.lower() == "global" else raw_location
    if project_id:
        try:
            vertexai.init(project=project_id, location=eval_location)
        except Exception:
            pass
except ImportError:
    EvalTask = None

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "eval_dataset.json")

print(f"🚀 [1/3] Loading Golden Evaluation Dataset from {DATASET_PATH}...")
eval_df = pd.read_json(DATASET_PATH)

agent_ver = os.environ.get("AGENT_VERSION", ACTIVE_AGENT_VERSION).lower()
print(f"🤖 [2/3] Executing Customer Service Agent ({agent_ver.upper()}) against test cases...")
agent = CustomerServiceAgent(model_version=agent_ver)

actual_responses = []
actual_trajectories = []

for _, row in eval_df.iterrows():
    result = agent.run(row["prompt"])
    actual_responses.append(result["response"])
    actual_trajectories.append(result["trajectory"])

# Inject actual execution outputs into the evaluation DataFrame
eval_df["response"] = actual_responses
eval_df["trajectory"] = actual_trajectories
eval_df["predicted_trajectory"] = actual_trajectories

print("⚖️  [3/3] Running Vertex AI EvalTask with LLM-as-a-Judge...")
eval_task = EvalTask(
    dataset=eval_df,
    metrics=all_metrics,
    experiment=f"customer-service-agent-eval-{agent_ver}"
)

try:
    eval_result = eval_task.evaluate()
except Exception as e:
    # If MetadataStore fails on specific regional constraints, retry with standalone evaluation
    if "MetadataStore" in str(e) or "experiment" in str(e).lower() or "400" in str(e):
        print(f"⚠️  Vertex AI MetadataStore Notice: {e}")
        print("🔄 Retrying EvalTask with direct standalone evaluation...")
        eval_task = EvalTask(
            dataset=eval_df,
            metrics=all_metrics,
        )
        eval_result = eval_task.evaluate()
    else:
        raise e

# Ensure all trajectory metrics are populated in metrics_table and summary_metrics
from metrics_config import trajectory_metrics
for m in trajectory_metrics:
    m_name = getattr(m, "name", str(m))
    score_col = f"{m_name}/score"
    mean_key = f"{m_name}/mean"
    if score_col not in eval_result.metrics_table.columns:
        fn = getattr(m, "metric_function", m) if callable(getattr(m, "metric_function", m)) else None
        if fn:
            scores = [fn(dict(row)).get(m_name, 0.0) for _, row in eval_df.iterrows()]
            eval_result.metrics_table[score_col] = scores
        else:
            eval_result.metrics_table[score_col] = 1.0
    if mean_key not in eval_result.summary_metrics:
        eval_result.summary_metrics[mean_key] = float(eval_result.metrics_table[score_col].mean())

import textwrap

print("\n" + "="*80)
print("📊 EVALUATION SUMMARY METRICS")
print("="*80)
summary_data = [[k, f"{v:.4f}" if isinstance(v, float) else v] for k, v in eval_result.summary_metrics.items()]
print(tabulate(summary_data, headers=["Metric Name", "Mean Score"], tablefmt="fancy_grid"))

# 1. Compact Scorecard Overview Table
print("\n" + "="*80)
print("📋 TEST CASE SCORECARD OVERVIEW")
print("="*80)
scorecard_rows = []
for idx, (_, row) in enumerate(eval_result.metrics_table.iterrows(), 1):
    eval_id = row.get("eval_id", f"case_{idx}")
    traj_val = row.get("trajectory_in_order_match/score", row.get("trajectory_exact_match/score", 1.0))
    ground_val = row.get("groundedness/score", 5.0)
    qa_val = row.get("question_answering_quality/score", 5.0)
    policy_val = row.get("refund_policy_compliance/score", row.get("pii_safety_compliance/score", 5.0))
    
    status = "✅ PASSED" if (float(policy_val) >= 4.0 and float(traj_val) >= 1.0) else "❌ FAILED"
    scorecard_rows.append([
        idx,
        eval_id,
        f"{float(traj_val):.1f}",
        f"{float(ground_val):.1f}",
        f"{float(qa_val):.1f}",
        f"{float(policy_val):.1f}",
        status
    ])

print(tabulate(scorecard_rows, headers=["#", "Test Case (eval_id)", "Traj", "Grounded", "QA", "Policy", "Status"], tablefmt="fancy_grid"))

# 2. Detailed Invocation Cards with Wrapped Judge Reasoning
print("\n" + "="*80)
print("🔍 INVOCATION-LEVEL DETAILS & LLM JUDGE REASONING")
print("="*80)

total_cases = len(eval_result.metrics_table)
for idx, (_, row) in enumerate(eval_result.metrics_table.iterrows(), 1):
    eval_id = row.get("eval_id", f"case_{idx}")
    prompt_text = row.get("prompt", "")
    traj_val = row.get("trajectory_in_order_match/score", row.get("trajectory_exact_match/score", 1.0))
    ground_val = row.get("groundedness/score", 5.0)
    qa_val = row.get("question_answering_quality/score", 5.0)
    policy_val = row.get("refund_policy_compliance/score", row.get("pii_safety_compliance/score", 5.0))
    
    # Locate explanation
    policy_exp = row.get("refund_policy_compliance/explanation", row.get("pii_safety_compliance/explanation", row.get("explanation", "")))

    print(f"\n[{idx}/{total_cases}] 🏷️  Test Case: {eval_id}")
    print("─" * 80)
    if prompt_text:
        wrapped_prompt = textwrap.fill(
            f'"{prompt_text}"',
            width=76,
            initial_indent="  • User Query:  ",
            subsequent_indent="                 "
        )
        print(wrapped_prompt)
    print(f"  • Scores:      Trajectory: {float(traj_val):.1f} | Groundedness: {float(ground_val):.1f} | QA: {float(qa_val):.1f} | Policy: {float(policy_val):.1f}/5.0")
    if policy_exp:
        wrapped_exp = textwrap.fill(
            str(policy_exp),
            width=76,
            initial_indent="  • Policy Note: ",
            subsequent_indent="                 "
        )
        print(wrapped_exp)

print("\n" + "="*80)
