import os
import sys
import json

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

from agent import CustomerServiceAgent
from metrics_config import pairwise_comparison_metric

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

print("="*80)
print("⚔️  STARTING PAIRWISE A/B AGENT EVALUATION (v1 Baseline vs. v2 Challenger)")
print("="*80)

print(f"📂 [1/3] Loading Golden Evaluation Dataset from {DATASET_PATH}...")
eval_df = pd.read_json(DATASET_PATH)

print("🤖 [2/3] Generating Responses for Candidate A (v2 Challenger) and Candidate B (v1 Baseline)...")
agent_v1 = CustomerServiceAgent(model_version="v1")
agent_v2 = CustomerServiceAgent(model_version="v2")

responses_v1 = []
responses_v2 = []

for _, row in eval_df.iterrows():
    responses_v1.append(agent_v1.run(row["prompt"])["response"])
    responses_v2.append(agent_v2.run(row["prompt"])["response"])

# Candidate A = v2 Challenger (New Agent)
# Candidate B / baseline_model_response = v1 Baseline (Production Agent)
pairwise_df = pd.DataFrame({
    "prompt": eval_df["prompt"],
    "response": responses_v2,
    "baseline_model_response": responses_v1,
    "context": eval_df["context"],
    "eval_id": eval_df["eval_id"]
})

print("⚖️  [3/3] Running Vertex AI Pairwise Comparative EvalTask with LLM Judge...")
eval_task = EvalTask(
    dataset=pairwise_df,
    metrics=[pairwise_comparison_metric],
    experiment="agent-v1-vs-v2-pairwise-benchmark"
)

try:
    eval_result = eval_task.evaluate()
except Exception as e:
    # If MetadataStore fails on specific regional constraints, retry with standalone evaluation
    if "MetadataStore" in str(e) or "experiment" in str(e).lower() or "400" in str(e):
        print(f"⚠️  Vertex AI MetadataStore Notice: {e}")
        print("🔄 Retrying EvalTask with direct standalone evaluation...")
        eval_task = EvalTask(
            dataset=pairwise_df,
            metrics=[pairwise_comparison_metric],
        )
        eval_result = eval_task.evaluate()
    else:
        raise e

print("\n" + "="*80)
print("🏆 PAIRWISE A/B EVALUATION SUMMARY")
print("="*80)
summary_data = [[k, f"{v:.2%}" if "rate" in k else v] for k, v in eval_result.summary_metrics.items()]
print(tabulate(summary_data, headers=["Pairwise Metric", "Distribution / Score"], tablefmt="fancy_grid"))

print("\n" + "="*80)
print("🔍 HEAD-TO-HEAD DECISION DETAILS & REASONING")
print("="*80)
pairwise_cols = [
    "eval_id",
    "agent_pairwise_comparison/choice",
    "agent_pairwise_comparison/explanation"
]
available_cols = [c for c in pairwise_cols if c in eval_result.metrics_table.columns]
print(tabulate(eval_result.metrics_table[available_cols], headers="keys", tablefmt="grid"))
