import os
import sys
import json
import pandas as pd
from tabulate import tabulate

# Add src/ to sys.path so modules resolve whether executed from root or src/
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from agent import CustomerServiceAgent, ACTIVE_AGENT_VERSION
from metrics_config import all_metrics
from vertexai.evaluation import EvalTask

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "eval_dataset.jsonl")

print(f"🚀 [1/3] Loading Golden Evaluation Dataset from {DATASET_PATH}...")
eval_df = pd.read_json(DATASET_PATH, lines=True)

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

print("⚖️  [3/3] Running Vertex AI EvalTask with LLM-as-a-Judge...")
eval_task = EvalTask(
    dataset=eval_df,
    metrics=all_metrics,
    experiment=f"customer-service-agent-eval-{agent_ver}"
)


eval_result = eval_task.evaluate()

print("\n" + "="*80)
print("📊 EVALUATION SUMMARY METRICS")
print("="*80)
summary_data = [[k, f"{v:.4f}" if isinstance(v, float) else v] for k, v in eval_result.summary_metrics.items()]
print(tabulate(summary_data, headers=["Metric Name", "Mean Score"], tablefmt="fancy_grid"))

print("\n" + "="*80)
print("🔍 INVOCATION-LEVEL DETAILS WITH LLM JUDGE EXPLANATIONS")
print("="*80)
detail_cols = [
    "eval_id",
    "trajectory_in_order_match/score",
    "groundedness/score",
    "question_answering_quality/score",
    "refund_policy_compliance/score",
    "refund_policy_compliance/explanation"
]
available_cols = [c for c in detail_cols if c in eval_result.metrics_table.columns]
print(tabulate(eval_result.metrics_table[available_cols], headers="keys", tablefmt="grid"))
