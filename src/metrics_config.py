import vertexai
from vertexai.evaluation import (
    MetricPromptTemplateExamples,
    PointwiseMetric,
    TrajectoryExactMatch,
    TrajectoryInOrderMatch,
    TrajectoryPrecision,
    TrajectoryRecall,
)

# 1. Deterministic Trajectory Metrics
trajectory_metrics = [
    TrajectoryExactMatch(),
    TrajectoryInOrderMatch(),
    TrajectoryPrecision(),
    TrajectoryRecall(),
]

# 2. Standard Vertex AI Model-Based LLM Judges
standard_llm_metrics = [
    MetricPromptTemplateExamples.Pointwise.GROUNDEDNESS,
    MetricPromptTemplateExamples.Pointwise.QUESTION_ANSWERING_QUALITY,
]

# 3. Custom Pointwise LLM-as-a-Judge: Refund Policy Compliance
refund_policy_rubric = {
    "1": "The response issued a refund for an order older than 30 days or without checking damage/order history.",
    "2": "The response rejected a valid damaged refund claim or failed to explain the policy.",
    "3": "The response followed refund policies but hallucinated amounts or failed to record the reason.",
    "4": "The response correctly followed refund rules with minor tone or formatting imperfections.",
    "5": "The response strictly enforced 30-day return windows, validated order status, and communicated clearly.",
}

custom_policy_metric = PointwiseMetric(
    metric="refund_policy_compliance",
    metric_prompt_template=MetricPromptTemplateExamples.get_prompt_template(
        "pointwise_reasoning"
    ),
    criteria={
        "refund_policy_compliance": "Evaluate whether the agent strictly enforces the 30-day return policy and requires order lookup before issuing refunds."
    },
    rating_rubric=refund_policy_rubric,
)

all_metrics = trajectory_metrics + standard_llm_metrics + [custom_policy_metric]
