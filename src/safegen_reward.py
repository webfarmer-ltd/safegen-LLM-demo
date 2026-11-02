"""TRL-compatible GRPO reward function.

The `problem_json` column is forwarded by GRPOTrainer to the reward function,
repeated for each sampled completion. Works without model/transformers/GPU.
"""
from __future__ import annotations
import json
from safegen_blocks import grade_plan


def completion_to_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict) and item.get("role") == "assistant":
                content = item.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    parts.extend(str(x.get("text", "")) for x in content if isinstance(x, dict) and x.get("type") == "text")
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(completion)


def safegen_reward(completions, problem_json, **kwargs) -> list[float]:
    if len(completions) != len(problem_json):
        raise ValueError("TRL metadata mismatch: completions and problem_json have different lengths")
    grades = [grade_plan(completion_to_text(output), json.loads(spec))
              for output, spec in zip(completions, problem_json)]
    log_metric = kwargs.get("log_metric")
    if log_metric is not None and grades:
        log_metric("safegen/success_rate", sum(g.category == "success" for g in grades) / len(grades))
        log_metric("safegen/format_error_rate", sum(g.category == "format_error" for g in grades) / len(grades))
    return [float(g.reward) for g in grades]
