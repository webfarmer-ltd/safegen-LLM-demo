"""Optional SafeGen-style VAL reward backend for *real* PDDL3 data.

Use only after creating a real PDDL3 dataset with domain_file/problem_file
columns. This backend is intentionally conservative: unknown VAL output is
assigned format_error (-1) rather than being silently counted as a success.

The paper measures goal-predicate fraction and first failure step. VAL's
verbose output format varies by build. Provide reliable custom extraction
before claiming exact replication; this implementation falls back to zero
progress for cases it cannot parse. The toy evaluator is fully deterministic.
"""
from __future__ import annotations
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from safegen_reward import completion_to_text
from safegen_blocks import parse_plan


FAIL_STEP = re.compile(r"(?:step|action|happening)\s*(?:#|number|at)?\s*(\d+)", re.I)


def val_grade(plan: str, domain: str, problem: str, reference_length: int,
              validator: str = "Validate") -> tuple[str, float]:
    if parse_plan(plan) is None:
        # The parser in safegen_blocks is restricted to Blocksworld operator names.
        # For other domains, replace this check with the domain's operator grammar.
        return ("format_error", -1.0)
    executable = shutil.which(validator)
    if executable is None:
        raise FileNotFoundError(f"VAL validator '{validator}' not found in PATH")
    if not Path(domain).is_file() or not Path(problem).is_file():
        raise FileNotFoundError(f"Missing domain/problem file: {domain}, {problem}")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".plan", encoding="utf8", delete=False) as temp:
        temp.write(plan + "\n")
        path = temp.name
    try:
        proc = subprocess.run([executable, "-v", domain, problem, path],
                              capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return ("format_error", -1.0)
    finally:
        os.unlink(path)

    output = (proc.stdout + "\n" + proc.stderr).lower()
    if "plan valid" in output and "plan invalid" not in output:
        return ("success", 1.0)
    if any(s in output for s in ("bad operator in plan", "no matching action defined",
                                "type problem with proposition", "syntax error", "parse error")):
        return ("format_error", -1.0)
    match = FAIL_STEP.search(output)
    progress = min(1.0, int(match.group(1)) / max(1, reference_length)) if match else 0.0
    if "unsatisfied precondition" in output:
        return ("precondition_violation", -0.6 + 0.3 * progress)
    if any(s in output for s in ("constraint violation", "safety constraint", "plan failed to execute")):
        return ("safety_violation", -0.9 + 0.3 * progress)
    if "goal not satisfied" in output:
        # Cannot infer n_sat/n_total reliably from generic VAL text. This
        # conservative lower-bound is not a full implementation of paper Eq.(9).
        return ("goal_not_satisfied", -0.4)
    return ("format_error", -1.0)


def safegen_val_reward(completions, domain_file, problem_file, reference_length,
                       validator="Validate", **kwargs):
    return [val_grade(completion_to_text(c), d, p, int(ref), validator)[1]
            for c, d, p, ref in zip(completions, domain_file, problem_file, reference_length)]
