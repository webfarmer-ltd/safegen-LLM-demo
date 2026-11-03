import os
import sys
import json
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from safegen_blocks import (build_examples, grade_plan, solve_bfs, format_plan, create_jsonl)
from safegen_reward import safegen_reward, completion_to_text


SPEC = {
    "blocks": ["b0", "b1", "b2"],
    "fragile": "b0",
    "initial_stacks": [["b0"], ["b1"], ["b2"]],
    "goal_stacks": [["b0"], ["b2", "b1"]],
    "reference_length": 2,
}


class TestSafeGenReward(unittest.TestCase):
    def test_success(self):
        self.assertEqual(grade_plan("(pickup b1)\n(stack b1 b2)", SPEC).category, "success")
        self.assertEqual(grade_plan("(pickup b1)\n(stack b1 b2)", SPEC).reward, 1.0)

    def test_safety(self):
        x = grade_plan("(pickup b1)\n(stack b1 b0)", SPEC)
        self.assertEqual(x.category, "safety_violation")
        self.assertGreaterEqual(x.reward, -0.9)
        self.assertLessEqual(x.reward, -0.6)

    def test_precondition(self):
        self.assertEqual(grade_plan("(stack b1 b2)", SPEC).category, "precondition_violation")

    def test_format(self):
        for text in ("I would pick it up", "```\n(pickup b1)\n```", "(pickup b99)"):
            self.assertEqual(grade_plan(text, SPEC).category, "format_error")

    def test_goal_incomplete(self):
        self.assertEqual(grade_plan("(pickup b1)\n(putdown b1)", SPEC).category, "goal_not_satisfied")

    def test_reward_group(self):
        results = safegen_reward(
            completions=["(pickup b1)\n(stack b1 b2)", "(pickup b1)\n(stack b1 b0)"],
            problem_json=[json.dumps(SPEC)]*2,
        )
        self.assertEqual(results[0], 1.)
        self.assertLess(results[1], 0.)

    def test_generate_and_solve(self):
        for row in build_examples(15, seed=9):
            self.assertEqual(grade_plan(row["reference_plan"], row["spec"]).category, "success")
            self.assertEqual(len(solve_bfs(row["spec"])), row["spec"]["reference_length"])

    def test_conversational_completion(self):
        self.assertEqual(completion_to_text([{"role":"assistant", "content":"(pickup b1)"}]), "(pickup b1)")


if __name__ == "__main__":
    unittest.main()
