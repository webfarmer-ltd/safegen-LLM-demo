"""Greedy plan-generation evaluation of an SFT or GRPO LoRA adapter."""
import argparse
import json
import torch
from peft import PeftModel
from datasets import load_dataset
from safegen_blocks import grade_plan
from train_sft_safegen import MODEL_ID, load_base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--adapter", required=True)
    p.add_argument("--test-jsonl", default="/workspace/data/safegen_test_grpo.jsonl")
    p.add_argument("--max-new-tokens", type=int, default=128)
    opts = p.parse_args()
    model, processor, _ = load_base(opts.model)
    model = PeftModel.from_pretrained(model, opts.adapter, is_trainable=False)
    model.eval()
    data = load_dataset("json", data_files={"test": opts.test_jsonl})["test"]
    counts: dict[str, int] = {}
    for i, sample in enumerate(data):
        prompt = processor.apply_chat_template(
            sample["prompt"], tokenize=False, add_generation_prompt=True
        )
        inputs = processor(prompt, return_tensors="pt")
        inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=opts.max_new_tokens, do_sample=False,
                pad_token_id=processor.pad_token_id,
            )
        continuation = outputs[0, inputs["input_ids"].shape[-1]:]
        text = processor.decode(continuation, skip_special_tokens=True)
        grade = grade_plan(text, json.loads(sample["problem_json"]))
        counts[grade.category] = counts.get(grade.category, 0) + 1
        print(f"[{i+1}/{len(data)}] {grade.category} reward={grade.reward:.3f}\n{text}\n")
    print("Category counts:", counts)
    print("Success rate:", counts.get("success", 0) / len(data))


if __name__ == "__main__":
    main()
