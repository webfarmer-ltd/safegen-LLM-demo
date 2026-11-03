"""Stage 2: SafeGen-style GRPO, initialized from the SFT LoRA adapter.

Single GPU, no vLLM, no external PDDL validator in this first prototype.
The reward is a deterministic symbolic evaluator implemented in safegen_blocks.py.
"""
import argparse
import torch
from datasets import load_dataset
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer
from safegen_reward import safegen_reward
from train_sft_safegen import MODEL_ID, load_base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--sft-adapter", default="/workspace/outputs/gemma4-e2b-safegen-sft")
    p.add_argument("--train-jsonl", default="/workspace/data/safegen_train_grpo.jsonl")
    p.add_argument("--output", default="/workspace/outputs/gemma4-e2b-safegen-grpo")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--generations", type=int, default=4)
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--max-completion-length", type=int, default=128)
    p.add_argument("--beta", type=float, default=0.01)
    opts = p.parse_args()

    model, processor, dtype = load_base(opts.model)
    processor.padding_side = "left"
    # Crucial: GRPO resumes *trainable* SFT adapter; don't start another fresh LoRA.
    model = PeftModel.from_pretrained(model, opts.sft_adapter, is_trainable=True)
    dataset = load_dataset("json", data_files={"train": opts.train_jsonl})["train"]

    config = GRPOConfig(
        output_dir=opts.output,
        max_steps=opts.steps,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        # 1 prompt per microbatch * 4 accumulation = group of 4 rollouts.
        gradient_accumulation_steps=opts.generations,
        num_generations=opts.generations,
        # max_prompt_length=opts.max_prompt_length,
        max_completion_length=opts.max_completion_length,
        temperature=0.6,
        top_p=0.9,
        beta=opts.beta,
        scale_rewards=False,  # matches SafeGen Eq.(5) (mean-only advantage)
        loss_type="grpo",  # standard PPO-style per-sequence GRPO formulation
        use_vllm=False,
        remove_unused_columns=False,  # keep problem_json for reward callback
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_steps=min(25, opts.steps),
        logging_steps=1,
        report_to="none",
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=safegen_reward,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(opts.output)
    processor.save_pretrained(opts.output)
    print("Saved GRPO LoRA adapter:", opts.output)


if __name__ == "__main__":
    main()
