"""Stage 1: QLoRA SFT on the synthetic SafeGen-style BlocksWorld data."""
import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "google/gemma-4-E2B-it"


def load_base(model_id: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for 4-bit QLoRA training")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_storage=dtype,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=dtype, quantization_config=quant,
        device_map={"": torch.cuda.current_device()},
    )
    model.config.use_cache = False
    # processor = AutoProcessor.from_pretrained(model_id)
    # return model, processor, dtype
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    return model, tokenizer, dtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--train-jsonl", default="/workspace/data/safegen_train_sft.jsonl")
    parser.add_argument("--output", default="/workspace/outputs/gemma4-e2b-safegen-sft")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    opts = parser.parse_args()

    model, tokenizer, dtype = load_base(opts.model)
    dataset = load_dataset("json", data_files={"train": opts.train_jsonl})["train"]

    # Gemma 4: omit target_modules so PEFT uses the language-model defaults
    # instead of accidentally inserting LoRA into the vision/audio towers.
    # We do not retrain/save large token embeddings or lm_head: no new tokens.
    lora = LoraConfig(
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM",
        target_modules=(
            r".*language_model.*\."
            r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        ),
    )
    
    config = SFTConfig(
        output_dir=opts.output,
        max_length=opts.max_length,
        num_train_epochs=opts.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        completion_only_loss=True,  # no loss on task description / constraints
        packing=False,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
    )
    trainer = SFTTrainer(
        model=model, args=config, train_dataset=dataset,
        peft_config=lora, processing_class=tokenizer,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(opts.output)
    tokenizer.save_pretrained(opts.output)
    print("Saved SFT LoRA adapter:", opts.output)


if __name__ == "__main__":
    main()
