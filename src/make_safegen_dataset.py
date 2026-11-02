import argparse
from safegen_blocks import create_jsonl

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="/workspace/data")
    p.add_argument("--train-size", type=int, default=80)
    p.add_argument("--test-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    opts = p.parse_args()
    create_jsonl(opts.out_dir, opts.train_size, opts.test_size, opts.seed)
    print("Generated train/test SFT and GRPO JSONL in", opts.out_dir)
