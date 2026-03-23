"""
KV Cache Ablation — Generation Phase (v2)
==========================================
For each prompt: baseline + (N_UNMASKED × K_SCALES) masked generations.
Tests 4 unmasked token counts (last 1, 2, 3, 4 tokens of input).
Single forward pass per prompt; batched generation per n_unmasked config.
Outputs to eval_results/n{N}/eval_generations.jsonl (crash-resumable).

Usage:
    python eval/eval_harness.py              # full 300-prompt run
    python eval/eval_harness.py --limit 10   # test with first 10 prompts
"""

import argparse
import json
import os
import sys
import time

# Allow importing engine from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from engine import KVEngine
from prompts import ALL_PROMPTS, get_by_category

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 200

K_SCALES = [-1.0, -0.5, -0.25, -0.05, 0, 0.05, 0.25, 0.5, 0.75, 1.0, 1.50]
N_UNMASKED_VALUES = [4, 5, 6]

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(ROOT_DIR, "llama_eval_results")


def gen_file(n_unmasked):
    """Path to the generations file for a given n_unmasked config."""
    d = os.path.join(OUTPUT_DIR, f"n{n_unmasked}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "eval_generations.jsonl")


def count_lines(filepath):
    if not os.path.exists(filepath):
        return 0
    with open(filepath) as f:
        return sum(1 for line in f if line.strip())


def append_jsonl(filepath, record):
    with open(filepath, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def truncate_file(filepath, n_lines):
    """Keep only the first n_lines of a JSONL file."""
    lines = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            lines.append(line)
    with open(filepath, "w") as f:
        f.writelines(lines)


def run_single_prompt(engine, prompt):
    """Run baseline + all (n_unmasked × k_scale) masked generations for one prompt."""
    text = engine.tokenizer.apply_chat_template(
        prompt["messages"], tokenize=False, add_generation_prompt=True
    )
    input_ids = torch.tensor(
        [engine.tokenizer.encode(text, add_special_tokens=False)], device=engine.device
    )
    seq_len = input_ids.shape[1]

    # Single forward pass → full KV cache (reused for all configs)
    cache = engine.forward(input_ids)

    # Baseline: no masking (shared across all n_unmasked values)
    baseline_kv = KVEngine.clone_kv(cache)
    baseline_output = engine.generate_from_cache(
        input_ids, baseline_kv, MAX_NEW_TOKENS
    )

    # For each n_unmasked, batch all 12 k-values
    configs = {}
    for n in N_UNMASKED_VALUES:
        unmasked = set(range(seq_len - n, seq_len))
        positions_to_mask = [p for p in range(seq_len) if p not in unmasked]

        kv_lists = [
            KVEngine.mask_kv_cache(cache, positions_to_mask, k_scale=k)
            for k in K_SCALES
        ]

        batch_outputs = engine.generate_batched(input_ids, kv_lists, MAX_NEW_TOKENS)
        masked_outputs = {str(k): out for k, out in zip(K_SCALES, batch_outputs)}

        configs[n] = {
            "unmasked_positions": sorted(unmasked),
            "masked_outputs": masked_outputs,
        }

        del kv_lists

    del cache
    torch.cuda.empty_cache()

    return {
        "baseline_output": baseline_output,
        "seq_len": seq_len,
        "configs": configs,
    }


def main():
    parser = argparse.ArgumentParser(description="KV cache ablation generation phase v2")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N prompts (0 = all)")
    args = parser.parse_args()

    prompts = ALL_PROMPTS
    #prompts = get_by_category("multi_step")

    if args.limit > 0:
        prompts = prompts[:args.limit]
    
    # Resume: use minimum line count across all output files to stay in sync
    completed_counts = {n: count_lines(gen_file(n)) for n in N_UNMASKED_VALUES}
    completed = min(completed_counts.values())

    # Truncate any file that got ahead (e.g., crash mid-write across files)
    for n, count in completed_counts.items():
        if count > completed:
            print(f"  Truncating n{n} from {count} to {completed} lines (sync)")
            truncate_file(gen_file(n), completed)

    remaining = prompts[completed:]
    total = len(prompts)

    if not remaining:
        print("All prompts already processed.")
        return

    if completed > 0:
        print(f"Resuming: {completed} prompts already done, skipping.")

    print(f"Model: {MODEL_ID}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"K-scales: {K_SCALES}")
    print(f"N-unmasked: {N_UNMASKED_VALUES}")
    print(f"Prompts: {total} ({len(remaining)} remaining)")
    for n in N_UNMASKED_VALUES:
        print(f"  Output: {gen_file(n)}")
    print()

    engine = KVEngine()
    engine.load_model(MODEL_ID)

    t0 = time.time()

    for i, prompt in enumerate(remaining):
        idx = completed + i + 1

        result = run_single_prompt(engine, prompt)

        # Write one line to each n_unmasked file
        for n in N_UNMASKED_VALUES:
            record = {
                "id": prompt["id"],
                "category": prompt["category"],
                "messages": prompt["messages"],
                "reference_answer": prompt["reference_answer"],
                "seq_len": result["seq_len"],
                "n_unmasked": n,
                "unmasked_positions": result["configs"][n]["unmasked_positions"],
                "baseline_output": result["baseline_output"],
                "masked_outputs": result["configs"][n]["masked_outputs"],
                "timestamp": time.time(),
            }
            append_jsonl(gen_file(n), record)

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(remaining) - i - 1) / rate if rate > 0 else 0

        cat_short = prompt["category"][:8]
        print(
            f"[{idx:3d}/{total}] {cat_short:<8s} "
            f"seq={result['seq_len']:3d} "
            f"({elapsed:.0f}s, eta {eta:.0f}s) "
            f"{prompt['id']}"
        )

    total_time = time.time() - t0
    print(f"\nDone. {len(remaining)} prompts in {total_time:.0f}s.")
    for n in N_UNMASKED_VALUES:
        print(f"  {gen_file(n)}")


if __name__ == "__main__":
    main()
