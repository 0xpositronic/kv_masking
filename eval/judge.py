"""
KV Cache Ablation — Gemini Judge Phase (Batched)
=================================================
Reads eval_generations.jsonl from all n_unmasked directories.
Batches multiple prompts × all n_unmasked configs into single API calls.
Each call evaluates B prompts × (1 baseline + 4×12 masked) = B×49 outputs.

With batch_size=10: 300 prompts → 30 API calls (vs 1200 unbatched).

Usage:
    python eval/judge.py                      # judge all, 10 prompts per batch
    python eval/judge.py --batch-size 5       # 5 prompts per batch (safer)
    python eval/judge.py --fresh              # delete existing and re-judge
    python eval/judge.py --dry-run            # show first batch without calling API
"""

import argparse
import json
import os
import time
import traceback

from dotenv import load_dotenv
from google import genai
from google.genai import types

K_SCALES = [-1.0, -0.5, -0.25, -0.05, 0, 0.05, 0.25, 0.5, 0.75, 1.0, 1.50]
N_UNMASKED_VALUES = [4, 5, 6]
CONFIG_LABELS = ["baseline"] + [f"k={k}" for k in K_SCALES]
VERDICTS = ["CORRECT", "PARTIALLY_CORRECT", "COHERENT_UNRELATED", "NONSENSE"]

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(ROOT_DIR, "llama_eval_results")
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"

JUDGE_SYSTEM = """\
You are evaluating outputs from a language model experiment where we mask parts \
of the model's internal state (KV cache) and observe whether it can still answer correctly.

Classify each output into EXACTLY ONE category:
- CORRECT: Correctly answers the question (matches reference in substance)
- PARTIALLY_CORRECT: Some relevant info or correct answer format but incomplete or mixed with errors
- COHERENT_UNRELATED: Coherent text but doesn't answer the question
- NONSENSE: Garbage — repeated tokens, random symbols, fragments, incoherent

Notes:
- For counterfactual prompts: judge against the REFERENCE ANSWER (the false premise), not real-world truth.
- For instruction_following: if answer is correct but format is wrong, mark PARTIALLY_CORRECT.
- Be lenient on wording — focus on whether the core answer is present."""


def read_jsonl(filepath):
    records = []
    if not os.path.exists(filepath):
        return records
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def is_valid_judgment(rec):
    """Check if a judgment record is valid (not an error)."""
    if rec.get("error"):
        return False
    for v in rec.get("verdicts", {}).values():
        if isinstance(v, dict) and v.get("verdict") == "ERROR":
            return False
    return True


def append_jsonl(filepath, record):
    with open(filepath, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def rate_limit(last_call_time, rpm):
    """Sleep if needed to stay under rpm limit. Returns new last_call_time."""
    if last_call_time is None:
        return time.time()
    min_interval = 60.0 / rpm
    elapsed = time.time() - last_call_time
    if elapsed < min_interval:
        wait = min_interval - elapsed
        print(f"  Rate limit: waiting {wait:.0f}s", end="\r")
        time.sleep(wait)
    return time.time()


def gen_file(n):
    return os.path.join(OUTPUT_DIR, f"n{n}", "eval_generations.jsonl")


def judge_file(n):
    return os.path.join(OUTPUT_DIR, f"n{n}", "eval_judgments.jsonl")


def load_all_generations():
    """Load generations from all n_unmasked dirs, grouped by prompt ID."""
    prompts = {}  # id -> {n -> generation_record}
    order = []    # preserve insertion order of IDs

    for n in N_UNMASKED_VALUES:
        records = read_jsonl(gen_file(n))
        for rec in records:
            pid = rec["id"]
            if pid not in prompts:
                prompts[pid] = {"id": pid, "category": rec["category"],
                                "messages": rec["messages"],
                                "reference_answer": rec["reference_answer"],
                                "seq_len": rec["seq_len"],
                                "baseline_output": rec["baseline_output"]}
                order.append(pid)
            prompts[pid][f"n{n}"] = {
                "unmasked_positions": rec["unmasked_positions"],
                "masked_outputs": rec["masked_outputs"],
            }

    return [prompts[pid] for pid in order]


def build_batch_prompt(batch):
    """Build the judge prompt for a batch of prompts."""
    parts = []

    for i, prompt_data in enumerate(batch, 1):
        question = ""
        for msg in reversed(prompt_data["messages"]):
            if msg["role"] == "user":
                question = msg["content"]
                break

        parts.append(f"=== PROMPT {i}: {prompt_data['id']} ===")
        parts.append(f"QUESTION: {question}")
        parts.append(f"CATEGORY: {prompt_data['category']}")
        parts.append(f"REFERENCE: {prompt_data['reference_answer']}")
        parts.append(f"\nBASELINE: {prompt_data['baseline_output']}")

        for n in N_UNMASKED_VALUES:
            n_data = prompt_data.get(f"n{n}")
            if not n_data:
                continue
            parts.append(f"\n[Last {n} token{'s' if n > 1 else ''} unmasked]")
            for k in K_SCALES:
                output = n_data["masked_outputs"].get(str(k), "[MISSING]")
                parts.append(f"n{n}_k={k}: {output}")

        parts.append("")  # blank line between prompts

    # Schema example
    example_keys = {}
    pid = batch[0]["id"]
    example_keys[pid] = {"baseline": "VERDICT"}
    for n in N_UNMASKED_VALUES:
        for k in K_SCALES:
            example_keys[pid][f"n{n}_k={k}"] = "VERDICT"

    parts.append("Return a JSON object. For EACH prompt ID, provide verdicts for baseline and every n*_k=* output.")
    parts.append("Use ONLY these verdict values: CORRECT, PARTIALLY_CORRECT, COHERENT_UNRELATED, NONSENSE")
    parts.append(f"Prompt IDs in this batch: {[p['id'] for p in batch]}")
    parts.append('Format: {"prompt_id": {"baseline": "VERDICT", "n1_k=-1.0": "VERDICT", ...}, ...}')
    parts.append("Return ONLY the JSON object.")

    return "\n".join(parts)


def parse_batch_response(response_text, batch):
    """Parse Gemini response into per-prompt, per-n_unmasked judgment records."""
    text = response_text.strip()

    # Strip markdown wrapper
    if text.startswith("```"):
        lines = text.split("\n")
        end_idx = len(lines) - 1
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break
        text = "\n".join(lines[1:end_idx]).strip()

    data = json.loads(text)

    all_judgments = []  # list of (n_unmasked, judgment_record)

    for prompt_data in batch:
        pid = prompt_data["id"]
        prompt_verdicts = data.get(pid)
        if prompt_verdicts is None:
            raise KeyError(f"Missing prompt ID in response: {pid}")

        # Extract baseline verdict
        baseline_raw = prompt_verdicts.get("baseline", "UNKNOWN")
        if isinstance(baseline_raw, dict):
            baseline_verdict = baseline_raw.get("verdict", "UNKNOWN").upper()
        else:
            baseline_verdict = str(baseline_raw).upper()
        if baseline_verdict not in VERDICTS:
            baseline_verdict = "UNKNOWN"

        baseline_failed = baseline_verdict != "CORRECT"

        # For each n_unmasked, build a judgment record
        for n in N_UNMASKED_VALUES:
            verdicts = {
                "baseline": {"verdict": baseline_verdict, "reasoning": ""},
            }

            any_masked_correct = False
            for k in K_SCALES:
                key = f"n{n}_k={k}"
                raw = prompt_verdicts.get(key, "UNKNOWN")
                if isinstance(raw, dict):
                    v = raw.get("verdict", "UNKNOWN").upper()
                    r = raw.get("reasoning", "")
                else:
                    v = str(raw).upper()
                    r = ""
                if v not in VERDICTS:
                    v = "UNKNOWN"

                verdicts[f"k={k}"] = {"verdict": v, "reasoning": r}
                if v == "CORRECT":
                    any_masked_correct = True

            judgment = {
                "id": pid,
                "category": prompt_data["category"],
                "verdicts": verdicts,
                "baseline_failed": baseline_failed,
                "baseline_refuses_masked_correct": baseline_failed and any_masked_correct,
                "timestamp": time.time(),
            }
            all_judgments.append((n, judgment))

    return all_judgments


def build_failure_judgments(batch, error_msg):
    """Build failure records for all prompts in a batch."""
    all_judgments = []
    for prompt_data in batch:
        for n in N_UNMASKED_VALUES:
            verdicts = {"baseline": {"verdict": "ERROR", "reasoning": error_msg}}
            for k in K_SCALES:
                verdicts[f"k={k}"] = {"verdict": "ERROR", "reasoning": error_msg}
            judgment = {
                "id": prompt_data["id"],
                "category": prompt_data["category"],
                "verdicts": verdicts,
                "baseline_failed": True,
                "baseline_refuses_masked_correct": False,
                "error": error_msg,
                "timestamp": time.time(),
            }
            all_judgments.append((n, judgment))
    return all_judgments


def retry_errors(all_prompts, args, api_key):
    """Find prompts with ERROR verdicts, re-judge them, replace in-place."""
    # Find error prompt IDs from the first n_unmasked file (same across all)
    n0 = N_UNMASKED_VALUES[0]
    existing = read_jsonl(judge_file(n0))
    error_ids = set()
    for j in existing:
        if j.get("error") or any(
            v.get("verdict") == "ERROR"
            for v in j.get("verdicts", {}).values()
        ):
            error_ids.add(j["id"])

    if not error_ids:
        print("No ERROR records found.")
        return

    # Filter to prompts that need retrying
    prompts_by_id = {p["id"]: p for p in all_prompts}
    retry_prompts = [prompts_by_id[pid] for pid in error_ids if pid in prompts_by_id]
    retry_prompts.sort(key=lambda p: p["id"])

    print(f"Found {len(error_ids)} prompts with ERROR verdicts:")
    for pid in sorted(error_ids):
        print(f"  {pid}")

    # Batch them
    batches = []
    for i in range(0, len(retry_prompts), args.batch_size):
        batches.append(retry_prompts[i:i + args.batch_size])

    print(f"\nRe-judging with batch_size={args.batch_size} ({len(batches)} API calls)")

    if args.dry_run:
        prompt_text = build_batch_prompt(batches[0])
        print(f"\n=== DRY RUN: First retry batch ===")
        print(f"Prompts: {[p['id'] for p in batches[0]]}")
        print(f"Prompt length: {len(prompt_text)} chars")
        print(f"\n{prompt_text[:3000]}...")
        return

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=32768,
        thinking_config=types.ThinkingConfig(thinking_budget=2048),
    )

    # Collect new judgments keyed by (n_unmasked, prompt_id)
    new_judgments = {}  # (n, pid) -> judgment
    t0 = time.time()
    last_call = None

    for batch_idx, batch in enumerate(batches):
        batch_num = batch_idx + 1
        prompt_ids = [p["id"] for p in batch]
        prompt_text = build_batch_prompt(batch)

        success = False
        for attempt in range(args.max_retries):
            last_call = rate_limit(last_call, args.rpm)
            try:
                response = client.models.generate_content(
                    model=args.model,
                    contents=prompt_text,
                    config=config,
                )
                judgments = parse_batch_response(response.text, batch)
                success = True
                break
            except Exception as e:
                wait = max(2 ** (attempt + 1), 60.0 / args.rpm)
                print(f"  Retry {attempt+1}/{args.max_retries} batch {batch_num}: {e}")
                if attempt < args.max_retries - 1:
                    time.sleep(wait)
                    last_call = time.time()
                else:
                    print(f"  FAILED batch {batch_num} again after {args.max_retries} retries")
                    traceback.print_exc()
                    judgments = build_failure_judgments(batch, str(e))

        for n_unmasked, judgment in judgments:
            new_judgments[(n_unmasked, judgment["id"])] = judgment

        elapsed = time.time() - t0
        status = "OK" if success else "FAILED"
        print(f"[retry {batch_num}/{len(batches)}] {status} {prompt_ids[0]}..{prompt_ids[-1]} ({elapsed:.0f}s)")

    # Replace ERROR records in judgment files
    replaced = 0
    for n in N_UNMASKED_VALUES:
        jf = judge_file(n)
        records = read_jsonl(jf)
        updated = []
        for rec in records:
            replacement = new_judgments.get((n, rec["id"]))
            if replacement is not None:
                updated.append(replacement)
                replaced += 1
            else:
                updated.append(rec)
        with open(jf, "w") as f:
            for rec in updated:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    still_errors = sum(1 for (n, pid), j in new_judgments.items()
                       if j.get("error") and n == N_UNMASKED_VALUES[0])
    print(f"\nReplaced {replaced} records across {len(N_UNMASKED_VALUES)} files.")
    if still_errors:
        print(f"  {still_errors} prompts still have errors — run --retry-errors again with smaller --batch-size")
    else:
        print("  All errors resolved.")


def main():
    parser = argparse.ArgumentParser(description="KV cache ablation Gemini judge (batched)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Prompts per API call (default: 1)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing judgments and re-judge all")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Re-judge only prompts with ERROR verdicts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print first batch prompt without calling API")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--rpm", type=int, default=5,
                        help="Max requests per minute (default: 5, free tier limit)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Gemini model to use (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    # Load API key
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
    load_dotenv(env_path)
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment or .env file")
        return

    # Load all generations grouped by prompt
    all_prompts = load_all_generations()
    if not all_prompts:
        print("No generations found. Run eval_harness.py first.")
        return

    # Retry-errors mode: find ERROR prompts, re-judge, replace in-place
    if args.retry_errors:
        retry_errors(all_prompts, args, api_key)
        return

    # Fresh: delete existing judgments
    if args.fresh:
        for n in N_UNMASKED_VALUES:
            jf = judge_file(n)
            if os.path.exists(jf):
                os.remove(jf)
        print("Deleted existing judgments.")

    # Resume: find prompts with valid (non-error) judgments across all n files
    judged_ids = None  # intersection of valid IDs across all n files
    for n in N_UNMASKED_VALUES:
        records = read_jsonl(judge_file(n))
        valid_ids = {rec["id"] for rec in records if is_valid_judgment(rec)}
        judged_ids = valid_ids if judged_ids is None else judged_ids & valid_ids
    if judged_ids is None:
        judged_ids = set()

    # Strip error entries from judgment files, keep only valid ones
    for n in N_UNMASKED_VALUES:
        jf = judge_file(n)
        records = read_jsonl(jf)
        valid = [r for r in records if r["id"] in judged_ids]
        if len(valid) < len(records):
            stripped = len(records) - len(valid)
            with open(jf, "w") as f:
                for r in valid:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  Stripped {stripped} error entries from {jf}")

    remaining = [p for p in all_prompts if p["id"] not in judged_ids]
    total = len(all_prompts)
    completed = total - len(remaining)

    if not remaining:
        print("All prompts already judged.")
        return

    if completed > 0:
        print(f"Resuming: {completed} prompts already done.")

    # Batch the remaining prompts
    batches = []
    for i in range(0, len(remaining), args.batch_size):
        batches.append(remaining[i:i + args.batch_size])

    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total prompts: {total} ({len(remaining)} remaining)")
    print(f"API calls needed: {len(batches)}")
    for n in N_UNMASKED_VALUES:
        print(f"  Output: {judge_file(n)}")
    print()

    if args.dry_run:
        prompt_text = build_batch_prompt(batches[0])
        print("=== DRY RUN: First batch ===")
        print(f"Prompts in batch: {[p['id'] for p in batches[0]]}")
        print(f"Prompt length: {len(prompt_text)} chars")
        print(f"\n{prompt_text[:3000]}...")
        return

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=32768,
        thinking_config=types.ThinkingConfig(thinking_budget=2048),
    )

    t0 = time.time()
    last_call = None

    for batch_idx, batch in enumerate(batches):
        batch_num = batch_idx + 1
        prompt_ids = [p["id"] for p in batch]

        prompt_text = build_batch_prompt(batch)

        success = False
        for attempt in range(args.max_retries):
            last_call = rate_limit(last_call, args.rpm)
            try:
                response = client.models.generate_content(
                    model=args.model,
                    contents=prompt_text,
                    config=config,
                )
                judgments = parse_batch_response(response.text, batch)
                success = True
                break

            except Exception as e:
                wait = max(2 ** (attempt + 1), 60.0 / args.rpm)
                print(f"  Retry {attempt+1}/{args.max_retries} batch {batch_num}: {e}")
                if attempt < args.max_retries - 1:
                    time.sleep(wait)
                    last_call = time.time()
                else:
                    print(f"  FAILED batch {batch_num} after {args.max_retries} retries")
                    traceback.print_exc()
                    judgments = build_failure_judgments(batch, str(e))

        # Write judgments to appropriate files
        for n_unmasked, judgment in judgments:
            append_jsonl(judge_file(n_unmasked), judgment)

        elapsed = time.time() - t0
        prompts_done = completed + (batch_idx + 1) * args.batch_size
        prompts_done = min(prompts_done, total)
        rate = prompts_done / elapsed if elapsed > 0 else 1
        eta = (total - prompts_done) / rate if rate > 0 else 0

        # Show summary for this batch
        if success:
            # Count verdicts for k=0 in n3 (most comparable to previous runs)
            n3_verdicts = [v for n, j in judgments if n == 3
                          for k, v in [(j["verdicts"].get("k=0", {}).get("verdict", "?"), None)]
                          ]
            status = "OK"
        else:
            status = "FAILED"

        print(
            f"[batch {batch_num:3d}/{len(batches)}] "
            f"{status} {prompt_ids[0]}..{prompt_ids[-1]} "
            f"({elapsed:.0f}s, eta {eta:.0f}s)"
        )

    total_time = time.time() - t0
    print(f"\nDone. {len(remaining)} prompts in {total_time:.0f}s ({len(batches)} API calls).")
    for n in N_UNMASKED_VALUES:
        print(f"  {judge_file(n)}")


if __name__ == "__main__":
    main()
