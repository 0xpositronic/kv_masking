"""
KV Cache Ablation — Analysis Phase
====================================
Reads eval_judgments.jsonl, produces aggregate statistics and tables.

Usage:
    python analyze.py                  # print tables and save summary.json
    python analyze.py --verbose        # also print per-prompt details
"""

import argparse
import json
import os

K_SCALES = [-1.0, -0.5, -0.25, -0.05, 0, 0.05, 0.25, 0.5, 0.75, 1.0, 1.50]
CONFIG_LABELS = ["baseline"] + [f"k={k}" for k in K_SCALES]
VERDICTS = ["CORRECT", "PARTIALLY_CORRECT", "COHERENT_UNRELATED", "NONSENSE"]
CATEGORIES = [
    "factual_recall", "counterfactual", "multi_step",
    "long_context", "multi_turn", "instruction_following",
]

OUTPUT_DIR = "llama_eval_results/n4"
JUDGMENTS_FILE = os.path.join(OUTPUT_DIR, "eval_judgments.jsonl")
GENERATIONS_FILE = os.path.join(OUTPUT_DIR, "eval_generations.jsonl")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "summary.json")


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


def print_table(title, headers, rows, col_widths=None):
    """Print a formatted ASCII table."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = len(str(h))
            for row in rows:
                w = max(w, len(str(row[i])))
            col_widths.append(w + 2)

    print(f"\n{'='*sum(col_widths)}")
    print(title)
    print(f"{'='*sum(col_widths)}")

    header_str = "".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_str)
    print("-" * sum(col_widths))

    for row in rows:
        row_str = "".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row)))
        print(row_str)


def analyze(judgments, generations_lookup, verbose=False):
    """Run all analyses and return summary dict."""
    total = len(judgments)
    baseline_correct = [j for j in judgments if not j["baseline_failed"]]
    baseline_failed = [j for j in judgments if j["baseline_failed"]]
    bc_count = len(baseline_correct)

    print(f"Total prompts: {total}")
    print(f"Baseline correct: {bc_count}")
    print(f"Baseline failed: {len(baseline_failed)}")

    if baseline_failed:
        print(f"  Failed IDs: {[j['id'] for j in baseline_failed]}")

    # Baseline-refuses-masked-correct cases
    bfmc = [j for j in judgments if j.get("baseline_refuses_masked_correct")]
    if bfmc:
        print(f"\nBaseline refuses but masked correct ({len(bfmc)}):")
        for j in bfmc:
            print(f"  {j['id']}")

    # ── Table 1: Overall accuracy (baseline-correct subset) ─────────────
    headers = ["Config", "Correct", "Partial", "Unrelated", "Nonsense", "Acc%"]
    rows = []
    by_k = {}

    for label in CONFIG_LABELS:
        counts = {v: 0 for v in VERDICTS}
        for j in baseline_correct:
            v = j["verdicts"].get(label, {}).get("verdict", "UNKNOWN")
            if v in counts:
                counts[v] += 1

        correct = counts["CORRECT"]
        acc = f"{100*correct/bc_count:.1f}" if bc_count > 0 else "N/A"
        rows.append([
            label,
            f"{correct}/{bc_count}",
            counts["PARTIALLY_CORRECT"],
            counts["COHERENT_UNRELATED"],
            counts["NONSENSE"],
            acc,
        ])
        by_k[label] = {
            "correct": correct,
            "partially_correct": counts["PARTIALLY_CORRECT"],
            "coherent_unrelated": counts["COHERENT_UNRELATED"],
            "nonsense": counts["NONSENSE"],
            "total": bc_count,
            "accuracy": round(correct / bc_count, 4) if bc_count > 0 else 0,
        }

    print_table(
        f"OVERALL ACCURACY (baseline-correct subset, n={bc_count})",
        headers, rows
    )

    # ── Table 2: Category-stratified accuracy ───────────────────────────
    by_category = {}
    for cat in CATEGORIES:
        cat_bc = [j for j in baseline_correct if j["category"] == cat]
        cat_n = len(cat_bc)
        if cat_n == 0:
            continue

        cat_rows = []
        cat_data = {}
        for label in CONFIG_LABELS[1:]:  # skip baseline
            correct = sum(
                1 for j in cat_bc
                if j["verdicts"].get(label, {}).get("verdict") == "CORRECT"
            )
            acc = f"{100*correct/cat_n:.1f}"
            cat_rows.append([label, f"{correct}/{cat_n}", acc])
            cat_data[label] = {
                "correct": correct,
                "total": cat_n,
                "accuracy": round(correct / cat_n, 4),
            }

        print_table(
            f"{cat.upper()} (n={cat_n})",
            ["Config", "Correct", "Acc%"],
            cat_rows,
        )
        by_category[cat] = cat_data

    # ── Table 3: Compact cross-category comparison ──────────────────────
    # One row per k-value, one column per category
    print(f"\n{'='*90}")
    print("ACCURACY BY K-VALUE AND CATEGORY (baseline-correct subset)")
    print(f"{'='*90}")

    cat_short = {
        "factual_recall": "Fact",
        "counterfactual": "CntrF",
        "multi_step": "Multi",
        "long_context": "Long",
        "multi_turn": "MTurn",
        "instruction_following": "Instr",
    }
    header = f"{'Config':<10}"
    for cat in CATEGORIES:
        header += f"{cat_short.get(cat, cat[:5]):>8}"
    header += f"{'ALL':>8}"
    print(header)
    print("-" * len(header))

    for label in CONFIG_LABELS[1:]:  # skip baseline
        line = f"{label:<10}"
        for cat in CATEGORIES:
            cat_bc = [j for j in baseline_correct if j["category"] == cat]
            if not cat_bc:
                line += f"{'N/A':>8}"
                continue
            correct = sum(
                1 for j in cat_bc
                if j["verdicts"].get(label, {}).get("verdict") == "CORRECT"
            )
            pct = f"{100*correct/len(cat_bc):.0f}%"
            line += f"{pct:>8}"
        # ALL
        all_correct = by_k.get(label, {}).get("correct", 0)
        all_pct = f"{100*all_correct/bc_count:.0f}%" if bc_count > 0 else "N/A"
        line += f"{all_pct:>8}"
        print(line)

    # ── Verbose: per-prompt details ─────────────────────────────────────
    if verbose:
        print(f"\n{'='*80}")
        print("PER-PROMPT DETAILS")
        print(f"{'='*80}")
        for j in judgments:
            gen = generations_lookup.get(j["id"], {})
            question = ""
            for msg in reversed(gen.get("messages", [])):
                if msg.get("role") == "user":
                    question = msg["content"][:80]
                    break

            print(f"\n--- {j['id']} ({j['category']}) ---")
            print(f"Q: {question}")
            print(f"Ref: {gen.get('reference_answer', '?')}")
            if gen:
                print(f"Baseline: {gen.get('baseline_output', '?')[:100]}")
            for label in CONFIG_LABELS:
                v = j["verdicts"].get(label, {})
                verdict = v.get("verdict", "?")
                reasoning = v.get("reasoning", "")[:80]
                marker = "  " if verdict == "CORRECT" else ">>"
                if label == "baseline":
                    continue
                output_text = ""
                if gen and label.startswith("k="):
                    k_str = label[2:]
                    output_text = gen.get("masked_outputs", {}).get(k_str, "")[:60]
                print(f"  {marker} {label:<10} {verdict:<20} {output_text}")

    # ── Build summary ───────────────────────────────────────────────────
    summary = {
        "total_prompts": total,
        "baseline_correct": bc_count,
        "baseline_failed_ids": [j["id"] for j in baseline_failed],
        "baseline_refuses_masked_correct": [j["id"] for j in bfmc],
        "by_k_value": by_k,
        "by_category": by_category,
    }

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {SUMMARY_FILE}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="KV cache ablation analysis")
    parser.add_argument("--dir", type=str, default=OUTPUT_DIR,
                        help="Results directory (default: eval_results)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-prompt details")
    args = parser.parse_args()

    judgments_file = os.path.join(args.dir, "eval_judgments.jsonl")
    generations_file = os.path.join(args.dir, "eval_generations.jsonl")
    summary_file = os.path.join(args.dir, "summary.json")

    judgments = read_jsonl(judgments_file)
    if not judgments:
        print(f"No judgments found at {judgments_file}. Run judge.py first.")
        return

    # Load generations for verbose output
    generations = read_jsonl(generations_file)
    gen_lookup = {g["id"]: g for g in generations}

    # Override summary output path
    global SUMMARY_FILE
    SUMMARY_FILE = summary_file

    analyze(judgments, gen_lookup, verbose=args.verbose)


if __name__ == "__main__":
    main()
