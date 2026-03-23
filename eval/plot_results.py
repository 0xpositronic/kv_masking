"""
KV Cache Ablation — Plot Generation (v2)
==========================================
Generates publication-quality plots from llama_eval_results.
Supports multi-t comparison (t=4 vs t=5 vs t=6) and uses proportional K-scale axis.

Usage:
    python eval/plot_results.py                                  # default: llama_eval_results
    python eval/plot_results.py --dir llama_eval_results        # explicit
    python eval/plot_results.py --dir llama_eval_results/n5     # single config
"""

import json
import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(ROOT_DIR, "llama_eval_results/")
K_SCALES = [-1.0, -0.5, -0.25, -0.05, 0, 0.05, 0.25, 0.5, 0.75, 1.0, 1.50]

VERDICTS = ["CORRECT", "PARTIALLY_CORRECT", "COHERENT_UNRELATED", "NONSENSE"]
VERDICT_COLORS = {
    "CORRECT": "#2ecc71",
    "PARTIALLY_CORRECT": "#f39c12",
    "COHERENT_UNRELATED": "#e74c3c",
    "NONSENSE": "#8e44ad",
}
VERDICT_LABELS = {
    "CORRECT": "Correct",
    "PARTIALLY_CORRECT": "Partial",
    "COHERENT_UNRELATED": "Unrelated",
    "NONSENSE": "Nonsense",
}
CATEGORY_NAMES = {
    "factual_recall": "Factual Recall",
    "counterfactual": "Counterfactual",
    "multi_step": "Multi-Step",
    "long_context": "Long Context",
    "multi_turn": "Multi-Turn",
    "instruction_following": "Instruction",
}
CATEGORIES = list(CATEGORY_NAMES.keys())
CATEGORY_COLORS = {
    "factual_recall": "#3498db",
    "counterfactual": "#e74c3c",
    "multi_step": "#2ecc71",
    "long_context": "#9b59b6",
    "multi_turn": "#f39c12",
    "instruction_following": "#1abc9c",
}

# t-config display info (t = number of unmasked tokens)
T_INFO = {
    4: {"label": "last 4 tokens (includes the assistant header)", "color": "#f1d206", "ls": "-"},
    5: {"label": "last 5 tokens (includes the user eot)", "color": "#00d2ff", "ls": "-"},
    6: {"label": "last 6 tokens (includes the user's last token)", "color": "#ff6b6b", "ls": "-"},
}

# Dark theme
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.5,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.facecolor": "#16213e",
    "legend.edgecolor": "#e0e0e0",
    "legend.labelcolor": "#e0e0e0",
    "text.usetex": False,
})


# ── Data loading ─────────────────────────────────────────────────────────────

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


def load_config(config_dir):
    """Load summary + judgments + generations for one n-config directory."""
    summary_file = os.path.join(config_dir, "summary.json")
    judgments_file = os.path.join(config_dir, "eval_judgments.jsonl")
    generations_file = os.path.join(config_dir, "eval_generations.jsonl")

    summary = None
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            summary = json.load(f)

    judgments = read_jsonl(judgments_file)
    generations = read_jsonl(generations_file)

    return summary, judgments, generations


def discover_configs(base_dir):
    """Find n* subdirectories with summary.json files."""
    configs = {}
    for name in sorted(os.listdir(base_dir)):
        subdir = os.path.join(base_dir, name)
        if os.path.isdir(subdir) and name.startswith("n") and name[1:].isdigit():
            n = int(name[1:])
            summary_file = os.path.join(subdir, "summary.json")
            if os.path.exists(summary_file):
                configs[n] = subdir
    return configs


# ── Helper functions ─────────────────────────────────────────────────────────

def get_accs(summary, cat=None):
    """Extract accuracy array for K_SCALES from summary."""
    if cat:
        by_cat = summary["by_category"].get(cat, {})
        return [by_cat.get(f"k={k}", {}).get("accuracy", 0) * 100 for k in K_SCALES]
    else:
        by_k = summary["by_k_value"]
        return [by_k.get(f"k={k}", {}).get("accuracy", 0) * 100 for k in K_SCALES]


def get_verdict_counts(summary, verdict, normalize=True):
    """Get verdict count array across K_SCALES."""
    by_k = summary["by_k_value"]
    n = by_k.get("k=0", {}).get("total", 1)
    key = verdict.lower()
    counts = [by_k.get(f"k={k}", {}).get(key, 0) for k in K_SCALES]
    if normalize:
        return [c / n * 100 for c in counts]
    return counts


def setup_k_axis(ax, label=True):
    """Configure a proportional K-scale x-axis."""
    ax.set_xlim(K_SCALES[0] - 0.15, K_SCALES[-1] + 0.15)
    ax.set_xticks(K_SCALES)
    ax.set_xticklabels([str(k) for k in K_SCALES], rotation=45, ha="right")
    if label:
        ax.set_xlabel("K-Scale")
    ax.grid(True, alpha=0.3)


def save(fig, plots_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


# ── Single-config plots ─────────────────────────────────────────────────────

def plot_overall_accuracy(summary, t, plots_dir):
    """Overall accuracy vs K-scale with proportional axis."""
    fig, ax = plt.subplots(figsize=(12, 6))
    accs = get_accs(summary)
    info = T_INFO.get(t, {"color": "#00d2ff", "label": f"t={t}"})

    ax.plot(K_SCALES, accs, "o-", color=info["color"], linewidth=2.5, markersize=8,
            label=f"Overall (n={summary['baseline_correct']})")

    peak_i = int(np.argmax(accs))
    ax.annotate(f"{accs[peak_i]:.1f}%", (K_SCALES[peak_i], accs[peak_i]),
                textcoords="offset points", xytext=(0, 14),
                ha="center", fontsize=11, fontweight="bold", color=info["color"])

    setup_k_axis(ax)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Overall Accuracy vs K-Scale — {info['label']}")
    ax.set_ylim(-2, max(accs) + 12)
    ax.legend()
    save(fig, plots_dir, f"t{t}_01_overall_accuracy.png")


def plot_category_accuracy(summary, t, plots_dir):
    """Multi-line category accuracy vs K-scale."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for cat in CATEGORIES:
        accs = get_accs(summary, cat)
        cat_n = summary["by_category"].get(cat, {}).get(f"k={K_SCALES[0]}", {}).get("total", 0)
        if cat_n < 5:
            continue
        color = CATEGORY_COLORS[cat]
        ax.plot(K_SCALES, accs, "o-", color=color, linewidth=2, markersize=6,
                label=f"{CATEGORY_NAMES[cat]} (n={cat_n})")

    setup_k_axis(ax)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Accuracy by Category — {T_INFO.get(t, {}).get('label', f't={t}')}")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, plots_dir, f"t{t}_02_category_accuracy.png")


def plot_verdict_distribution(summary, t, plots_dir):
    """Stacked bar: verdict distribution with proportional spacing."""
    fig, ax = plt.subplots(figsize=(14, 7))
    bc = summary["by_k_value"].get("k=0", {}).get("total", 1)
    bar_width = 0.15

    bottoms = np.zeros(len(K_SCALES))
    for verdict in VERDICTS:
        vals = np.array(get_verdict_counts(summary, verdict))
        ax.bar(K_SCALES, vals, width=bar_width, bottom=bottoms,
               label=VERDICT_LABELS[verdict], color=VERDICT_COLORS[verdict],
               edgecolor="#1a1a2e", linewidth=0.5)
        bottoms += vals

    setup_k_axis(ax)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Verdict Distribution — {T_INFO.get(t, {}).get('label', f't={t}')} (n={bc})")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    save(fig, plots_dir, f"t{t}_03_verdict_distribution.png")


def plot_verdict_area(summary, t, plots_dir):
    """Area chart: verdict composition across K-scale."""
    fig, ax = plt.subplots(figsize=(14, 6))

    stacks = [get_verdict_counts(summary, v) for v in VERDICTS]
    ax.stackplot(K_SCALES, *stacks, labels=[VERDICT_LABELS[v] for v in VERDICTS],
                 colors=[VERDICT_COLORS[v] for v in VERDICTS], alpha=0.85)

    setup_k_axis(ax)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Verdict Composition — {T_INFO.get(t, {}).get('label', f't={t}')}")
    ax.set_ylim(0, 100)
    ax.legend(loc="center right")
    save(fig, plots_dir, f"t{t}_04_verdict_area.png")


def plot_strict_vs_lenient(summary, t, plots_dir):
    """Strict (correct) vs lenient (correct+partial) accuracy."""
    fig, ax = plt.subplots(figsize=(12, 6))
    by_k = summary["by_k_value"]
    bc = by_k.get("k=0", {}).get("total", 1)

    strict = [by_k.get(f"k={k}", {}).get("correct", 0) / bc * 100 for k in K_SCALES]
    lenient = [(by_k.get(f"k={k}", {}).get("correct", 0) +
                by_k.get(f"k={k}", {}).get("partially_correct", 0)) / bc * 100
               for k in K_SCALES]

    ax.fill_between(K_SCALES, strict, lenient, alpha=0.3, color="#f39c12")
    ax.fill_between(K_SCALES, 0, strict, alpha=0.3, color="#2ecc71")
    ax.plot(K_SCALES, lenient, "s-", color="#f39c12", linewidth=2, markersize=6,
            label="Correct + Partial")
    ax.plot(K_SCALES, strict, "o-", color="#2ecc71", linewidth=2, markersize=6,
            label="Correct Only")

    setup_k_axis(ax)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Strict vs Lenient Accuracy — {T_INFO.get(t, {}).get('label', f't={t}')}")
    ax.set_ylim(0, max(lenient) + 10)
    ax.legend()
    save(fig, plots_dir, f"t{t}_05_strict_vs_lenient.png")


def plot_heatmap(summary, t, plots_dir):
    """Heatmap: K-scale (rows) x Category (columns)."""
    by_cat = summary["by_category"]
    cats = [c for c in CATEGORIES if c in by_cat]
    if not cats:
        return

    data = np.array([[by_cat[c].get(f"k={k}", {}).get("accuracy", 0) * 100
                       for c in cats] for k in K_SCALES])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(70, data.max()))

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([CATEGORY_NAMES[c] for c in cats], rotation=30, ha="right")
    ax.set_yticks(range(len(K_SCALES)))
    ax.set_yticklabels([str(k) for k in K_SCALES])
    ax.set_ylabel("K-Scale")
    ax.set_title(f"Accuracy Heatmap — {T_INFO.get(t, {}).get('label', f't={t}')}")

    for i in range(len(K_SCALES)):
        for j in range(len(cats)):
            val = data[i, j]
            color = "white" if val > 35 else "#e0e0e0"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Accuracy (%)")
    save(fig, plots_dir, f"t{t}_06_heatmap.png")


def plot_nonsense_by_category(judgments, t, plots_dir):
    """Nonsense rate per category across K-scale."""
    fig, ax = plt.subplots(figsize=(14, 7))
    bc = [j for j in judgments if not j["baseline_failed"]]

    for cat in CATEGORIES:
        cat_bc = [j for j in bc if j["category"] == cat]
        if len(cat_bc) < 5:
            continue
        rates = []
        for k in K_SCALES:
            ns = sum(1 for j in cat_bc
                     if j["verdicts"].get(f"k={k}", {}).get("verdict") == "NONSENSE")
            rates.append(ns / len(cat_bc) * 100)
        ax.plot(K_SCALES, rates, "o-", color=CATEGORY_COLORS[cat], linewidth=1.5,
                markersize=5, label=f"{CATEGORY_NAMES[cat]} (n={len(cat_bc)})")

    setup_k_axis(ax)
    ax.set_ylabel("Nonsense Rate (%)")
    ax.set_title(f"Nonsense Rate by Category — {T_INFO.get(t, {}).get('label', f't={t}')}")
    ax.legend(fontsize=9)
    save(fig, plots_dir, f"t{t}_07_nonsense_rate.png")


# ── Comparison plots ─────────────────────────────────────────────────────────

def plot_compare_overall(summaries, plots_dir):
    """Overlay: overall accuracy across all t configs."""
    t_vals = sorted(summaries.keys())
    t_label = " vs ".join(f"t={t}" for t in t_vals)
    fig, ax = plt.subplots(figsize=(12, 6))

    for t, summary in sorted(summaries.items()):
        accs = get_accs(summary)
        info = T_INFO.get(t, {"color": "#ffffff", "label": f"t={t}", "ls": "-"})
        ax.plot(K_SCALES, accs, f"o{info['ls']}", color=info["color"], linewidth=2.5,
                markersize=8, label=f"{info['label']} (bc={summary['baseline_correct']})")
        peak_i = int(np.argmax(accs))
        ax.annotate(f"{accs[peak_i]:.1f}%", (K_SCALES[peak_i], accs[peak_i]),
                    textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=10, fontweight="bold", color=info["color"])

    setup_k_axis(ax)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Overall Accuracy: {t_label}")
    ax.set_ylim(-2, 55)
    ax.legend(fontsize=11)
    save(fig, plots_dir, "cmp_01_overall_accuracy.png")


def plot_compare_category_lines(summaries, plots_dir):
    """Per-category comparison: one subplot per category, all t lines."""
    t_label = " vs ".join(f"t={t}" for t in sorted(summaries.keys()))
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, cat in enumerate(CATEGORIES):
        ax = axes[idx]
        for t, summary in sorted(summaries.items()):
            accs = get_accs(summary, cat)
            info = T_INFO.get(t, {"color": "#ffffff", "label": f"t={t}", "ls": "-"})
            ax.plot(K_SCALES, accs, f"o{info['ls']}", color=info["color"],
                    linewidth=2, markersize=5, label=info["label"])

        setup_k_axis(ax, label=(idx >= 3))
        ax.set_title(CATEGORY_NAMES[cat], fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel("Accuracy (%)")
        if idx == 2:
            ax.legend(fontsize=8)

    fig.suptitle(f"Per-Category Accuracy: {t_label}", fontsize=16, y=1.01)
    save(fig, plots_dir, "cmp_02_category_small_multiples.png")


def plot_compare_delta(summaries, plots_dir):
    """Delta accuracy vs smallest n across K-scale."""
    t_vals = sorted(summaries.keys())
    if len(t_vals) < 2:
        return
    t_base = t_vals[0]

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.15 / max(1, len(t_vals) - 1) * 2
    offsets = np.linspace(-bar_width * (len(t_vals) - 2) / 2, bar_width * (len(t_vals) - 2) / 2, len(t_vals) - 1)

    accs_base = get_accs(summaries[t_base])
    for i, t in enumerate(t_vals[1:]):
        accs_t = get_accs(summaries[t])
        delta = [at - ab for at, ab in zip(accs_t, accs_base)]
        info = T_INFO.get(t, {"color": "#ffffff", "label": f"t={t}"})
        positions = [k + offsets[i] for k in K_SCALES]
        ax.bar(positions, delta, width=bar_width * 0.9,
               color=info["color"], edgecolor="#1a1a2e", alpha=0.8,
               label=f"t={t} vs t={t_base}")

    ax.axhline(0, color="#e0e0e0", linewidth=0.5, alpha=0.5)
    setup_k_axis(ax)
    ax.set_ylabel("Accuracy Delta (pp)")
    ax.set_title(f"Accuracy Gain vs t={t_base}")
    ax.legend()
    save(fig, plots_dir, "cmp_03_delta_overall.png")


def plot_compare_delta_by_category(summaries, plots_dir):
    """Grouped bar: delta vs smallest t at key K-values, per category."""
    t_vals = sorted(summaries.keys())
    if len(t_vals) < 2:
        return
    t_base = t_vals[0]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(CATEGORIES))
    key_ks = [0, 0.25, 0.5, 0.75]
    width = 0.18

    for i, k in enumerate(key_ks):
        # Use max t vs base t
        t_max = t_vals[-1]
        deltas = []
        for cat in CATEGORIES:
            a_base = summaries[t_base]["by_category"].get(cat, {}).get(f"k={k}", {}).get("accuracy", 0) * 100
            a_max = summaries[t_max]["by_category"].get(cat, {}).get(f"k={k}", {}).get("accuracy", 0) * 100
            deltas.append(a_max - a_base)
        offset = (i - len(key_ks)/2 + 0.5) * width
        bars = ax.bar(x + offset, deltas, width, label=f"k={k}",
                      alpha=0.85, edgecolor="#1a1a2e")
        for j, bar in enumerate(bars):
            h = bar.get_height()
            if abs(h) > 1:
                ax.text(bar.get_x() + bar.get_width()/2, h + (0.5 if h >= 0 else -2),
                        f"{h:+.0f}", ha="center", fontsize=7, color="#e0e0e0")

    ax.axhline(0, color="#e0e0e0", linewidth=0.5, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_NAMES[c] for c in CATEGORIES], rotation=20, ha="right")
    ax.set_ylabel("Accuracy Delta (pp)")
    ax.set_title(f"Per-Category Accuracy Gain (t={t_max} vs t={t_base})")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    save(fig, plots_dir, "cmp_04_delta_by_category.png")


def plot_compare_heatmap_side_by_side(summaries, plots_dir):
    """Side-by-side heatmaps for all t configs."""
    t_vals = sorted(summaries.keys())
    fig, axes = plt.subplots(1, len(t_vals), figsize=(12 * len(t_vals), 8))
    if len(t_vals) == 1:
        axes = [axes]

    for ax, t in zip(axes, t_vals):
        by_cat = summaries[t]["by_category"]
        cats = [c for c in CATEGORIES if c in by_cat]
        data = np.array([[by_cat[c].get(f"k={k}", {}).get("accuracy", 0) * 100
                           for c in cats] for k in K_SCALES])

        im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=85)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([CATEGORY_NAMES[c] for c in cats], rotation=30, ha="right")
        ax.set_yticks(range(len(K_SCALES)))
        ax.set_yticklabels([str(k) for k in K_SCALES])
        ax.set_ylabel("K-Scale")
        info = T_INFO.get(t, {"label": f"t={t}"})
        ax.set_title(info["label"], fontsize=13)

        for i in range(len(K_SCALES)):
            for j in range(len(cats)):
                val = data[i, j]
                color = "white" if val > 35 else "#e0e0e0"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    fig.colorbar(im, ax=axes, shrink=0.6, label="Accuracy (%)", pad=0.02)
    fig.suptitle("Accuracy Heatmap Comparison", fontsize=16)
    fig.subplots_adjust(wspace=0.3)
    save(fig, plots_dir, "cmp_05_heatmap_comparison.png")


def plot_compare_verdict_side_by_side(summaries, plots_dir):
    """Side-by-side verdict area charts."""
    t_vals = sorted(summaries.keys())
    fig, axes = plt.subplots(1, len(t_vals), figsize=(10 * len(t_vals), 5), sharey=True)
    if len(t_vals) == 1:
        axes = [axes]

    for ax, t in zip(axes, t_vals):
        summary = summaries[t]
        stacks = [get_verdict_counts(summary, v) for v in VERDICTS]
        ax.stackplot(K_SCALES, *stacks,
                     labels=[VERDICT_LABELS[v] for v in VERDICTS],
                     colors=[VERDICT_COLORS[v] for v in VERDICTS], alpha=0.85)
        setup_k_axis(ax)
        info = T_INFO.get(t, {"label": f"t={t}"})
        ax.set_title(info["label"])
        ax.set_ylim(0, 100)
        if t == t_vals[0]:
            ax.set_ylabel("Percentage (%)")
        if t == t_vals[-1]:
            ax.legend(loc="center right", fontsize=8)

    fig.suptitle("Verdict Composition Comparison", fontsize=16)
    save(fig, plots_dir, "cmp_06_verdict_comparison.png")


def plot_compare_grouped_bars_at_k(summaries, plots_dir):
    """Grouped bars: all t configs at key K values, per category."""
    t_vals = sorted(summaries.keys())
    if len(t_vals) < 2:
        return
    t_label = " vs ".join(f"t={t}" for t in t_vals)

    key_ks = [0, 0.5]
    fig, axes = plt.subplots(1, len(key_ks), figsize=(8 * len(key_ks), 6), sharey=True)

    for ki, (ax, k) in enumerate(zip(axes, key_ks)):
        x = np.arange(len(CATEGORIES))
        width = 0.8 / len(t_vals)

        for ti, t in enumerate(t_vals):
            accs = [summaries[t]["by_category"].get(c, {}).get(f"k={k}", {}).get("accuracy", 0) * 100
                    for c in CATEGORIES]
            info = T_INFO.get(t, {"color": "#ffffff", "label": f"t={t}"})
            offset = (ti - len(t_vals) / 2 + 0.5) * width
            bars = ax.bar(x + offset, accs, width, label=info["label"],
                          color=info["color"], edgecolor="#1a1a2e", alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.0f}%",
                            ha="center", va="bottom", fontsize=7, color=info["color"])

        ax.set_xticks(x)
        ax.set_xticklabels([CATEGORY_NAMES[c] for c in CATEGORIES], rotation=25, ha="right")
        ax.set_title(f"k={k}")
        if ki == len(key_ks) - 1:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle(f"Per-Category Accuracy: {t_label}", fontsize=16)
    save(fig, plots_dir, "cmp_07_grouped_bars_at_key_k.png")


def plot_compare_factual_counterfactual_focus(summaries, plots_dir):
    """Focused comparison on factual + counterfactual (highest accuracy categories)."""
    t_vals = sorted(summaries.keys())
    if len(t_vals) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ai, (ax, cat, title) in enumerate([
        (axes[0], "factual_recall", "Factual Recall"),
        (axes[1], "counterfactual", "Counterfactual"),
    ]):
        for t in t_vals:
            accs = get_accs(summaries[t], cat)
            info = T_INFO.get(t, {"color": "#ffffff", "label": f"t={t}"})
            ax.plot(K_SCALES, accs, "o-", color=info["color"], linewidth=2.5,
                    markersize=7, label=info["label"])
            ax.fill_between(K_SCALES, accs, alpha=0.1, color=info["color"])

        setup_k_axis(ax)
        ax.set_title(title, fontsize=14)
        if ai == 1:
            ax.legend(fontsize=10)

    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("High-Accuracy Categories: Factual & Counterfactual", fontsize=16)
    save(fig, plots_dir, "cmp_08_factual_counterfactual_focus.png")


def plot_compare_nonsense(summaries, judgments_by_t, plots_dir):
    """Nonsense rate comparison across all t configs."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for t in sorted(summaries.keys()):
        rates = get_verdict_counts(summaries[t], "NONSENSE")
        info = T_INFO.get(t, {"color": "#ffffff", "label": f"t={t}"})
        ax.plot(K_SCALES, rates, "o-", color=info["color"], linewidth=2.5,
                markersize=7, label=info["label"])

    setup_k_axis(ax)
    ax.set_ylabel("Nonsense Rate (%)")
    t_label = " vs ".join(f"t={t}" for t in sorted(summaries.keys()))
    ax.set_title(f"Nonsense Rate: {t_label}")
    ax.legend(fontsize=11)
    save(fig, plots_dir, "cmp_09_nonsense_comparison.png")


def plot_compare_strict_lenient(summaries, plots_dir):
    """Strict vs lenient for all t values."""
    t_vals = sorted(summaries.keys())
    fig, axes = plt.subplots(1, len(t_vals), figsize=(7 * len(t_vals), 6), sharey=True)
    if len(t_vals) == 1:
        axes = [axes]

    for ax, t in zip(axes, t_vals):
        by_k = summaries[t]["by_k_value"]
        bc = by_k.get("k=0", {}).get("total", 1)
        strict = [by_k.get(f"k={k}", {}).get("correct", 0) / bc * 100 for k in K_SCALES]
        lenient = [(by_k.get(f"k={k}", {}).get("correct", 0) +
                    by_k.get(f"k={k}", {}).get("partially_correct", 0)) / bc * 100
                   for k in K_SCALES]

        ax.fill_between(K_SCALES, strict, lenient, alpha=0.3, color="#f39c12")
        ax.fill_between(K_SCALES, 0, strict, alpha=0.3, color="#2ecc71")
        ax.plot(K_SCALES, lenient, "s-", color="#f39c12", linewidth=2, markersize=6,
                label="Correct + Partial")
        ax.plot(K_SCALES, strict, "o-", color="#2ecc71", linewidth=2, markersize=6,
                label="Correct Only")

        setup_k_axis(ax)
        info = T_INFO.get(t, {"label": f"t={t}"})
        ax.set_title(info["label"])
        if t == t_vals[-1]:
            ax.legend(fontsize=9)

    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("Strict vs Lenient Accuracy Comparison", fontsize=16)
    save(fig, plots_dir, "cmp_10_strict_lenient_comparison.png")


def plot_compare_overall_with_categories(summaries, plots_dir):
    """Combined: overall accuracy for all t with category lines for context."""
    t_vals = sorted(summaries.keys())
    if len(t_vals) < 2:
        return
    t_max = t_vals[-1]

    fig, ax = plt.subplots(figsize=(14, 7))

    # Category lines (thin, dimmed) for largest t only
    for cat in CATEGORIES:
        accs = get_accs(summaries[t_max], cat)
        ax.plot(K_SCALES, accs, "-", color=CATEGORY_COLORS[cat], linewidth=1,
                alpha=0.35, label=f"t={t_max} {CATEGORY_NAMES[cat]}")

    # Overall lines (bold) for all t
    for t in t_vals:
        accs = get_accs(summaries[t])
        info = T_INFO.get(t, {"color": "#ffffff", "label": f"t={t}"})
        ax.plot(K_SCALES, accs, "o-", color=info["color"], linewidth=3,
                markersize=9, label=info["label"], zorder=10)

    setup_k_axis(ax)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Overall Accuracy with Category Context (t={t_max})")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    save(fig, plots_dir, "cmp_11_overall_with_categories.png")


def plot_compare_baseline_breakdown(judgments_by_t, plots_dir):
    """Baseline breakdown pie charts side by side."""
    from collections import Counter

    t_vals = sorted(judgments_by_t.keys())
    fig, axes = plt.subplots(1, len(t_vals), figsize=(7 * len(t_vals), 6))
    if len(t_vals) == 1:
        axes = [axes]

    for ax, t in zip(axes, t_vals):
        judgments = judgments_by_t[t]
        bc = sum(1 for j in judgments if not j["baseline_failed"])
        bf = sum(1 for j in judgments if j["baseline_failed"])
        ax.pie([bc, bf], labels=[f"Correct ({bc})", f"Failed ({bf})"],
               colors=["#2ecc71", "#e74c3c"], autopct="%1.0f%%",
               textprops={"color": "#e0e0e0"}, startangle=90,
               wedgeprops={"edgecolor": "#1a1a2e"})
        info = T_INFO.get(t, {"label": f"t={t}"})
        ax.set_title(f"Baseline — {info['label']}")

    fig.suptitle("Baseline Accuracy", fontsize=16)
    save(fig, plots_dir, "cmp_12_baseline_breakdown.png")


def plot_compare_seqlen(summaries, judgments_by_t, generations_by_t, plots_dir):
    """Sequence length vs accuracy at k=0 for each t config."""
    t_vals = sorted(summaries.keys())
    fig, axes = plt.subplots(1, len(t_vals), figsize=(8 * len(t_vals), 6), sharey=True)
    if len(t_vals) == 1:
        axes = [axes]

    for ax, t in zip(axes, t_vals):
        judgments = judgments_by_t[t]
        generations = generations_by_t[t]
        j_by_id = {j["id"]: j for j in judgments}

        seq_lens = []
        correct = []
        cats = []
        for g in generations:
            j = j_by_id.get(g["id"])
            if j is None or j["baseline_failed"]:
                continue
            seq_lens.append(g["seq_len"])
            v = j["verdicts"].get("k=0", {}).get("verdict", "")
            correct.append(1 if v == "CORRECT" else 0)
            cats.append(g["category"])

        seq_lens = np.array(seq_lens)
        correct = np.array(correct)

        # Bin by quartiles
        bins = np.percentile(seq_lens, [0, 25, 50, 75, 100])
        info = T_INFO.get(t, {"color": "#ffffff", "label": f"t={t}"})

        for i in range(len(bins) - 1):
            mask = (seq_lens >= bins[i]) & (seq_lens < bins[i+1] + 1)
            if mask.sum() > 0:
                center = (bins[i] + bins[i+1]) / 2
                acc = correct[mask].mean() * 100
                ax.bar(center, acc, width=(bins[i+1] - bins[i]) * 0.8,
                       color=info["color"], alpha=0.7, edgecolor="#1a1a2e")
                ax.text(center, acc + 1.5, f"{acc:.0f}%\n(n={mask.sum()})",
                        ha="center", fontsize=8, color="#e0e0e0")

        setup_k_axis.__wrapped__ if hasattr(setup_k_axis, '__wrapped__') else None
        ax.set_xlabel("Sequence Length (tokens)")
        ax.set_title(f"{info['label']} — k=0")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Accuracy at k=0 (%)")
    fig.suptitle("Sequence Length vs Accuracy", fontsize=16)
    save(fig, plots_dir, "cmp_13_seqlen_vs_accuracy.png")


def plot_individual_category_pair(summaries, cat, plots_dir):
    """Individual detailed plot for a single category, all t values."""
    t_vals = sorted(summaries.keys())
    if len(t_vals) < 2:
        return
    t_base, t_max = t_vals[0], t_vals[-1]
    t_label = " vs ".join(f"t={t}" for t in t_vals)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Top: accuracy lines
    for t in t_vals:
        accs = get_accs(summaries[t], cat)
        info = T_INFO.get(t, {"color": "#ffffff", "label": f"t={t}"})
        ax1.plot(K_SCALES, accs, "o-", color=info["color"], linewidth=2.5,
                 markersize=7, label=info["label"])
        ax1.fill_between(K_SCALES, accs, alpha=0.1, color=info["color"])

    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(f"{CATEGORY_NAMES[cat]} — {t_label}")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bottom: delta (max t vs min t)
    accs_base = get_accs(summaries[t_base], cat)
    accs_max = get_accs(summaries[t_max], cat)
    delta = [am - ab for am, ab in zip(accs_max, accs_base)]

    colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in delta]
    ax2.bar(K_SCALES, delta, width=0.15, color=colors, edgecolor="#1a1a2e", alpha=0.8)
    ax2.axhline(0, color="#e0e0e0", linewidth=0.5, alpha=0.5)
    for i, d in enumerate(delta):
        if abs(d) > 0.5:
            ax2.text(K_SCALES[i], d + (1 if d >= 0 else -2),
                     f"{d:+.0f}", ha="center", fontsize=9, color="#e0e0e0")

    setup_k_axis(ax2)
    ax2.set_ylabel("Delta (pp)")
    ax2.set_title(f"Accuracy delta (t={t_max} vs t={t_base})")

    save(fig, plots_dir, f"cat_{cat}_detail.png")


def plot_delta_heatmap(summaries, plots_dir):
    """Heatmap of delta (max t vs min t) by K-scale x Category."""
    t_vals = sorted(summaries.keys())
    if len(t_vals) < 2:
        return
    t_base, t_max = t_vals[0], t_vals[-1]

    cats = [c for c in CATEGORIES
            if c in summaries[t_base]["by_category"] and c in summaries[t_max]["by_category"]]

    data = np.array([
        [summaries[t_max]["by_category"][c].get(f"k={k}", {}).get("accuracy", 0) * 100
         - summaries[t_base]["by_category"][c].get(f"k={k}", {}).get("accuracy", 0) * 100
         for c in cats]
        for k in K_SCALES
    ])

    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = max(abs(data.min()), abs(data.max()), 1)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([CATEGORY_NAMES[c] for c in cats], rotation=30, ha="right")
    ax.set_yticks(range(len(K_SCALES)))
    ax.set_yticklabels([str(k) for k in K_SCALES])
    ax.set_ylabel("K-Scale")
    ax.set_title(f"Accuracy Delta Heatmap (t={t_max} vs t={t_base})")

    for i in range(len(K_SCALES)):
        for j in range(len(cats)):
            val = data[i, j]
            color = "black" if abs(val) < vmax * 0.3 else "white"
            sign = "+" if val > 0 else ""
            ax.text(j, i, f"{sign}{val:.0f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Delta (pp)")
    save(fig, plots_dir, "cmp_14_delta_heatmap.png")


def plot_peak_accuracy_comparison(summaries, plots_dir):
    """Bar chart: peak accuracy per category for all t configs."""
    t_vals = sorted(summaries.keys())
    if len(t_vals) < 2:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(CATEGORIES))
    width = 0.8 / len(t_vals)

    for ti, t in enumerate(t_vals):
        peaks = []
        peak_ks = []
        for cat in CATEGORIES:
            accs = get_accs(summaries[t], cat)
            peaks.append(max(accs))
            peak_ks.append(K_SCALES[int(np.argmax(accs))])
        info = T_INFO.get(t, {"color": "#ffffff", "label": f"t={t}"})
        offset = (ti - len(t_vals) / 2 + 0.5) * width
        bars = ax.bar(x + offset, peaks, width, label=info["label"],
                      color=info["color"], edgecolor="#1a1a2e", alpha=0.85)
        for bar, pk in zip(bars, peak_ks):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f"{h:.0f}%\n(k={pk})", ha="center", fontsize=7, color=info["color"])

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_NAMES[c] for c in CATEGORIES], rotation=20, ha="right")
    ax.set_ylabel("Peak Accuracy (%)")
    ax.set_title("Peak Accuracy per Category with Optimal K-Scale")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    save(fig, plots_dir, "cmp_15_peak_accuracy.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--dir", type=str, default=OUTPUT_DIR,
                        help="Base results directory or single n-config dir")
    args = parser.parse_args()

    base_dir = args.dir
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Discover configs
    configs = discover_configs(base_dir)
    if not configs:
        # Single directory mode
        summary_file = os.path.join(base_dir, "summary.json")
        if os.path.exists(summary_file):
            # Guess t from directory name
            dirname = os.path.basename(base_dir)
            t = int(dirname[1:]) if dirname.startswith("n") and dirname[1:].isdigit() else 0
            configs = {t: base_dir}
        else:
            print(f"No summary.json found. Run analyze.py first.")
            return

    print(f"Found configs: {sorted(configs.keys())}")
    print(f"Output: {plots_dir}/\n")

    # Load all data
    summaries = {}
    judgments_by_t = {}
    generations_by_t = {}
    for t, config_dir in sorted(configs.items()):
        summary, judgments, generations = load_config(config_dir)
        if summary:
            summaries[t] = summary
            judgments_by_t[t] = judgments
            generations_by_t[t] = generations

    print("Generating per-config plots...")
    for t in sorted(summaries.keys()):
        print(f"\n  --- t={t} ---")
        plot_overall_accuracy(summaries[t], t, plots_dir)
        plot_category_accuracy(summaries[t], t, plots_dir)
        plot_verdict_distribution(summaries[t], t, plots_dir)
        plot_verdict_area(summaries[t], t, plots_dir)
        plot_strict_vs_lenient(summaries[t], t, plots_dir)
        plot_heatmap(summaries[t], t, plots_dir)
        plot_nonsense_by_category(judgments_by_t[t], t, plots_dir)

    if len(summaries) >= 2:
        print("\nGenerating comparison plots...")
        plot_compare_overall(summaries, plots_dir)
        plot_compare_category_lines(summaries, plots_dir)
        plot_compare_delta(summaries, plots_dir)
        plot_compare_delta_by_category(summaries, plots_dir)
        plot_compare_heatmap_side_by_side(summaries, plots_dir)
        plot_compare_verdict_side_by_side(summaries, plots_dir)
        plot_compare_grouped_bars_at_k(summaries, plots_dir)
        plot_compare_factual_counterfactual_focus(summaries, plots_dir)
        plot_compare_nonsense(summaries, judgments_by_t, plots_dir)
        plot_compare_strict_lenient(summaries, plots_dir)
        plot_compare_overall_with_categories(summaries, plots_dir)
        plot_compare_baseline_breakdown(judgments_by_t, plots_dir)
        plot_compare_seqlen(summaries, judgments_by_t, generations_by_t, plots_dir)
        plot_delta_heatmap(summaries, plots_dir)
        plot_peak_accuracy_comparison(summaries, plots_dir)

        # Individual category detail plots
        print("\nGenerating per-category detail plots...")
        for cat in CATEGORIES:
            plot_individual_category_pair(summaries, cat, plots_dir)

    total = len(os.listdir(plots_dir))
    print(f"\nDone. {total} plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
