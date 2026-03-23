"""
KV Cache Ablation — HTML Viewer Builder
========================================
Reads eval_generations.jsonl (and optionally eval_judgments.jsonl) from each
n_unmasked subdirectory. Produces a self-contained HTML page with:
- Tabs per n_unmasked config
- Filterable prompt cards showing all outputs + verdicts
- Plots tab (if plots exist)

Usage:
    python build_viewer.py                         # all n_unmasked dirs under eval_results/
    python build_viewer.py --dir eval_results/n3   # single directory
"""

import argparse
import base64
import html
import json
import os

K_SCALES = [-1.0, -0.5, -0.25, -0.05, 0, 0.05, 0.25, 0.5, 0.75, 1.0, 1.50]
CONFIG_LABELS = ["baseline"] + [f"k={k}" for k in K_SCALES]
OUTPUT_DIR = "llama_eval_results"


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


def esc(text):
    return html.escape(str(text))


def verdict_class(verdict):
    if not verdict or verdict == "?":
        return "none"
    return verdict.lower().replace("_", "-")


def format_messages_html(messages):
    """Format a list of chat messages as HTML, showing all turns."""
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = esc(msg["content"])
        if role == "user":
            parts.append(f'<span class="msg-role msg-user">user:</span> {content}')
        elif role == "assistant":
            parts.append(f'<span class="msg-role msg-assistant">assistant:</span> {content}')
        else:
            parts.append(f'<span class="msg-role">{esc(role)}:</span> {content}')
    return "<br>".join(parts)


def build_prompt_card(row, card_id):
    """Build HTML for a single prompt card."""
    has_verdicts = bool(row.get("verdicts"))

    baseline_v = row.get("verdicts", {}).get("baseline", {}).get("verdict", "") if has_verdicts else ""
    baseline_cls = verdict_class(baseline_v) if baseline_v else "none"
    k0_v = row.get("verdicts", {}).get("k=0", {}).get("verdict", "") if has_verdicts else ""
    k0_cls = verdict_class(k0_v) if k0_v else "none"
    k05_v = row.get("verdicts", {}).get("k=0.5", {}).get("verdict", "") if has_verdicts else ""
    k05_cls = verdict_class(k05_v) if k05_v else "none"

    baseline_failed = row.get("baseline_failed", False)
    failed_cls = " baseline-failed" if baseline_failed else ""

    flag = ""
    if has_verdicts and baseline_failed:
        for k in K_SCALES:
            if row.get("verdicts", {}).get(f"k={k}", {}).get("verdict") == "CORRECT":
                flag = '<span class="flag">MASKED&gt;BASE</span>'
                break

    # Summary badges
    baseline_badge = f'<span class="badge {baseline_cls}">{esc(baseline_v)}</span>' if baseline_v else '<span class="badge none">-</span>'
    k0_badge = f'<span class="badge {k0_cls}">k=0: {esc(k0_v)}</span>' if k0_v else ""
    k05_badge = f'<span class="badge {k05_cls}">k=.5: {esc(k05_v)}</span>' if k05_v else ""

    # For the header, show just the last user message as a preview
    last_question = row["question"]
    # Build full conversation HTML for the detail view
    messages_html = format_messages_html(row.get("messages", []))
    is_multi_turn = len(row.get("messages", [])) > 1
    turn_count = len(row.get("messages", []))
    turn_badge = f'<span class="turn-badge">{turn_count}T</span>' if is_multi_turn else ""

    # Search data includes all message content
    search_text = row["id"] + " " + " ".join(
        m.get("content", "")[:200] for m in row.get("messages", [])
    )

    parts = []
    parts.append(f'''
<div class="prompt-card{failed_cls}" data-cat="{esc(row['category'])}"
     data-baseline="{'fail' if baseline_failed else 'pass'}"
     data-k0="{esc(k0_v)}" data-search="{esc(search_text).lower()}">
  <div class="prompt-header" onclick="toggleDetail('{card_id}')">
    <span class="arrow" id="arrow-{card_id}">&#9654;</span>
    <span class="prompt-id">{esc(row['id'])}</span>
    <span class="prompt-cat">{esc(row['category'])}{turn_badge}</span>
    <span class="prompt-question">{esc(last_question[:120])}</span>
    <span class="prompt-baseline">{baseline_badge}</span>
    <span class="prompt-k0">{k0_badge}</span>
    <span class="prompt-k05">{k05_badge}</span>
    {flag}
  </div>
  <div class="prompt-detail" id="detail-{card_id}">
    <div class="detail-meta">
      <span class="label">Input:</span><span class="messages-block">{messages_html}</span>
      <span class="label">Reference:</span><span><b>{esc(row['reference'])}</b></span>
      <span class="label">Seq Length:</span><span>{row['seq_len']}</span>
      <span class="label">Unmasked:</span><span>{row.get('unmasked_positions', '?')}</span>
    </div>
    <div class="output-grid">
''')

    # Baseline row
    bv = row.get("verdicts", {}).get("baseline", {})
    bv_text = bv.get("verdict", "") if bv else ""
    bv_reasoning = bv.get("reasoning", "") if bv else ""
    verdict_html = f'''
        <span class="badge {verdict_class(bv_text)}">{esc(bv_text)}</span>
        <span class="reasoning">{esc(bv_reasoning)}</span>
    ''' if bv_text else '<span class="badge none">no judgment</span>'

    parts.append(f'''
      <div class="output-row">
        <span class="config">baseline</span>
        <span class="text">{esc(row['baseline_output'])}</span>
        <span class="verdict-col">{verdict_html}</span>
      </div>
''')

    # Masked output rows
    for k in K_SCALES:
        label = f"k={k}"
        output = row["masked_outputs"].get(str(k), "[MISSING]")
        v = row.get("verdicts", {}).get(label, {})
        v_text = v.get("verdict", "") if v else ""
        v_cls = verdict_class(v_text)
        reasoning = v.get("reasoning", "") if v else ""
        verdict_html = f'''
          <span class="badge {v_cls}">{esc(v_text)}</span>
          <span class="reasoning">{esc(reasoning)}</span>
        ''' if v_text else '<span class="badge none">no judgment</span>'

        parts.append(f'''
      <div class="output-row">
        <span class="config">{esc(label)}</span>
        <span class="text">{esc(output)}</span>
        <span class="verdict-col">{verdict_html}</span>
      </div>
''')

    parts.append("    </div>\n  </div>\n</div>\n")
    return "".join(parts)


def build_config_tab(tab_id, generations, judgments, label):
    """Build the content for one n_unmasked tab."""
    j_by_id = {j["id"]: j for j in judgments}

    rows = []
    for g in generations:
        j = j_by_id.get(g["id"], {})
        messages = g.get("messages", [])
        # Last user message for header preview
        question = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                question = msg["content"]
                break
        rows.append({
            "id": g["id"],
            "category": g.get("category", ""),
            "messages": messages,
            "question": question,
            "reference": g.get("reference_answer", ""),
            "seq_len": g.get("seq_len", 0),
            "unmasked_positions": g.get("unmasked_positions", []),
            "baseline_output": g.get("baseline_output", ""),
            "masked_outputs": g.get("masked_outputs", {}),
            "verdicts": j.get("verdicts", {}),
            "baseline_failed": j.get("baseline_failed", False),
        })

    parts = []
    for i, row in enumerate(rows):
        card_id = f"{tab_id}-{i}"
        parts.append(build_prompt_card(row, card_id))

    n_judged = len(judgments)
    status = f"{len(generations)} generations"
    if n_judged:
        bc = sum(1 for j in judgments if not j.get("baseline_failed"))
        status += f", {n_judged} judged, {bc} baseline-correct"

    return "".join(parts), status, rows


def discover_configs(base_dir):
    """Find all n_unmasked subdirectories."""
    configs = []
    for name in sorted(os.listdir(base_dir)):
        subdir = os.path.join(base_dir, name)
        gen_file = os.path.join(subdir, "eval_generations.jsonl")
        if os.path.isdir(subdir) and os.path.exists(gen_file):
            configs.append((name, subdir))
    return configs


CSS = """
:root {
    --bg: #0f0f1a; --surface: #1a1a2e; --surface2: #222240; --border: #333355;
    --text: #e0e0e0; --text-dim: #8888aa;
    --correct: #2ecc71; --partial: #f39c12; --unrelated: #e74c3c; --nonsense: #8e44ad;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace; font-size: 13px; }
.header { background: var(--surface); padding: 16px 24px; border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 100; }
.header h1 { font-size: 18px; margin-bottom: 8px; }
.controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.controls label { color: var(--text-dim); font-size: 12px; }
.controls select, .controls input { background: var(--surface2); border: 1px solid var(--border); color: var(--text); padding: 4px 8px; border-radius: 4px; font-family: inherit; font-size: 12px; }
.controls input[type="text"] { width: 200px; }
.stats { color: var(--text-dim); font-size: 12px; margin-left: auto; }
.tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); background: var(--surface); padding: 0 24px; position: sticky; top: 68px; z-index: 99; }
.tab { padding: 10px 20px; cursor: pointer; color: var(--text-dim); border-bottom: 2px solid transparent; font-size: 13px; white-space: nowrap; }
.tab:hover { color: var(--text); }
.tab.active { color: var(--correct); border-bottom-color: var(--correct); }
.tab-content { display: none; }
.tab-content.active { display: block; }
.tab-status { font-size: 10px; color: var(--text-dim); display: block; }
.prompt-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin: 12px 24px; overflow: hidden; }
.prompt-header { display: flex; align-items: center; gap: 12px; padding: 10px 16px; cursor: pointer; user-select: none; }
.prompt-header:hover { background: var(--surface2); }
.prompt-id { font-weight: bold; width: 160px; flex-shrink: 0; }
.prompt-cat { color: var(--text-dim); width: 150px; flex-shrink: 0; }
.prompt-question { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-dim); }
.prompt-baseline { width: 120px; text-align: center; flex-shrink: 0; }
.prompt-k0 { width: 130px; text-align: center; flex-shrink: 0; }
.prompt-k05 { width: 130px; text-align: center; flex-shrink: 0; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: bold; }
.badge.correct { background: var(--correct); color: #000; }
.badge.partially-correct { background: var(--partial); color: #000; }
.badge.coherent-unrelated { background: var(--unrelated); color: #fff; }
.badge.nonsense { background: var(--nonsense); color: #fff; }
.badge.unknown, .badge.error, .badge.none { background: #333; color: #888; }
.prompt-detail { display: none; padding: 0 16px 16px; }
.prompt-detail.open { display: block; }
.detail-meta { display: grid; grid-template-columns: 100px 1fr; gap: 4px 12px; margin-bottom: 12px; padding: 8px; background: var(--surface2); border-radius: 4px; }
.detail-meta .label { color: var(--text-dim); }
.output-grid { display: grid; grid-template-columns: 1fr; gap: 8px; }
.output-row { display: grid; grid-template-columns: 100px 1fr 160px; gap: 8px; padding: 6px 8px; border-radius: 4px; align-items: start; }
.output-row:nth-child(odd) { background: rgba(255,255,255,0.02); }
.output-row .config { font-weight: bold; white-space: nowrap; }
.output-row .text { white-space: pre-wrap; word-break: break-word; color: var(--text); font-size: 12px; max-height: 120px; overflow-y: auto; }
.output-row .verdict-col { display: flex; flex-direction: column; gap: 2px; }
.output-row .reasoning { font-size: 10px; color: var(--text-dim); max-height: 40px; overflow-y: auto; }
.baseline-failed { opacity: 0.6; }
.flag { color: #ff6b6b; font-size: 11px; font-weight: bold; margin-left: 8px; }
.arrow { margin-right: 8px; transition: transform 0.2s; display: inline-block; }
.arrow.open { transform: rotate(90deg); }
.turn-badge { display: inline-block; background: #445; color: #aab; font-size: 10px; padding: 1px 5px; border-radius: 3px; margin-left: 6px; vertical-align: middle; }
.messages-block { line-height: 1.8; }
.msg-role { font-weight: bold; font-size: 11px; padding: 1px 5px; border-radius: 3px; }
.msg-user { color: #3498db; }
.msg-assistant { color: #2ecc71; }
.plots-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(600px, 1fr)); gap: 16px; padding: 24px; }
.plots-grid img { width: 100%; border-radius: 8px; border: 1px solid var(--border); }
.plot-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.plot-title { padding: 8px 12px; font-size: 12px; color: var(--text-dim); border-bottom: 1px solid var(--border); }
.plot-section-title { grid-column: 1 / -1; padding: 12px 0 4px; color: var(--text-dim); font-size: 14px; border-bottom: 1px solid var(--border); margin-top: 8px; }
.empty-msg { padding: 48px; text-align: center; color: var(--text-dim); }
"""

JS = """
function toggleDetail(id) {
    const detail = document.getElementById('detail-' + id);
    const arrow = document.getElementById('arrow-' + id);
    detail.classList.toggle('open');
    arrow.classList.toggle('open');
}

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
        applyFilters();
    });
});

function applyFilters() {
    const cat = document.getElementById('filter-cat').value;
    const baseline = document.getElementById('filter-baseline').value;
    const k0 = document.getElementById('filter-k0').value;
    const search = document.getElementById('filter-search').value.toLowerCase();

    let shown = 0, total = 0;
    const activeTab = document.querySelector('.tab-content.active');
    if (!activeTab) return;

    activeTab.querySelectorAll('.prompt-card').forEach(card => {
        total++;
        let show = true;
        if (cat && card.dataset.cat !== cat) show = false;
        if (baseline && card.dataset.baseline !== baseline) show = false;
        if (k0 && card.dataset.k0 !== k0) show = false;
        if (search && !card.dataset.search.includes(search)) show = false;
        card.style.display = show ? '' : 'none';
        if (show) shown++;
    });
    document.getElementById('stats').textContent = `Showing ${shown} / ${total} prompts`;
}

document.getElementById('filter-cat').addEventListener('change', applyFilters);
document.getElementById('filter-baseline').addEventListener('change', applyFilters);
document.getElementById('filter-k0').addEventListener('change', applyFilters);
document.getElementById('filter-search').addEventListener('input', applyFilters);

// Init
applyFilters();
"""


def build_full_html(configs_data):
    """Build the complete HTML page from all configs."""
    # Collect all categories
    all_cats = set()
    for _, _, rows, _ in configs_data:
        for r in rows:
            all_cats.add(r["category"])
    categories = sorted(all_cats)

    total_gens = sum(len(rows) for _, _, rows, _ in configs_data)

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KV Cache Ablation — Evaluation Viewer</title>
<style>{CSS}</style>
</head>
<body>
<div class="header">
    <h1>KV Cache Ablation — Evaluation Viewer</h1>
    <div class="controls">
        <label>Category:
            <select id="filter-cat">
                <option value="">All</option>
                {"".join(f'<option value="{c}">{c}</option>' for c in categories)}
            </select>
        </label>
        <label>Baseline:
            <select id="filter-baseline">
                <option value="">All</option>
                <option value="pass">Correct</option>
                <option value="fail">Failed</option>
            </select>
        </label>
        <label>k=0 verdict:
            <select id="filter-k0">
                <option value="">All</option>
                <option value="CORRECT">Correct</option>
                <option value="PARTIALLY_CORRECT">Partial</option>
                <option value="COHERENT_UNRELATED">Unrelated</option>
                <option value="NONSENSE">Nonsense</option>
            </select>
        </label>
        <label>Search:
            <input type="text" id="filter-search" placeholder="ID or question text...">
        </label>
        <div class="stats" id="stats">{total_gens} total generations</div>
    </div>
</div>

<div class="tabs">
""")

    # Tab buttons
    for i, (label, status, _, _) in enumerate(configs_data):
        active = " active" if i == 0 else ""
        parts.append(f'    <div class="tab{active}" data-tab="{label}">{label}<span class="tab-status">{status}</span></div>\n')

    # Plots tab button
    parts.append('    <div class="tab" data-tab="plots">Plots</div>\n')
    parts.append("</div>\n\n")

    # Tab contents
    for i, (label, status, _, tab_html) in enumerate(configs_data):
        active = " active" if i == 0 else ""
        parts.append(f'<div class="tab-content{active}" id="tab-{label}">\n')
        parts.append(tab_html)
        parts.append("</div>\n\n")

    # Plots tab — embed as base64 for self-contained HTML
    parts.append('<div class="tab-content" id="tab-plots">\n')

    def embed_plots_from_dir(plots_dir, section_title=None):
        """Read PNGs from a directory and return embedded HTML."""
        embedded = []
        if not os.path.isdir(plots_dir):
            return embedded
        pngs = sorted(f for f in os.listdir(plots_dir) if f.endswith(".png"))
        if not pngs:
            return embedded
        if section_title:
            embedded.append(f'<h3 class="plot-section-title">{esc(section_title)}</h3>\n')
        for pf in pngs:
            filepath = os.path.join(plots_dir, pf)
            with open(filepath, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode("ascii")
            # Derive a readable title from filename
            title = pf.replace(".png", "").replace("_", " ")
            embedded.append(
                f'<div class="plot-card">'
                f'<div class="plot-title">{esc(title)}</div>'
                f'<img src="data:image/png;base64,{b64}" alt="{esc(pf)}">'
                f'</div>\n'
            )
        return embedded

    plot_parts = []

    # Per-n plots (n5_*, n6_*) grouped from base plots dir
    base_plots = os.path.join(OUTPUT_DIR, "plots")
    if os.path.isdir(base_plots):
        all_pngs = sorted(f for f in os.listdir(base_plots) if f.endswith(".png"))

        # Group: per-config plots (n5_*, n6_*), comparison plots (cmp_*), category detail (cat_*)
        per_n = {}
        cmp_plots = []
        cat_plots = []
        for pf in all_pngs:
            if pf.startswith("n") and "_" in pf:
                n_prefix = pf.split("_")[0]  # e.g. "n5"
                per_n.setdefault(n_prefix, []).append(pf)
            elif pf.startswith("cmp_"):
                cmp_plots.append(pf)
            elif pf.startswith("cat_"):
                cat_plots.append(pf)
            else:
                cmp_plots.append(pf)  # fallback

        def embed_png_list(pngs, section_title):
            result = []
            if not pngs:
                return result
            result.append(f'<h3 class="plot-section-title">{esc(section_title)}</h3>\n')
            for pf in pngs:
                filepath = os.path.join(base_plots, pf)
                with open(filepath, "rb") as img_f:
                    b64 = base64.b64encode(img_f.read()).decode("ascii")
                title = pf.replace(".png", "").replace("_", " ")
                result.append(
                    f'<div class="plot-card">'
                    f'<div class="plot-title">{esc(title)}</div>'
                    f'<img src="data:image/png;base64,{b64}" alt="{esc(pf)}">'
                    f'</div>\n'
                )
            return result

        for n_prefix in sorted(per_n.keys()):
            plot_parts.extend(embed_png_list(per_n[n_prefix], f"Per-Config: {n_prefix}"))
        plot_parts.extend(embed_png_list(cmp_plots, "Comparison Plots"))
        plot_parts.extend(embed_png_list(cat_plots, "Per-Category Detail"))

    # Also check per-config subdirectories for plots
    for label, _, _, _ in configs_data:
        sub_plots = os.path.join(OUTPUT_DIR, label, "plots")
        plot_parts.extend(embed_plots_from_dir(sub_plots, f"Config: {label}"))

    if plot_parts:
        parts.append('<div class="plots-grid">\n')
        parts.extend(plot_parts)
        parts.append('</div>\n')
    else:
        parts.append('<p class="empty-msg">No plots found. Run plot_results.py to generate plots.</p>\n')

    parts.append("</div>\n")

    parts.append(f"<script>{JS}</script>\n</body>\n</html>")
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Build HTML viewer for evaluation results")
    parser.add_argument("--dir", type=str, default=None,
                        help="Single results directory (default: auto-discover n* under eval_results/)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path (default: eval_results/viewer.html or <dir>/viewer.html)")
    args = parser.parse_args()

    if args.dir:
        # Single directory mode
        configs = [("results", args.dir)]
        output_path = args.output or os.path.join(args.dir, "viewer.html")
    else:
        # Auto-discover n* subdirectories
        configs = discover_configs(OUTPUT_DIR)
        if not configs:
            # Fall back to base dir
            gen_file = os.path.join(OUTPUT_DIR, "eval_generations.jsonl")
            if os.path.exists(gen_file):
                configs = [("results", OUTPUT_DIR)]
            else:
                print(f"No generation files found under {OUTPUT_DIR}/")
                return
        output_path = args.output or os.path.join(OUTPUT_DIR, "viewer.html")

    # Load data for each config
    configs_data = []
    for label, subdir in configs:
        gen_file = os.path.join(subdir, "eval_generations.jsonl")
        judge_file = os.path.join(subdir, "eval_judgments.jsonl")
        generations = read_jsonl(gen_file)
        judgments = read_jsonl(judge_file)

        tab_html, status, rows = build_config_tab(label, generations, judgments, label)
        configs_data.append((label, status, rows, tab_html))

    html_content = build_full_html(configs_data)

    with open(output_path, "w") as f:
        f.write(html_content)

    total_gens = sum(len(rows) for _, _, rows, _ in configs_data)
    print(f"Viewer saved to {output_path}")
    print(f"  {len(configs_data)} config(s), {total_gens} total generations")
    for label, status, _, _ in configs_data:
        print(f"  {label}: {status}")


if __name__ == "__main__":
    main()
