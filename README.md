# KV Cache Masking

Mechanistic interpretability tool for probing information compression in transformer KV cache entries. Mask all KV cache entries except a few token positions after a full forward pass, then generate from the modified cache to see what information survives in the remaining positions.

## Core Finding

With all tokens masked except 4-6 assistant header tokens, models still produce correct factual answers. The information from the full input context is compressed into the KV entries at the unmasked positions during the forward pass (where they attend to the full context via causal attention).

**Counterfactual validation**: prompts like "The capital of France is Tokyo. What is the capital of France?" produce "Tokyo" (not Paris) after masking, confirming the model reads information genuinely compressed into the unmasked KV entries rather than relying on prior knowledge.

### K-Scale Parameter

The `k_scale` parameter controls how much attention budget flows to masked (zero-content) positions:

| k_scale | Mechanism | Effect |
|---------|-----------|--------|
| 0 | Q*K=0, so softmax gives uniform weight | Attention budget diluted across all positions |
| 1 | Original K preserved, V zeroed | Attention follows original routing but "wastes" budget on zero-content positions |
| -inf | Attention mask excludes masked positions | All attention concentrates on unmasked tokens — catastrophic (0% accuracy) |

Dilution (k=0) works better than concentration (k=-inf) because concentrating all attention on a few tokens destabilizes the residual stream.

### Results (Llama 3.1 8B Instruct, 300 prompts)

| Unmasked tokens | Overall accuracy (k=0) | Factual recall accuracy (k=0) |
|-----------------|------------------------|-------------------------------|
| Last 4 (assistant header only) | 11.4% | 18.4% |
| Last 5 (+user EOT) | 24.6% | 55.1% |
| Last 6 (+user's last token) | 39.2% | 75.5% |

Accuracy is evaluated by an LLM judge across 6 prompt categories: factual recall, counterfactual, multi-step reasoning, long context, multi-turn, and instruction following.

## Components

| File | Description |
|------|-------------|
| `engine.py` | Core KV masking engine — model loading, forward pass, cache masking, generation |
| `server.py` | FastAPI web server wrapping the engine, serves the interactive UI |
| `index.html` | Interactive web app for experimenting with KV cache masking |
| `prompts.py` | 300 evaluation prompts across 6 categories (50 each) |
| `eval_harness.py` | Automated generation sweep across k-scale values and unmasked token counts |
| `judge.py` | LLM judge (Gemini) that classifies outputs as CORRECT / PARTIALLY_CORRECT / COHERENT_UNRELATED / NONSENSE |
| `analyze.py` | Aggregates judgments into summary statistics and tables |
| `plot_results.py` | Generates publication-quality plots from evaluation results |
| `build_viewer.py` | Builds a self-contained HTML viewer with all results, verdicts, and plots |
| `findings.md` | Detailed research findings and experimental roadmap |

## Setup

Requires Python 3.11+, CUDA GPU, and a HuggingFace model.

```bash
pip install torch transformers fastapi uvicorn
```

For the evaluation pipeline:
```bash
pip install python-dotenv google-genai matplotlib
```

## Usage

### Interactive Web App

```bash
# Default model (Llama 3.1 8B Instruct)
python server.py

# Any HuggingFace model or local path
python server.py --model Qwen/Qwen2.5-7B-Instruct
python server.py --model /path/to/local/model
```

Open `http://localhost:8000`. The UI lets you:
- Enter any prompt (auto-applies the model's chat template)
- Click tokens to mask/unmask them
- Adjust k_scale and generate to compare normal vs masked output
- Switch models at runtime via the model input field

### Programmatic Use

```python
from engine import KVEngine

engine = KVEngine()
engine.load_model("meta-llama/Llama-3.1-8B-Instruct")

# Tokenize and forward pass
tokens = engine.tokenize(raw_text)
input_ids = torch.tensor([[t["id"] for t in tokens]], device="cuda")
cache = engine.forward(input_ids)

# Mask everything except the last 4 tokens
seq_len = input_ids.shape[1]
positions_to_mask = list(range(seq_len - 4))
masked_cache = KVEngine.mask_kv_cache(cache, positions_to_mask, k_scale=0.0)

# Generate from masked cache
output = engine.generate_from_cache(input_ids, masked_cache)
```

### Evaluation Pipeline

Run the full evaluation pipeline (requires a CUDA GPU and a `GOOGLE_API_KEY` for Gemini judging):

```bash
# 1. Generate: baseline + masked outputs for 300 prompts
python eval_harness.py

# 2. Judge: classify outputs via Gemini
python judge.py

# 3. Analyze: print summary tables
python analyze.py --dir llama_eval_results/n4

# 4. Plot: generate visualizations
python plot_results.py

# 5. Build HTML viewer with all results
python build_viewer.py
```

The eval harness and judge are crash-resumable — they pick up where they left off.

## Project Structure

```
├── engine.py              # Core KV masking engine
├── server.py              # Web server (FastAPI)
├── index.html             # Interactive UI
├── prompts.py             # 300 eval prompts
├── eval_harness.py        # Generation sweep
├── judge.py               # LLM judge (Gemini)
├── analyze.py             # Summary statistics
├── plot_results.py        # Plot generation
├── build_viewer.py        # HTML viewer builder
├── findings.md            # Research findings
└── llama_eval_results/    # Evaluation outputs
    ├── n4/                # 4 unmasked tokens (assistant header only)
    ├── n5/                # 5 unmasked tokens (+EOT)
    ├── n6/                # 6 unmasked tokens (+last user token)
    └── plots/             # Generated plots
```
