# KV Cache Masking
Tool for probing information compression in transformer KV cache entries. Mask all KV cache entries except a few token positions after a full forward pass, then generate from the modified cache to see what information survives in the remaining positions.


## Setup

Requires CUDA GPU, and a HuggingFace model.

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
python eval/eval_harness.py

# 2. Judge: classify outputs via Gemini
python eval/judge.py

# 3. Analyze: print summary tables
python eval/analyze.py --dir llama_eval_results/n4

# 4. Plot: generate visualizations
python eval/plot_results.py

# 5. Build HTML viewer with all results
python eval/build_viewer.py
```

The eval harness and judge are crash-resumable — they pick up where they left off.

## Project Structure

```
├── engine.py              # Core KV masking engine
├── server.py              # Web server (FastAPI)
├── index.html             # Interactive UI
├── eval/                  # Evaluation pipeline
│   ├── prompts.py         # 300 eval prompts
│   ├── eval_harness.py    # Generation sweep
│   ├── judge.py           # LLM judge (Gemini)
│   ├── analyze.py         # Summary statistics
│   ├── plot_results.py    # Plot generation
│   └── build_viewer.py    # HTML viewer builder
└── llama_eval_results/    # Evaluation outputs
    ├── n4/                # 4 unmasked tokens (assistant header only)
    ├── n5/                # 5 unmasked tokens (+EOT)
    ├── n6/                # 6 unmasked tokens (+last user token)
    └── plots/             # Generated plots
```

