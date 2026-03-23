"""
KV Cache Masking — Web Server
==============================
FastAPI endpoints wrapping engine.py. Serves index.html.

Run:
    python server.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import torch
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import asyncio

from engine import KVEngine

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


# ── Pydantic models ──────────────────────────────────────────────────────────

class TokenizeRequest(BaseModel):
    raw_text: str

class TokenInfo(BaseModel):
    pos: int
    id: int
    text: str
    is_special: bool

class TokenizeResponse(BaseModel):
    tokens: list[TokenInfo]
    rebuilt_text: str

class GenerateRequest(BaseModel):
    raw_text: str
    masked_positions: list[int] = []
    max_new_tokens: int = 50
    k_scale: float = 0.0
    use_attention_mask: bool = False

class GenerateResponse(BaseModel):
    normal_output: str
    masked_output: str

class SwitchModelRequest(BaseModel):
    model_id: str


# ── App setup ────────────────────────────────────────────────────────────────

engine = KVEngine()
inference_lock = asyncio.Lock()
startup_model = DEFAULT_MODEL


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load_model(startup_model)
    yield
    engine.unload()


app = FastAPI(lifespan=lifespan)

INDEX_HTML = Path(__file__).parent / "index.html"


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/models")
async def list_models():
    return {
        "current": engine.current_model_id,
    }


@app.post("/switch_model")
async def switch_model_endpoint(req: SwitchModelRequest):
    async with inference_lock:
        engine.load_model(req.model_id)
        return {
            "model_id": engine.current_model_id,
        }


@app.get("/default_prompt")
async def default_prompt():
    msgs = [{"role": "user", "content": "What is the capital of France?"}]
    prompt = engine.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    return {
        "prompt": prompt,
        "model": engine.current_model_id,
    }


@app.get("/chat_template")
async def chat_template():
    """Return the chat template with a placeholder, tokenized, plus special token list."""
    sentinel = "__USER_MSG__"
    msgs = [{"role": "user", "content": sentinel}]
    raw = engine.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    token_dicts = engine.tokenize(raw)
    # Collect unique special tokens for the button palette
    special_tokens = []
    seen = set()
    for t in token_dicts:
        if t["is_special"] and t["text"] not in seen:
            special_tokens.append(t["text"])
            seen.add(t["text"])
    return {
        "raw": raw,
        "tokens": token_dicts,
        "special_tokens": special_tokens,
        "sentinel": sentinel,
    }


@app.post("/tokenize", response_model=TokenizeResponse)
async def tokenize_endpoint(req: TokenizeRequest):
    token_dicts = engine.tokenize(req.raw_text)
    tokens = [TokenInfo(**t) for t in token_dicts]
    rebuilt = "".join(t.text for t in tokens)
    return TokenizeResponse(tokens=tokens, rebuilt_text=rebuilt)


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(req: GenerateRequest):
    async with inference_lock:
        token_dicts = engine.tokenize(req.raw_text)
        input_ids_list = [t["id"] for t in token_dicts]
        input_ids = torch.tensor([input_ids_list], device=engine.device)

        cache = engine.forward(input_ids)

        # Masked generation first (clones internally)
        attn_mask = None
        if req.masked_positions:
            if req.use_attention_mask:
                # -inf mode: zero V, leave K untouched, use attention mask
                masked_kv = KVEngine.mask_kv_cache(
                    cache, req.masked_positions, k_scale=1.0
                )
                seq_len = input_ids.shape[1]
                attn_mask = torch.ones(1, seq_len, device=engine.device,
                                       dtype=torch.long)
                for pos in req.masked_positions:
                    attn_mask[0, pos] = 0
            else:
                masked_kv = KVEngine.mask_kv_cache(
                    cache, req.masked_positions, k_scale=req.k_scale
                )
        else:
            masked_kv = KVEngine.clone_kv(cache)
        masked_text = engine.generate_from_cache(
            input_ids, masked_kv, req.max_new_tokens, attn_mask=attn_mask
        )

        # Normal generation (reuse cache directly — not mutated)
        normal_text = engine.generate_from_cache(
            input_ids, cache, req.max_new_tokens
        )

        del cache, masked_kv
        torch.cuda.empty_cache()

        return GenerateResponse(
            normal_output=normal_text, masked_output=masked_text
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV Cache Masking web server")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"HuggingFace model ID or local path (default: {DEFAULT_MODEL})")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    startup_model = args.model
    uvicorn.run("server:app", host=args.host, port=args.port)
