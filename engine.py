"""
KV Cache Masking Engine
=======================
Core inference logic for KV cache ablation experiments.
No web framework dependencies. Importable by scripts and servers alike.

Usage:
    from engine import KVEngine
    engine = KVEngine()
    engine.load_model("meta-llama/Llama-3.1-8B-Instruct")
    tokens = engine.tokenize(raw_text)
    cache = engine.forward(input_ids)
    masked = KVEngine.mask_kv_cache(cache, positions, k_scale=0.0)
    output = engine.generate_from_cache(input_ids, masked)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

DEVICE = "cuda"


class KVEngine:
    """Manages model loading and KV cache masking operations."""

    def __init__(self, device=DEVICE):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.current_model_id = None
        self.eos_ids = set()
        self._special_token_ids = None

    def load_model(self, model_id: str):
        """Load a model by HuggingFace ID or local path. Unloads any previously loaded model."""
        if model_id == self.current_model_id:
            return

        self.unload()
        self._special_token_ids = None

        print(f"Loading {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map=self.device
        )
        self.model.eval()

        # Build EOS set from tokenizer
        self.eos_ids = {self.tokenizer.eos_token_id}
        for name, tid in self.tokenizer.get_added_vocab().items():
            nl = name.lower()
            if "eot_id" in nl or "end_of_text" in nl or "endoftext" in nl:
                self.eos_ids.add(tid)

        print(f"Loaded {model_id} (eos_ids={self.eos_ids})")
        self.current_model_id = model_id

    def unload(self):
        """Free model from GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
        self.tokenizer = None
        self.current_model_id = None
        self.eos_ids = set()
        self._special_token_ids = None

    @property
    def special_token_ids(self) -> set:
        """Set of token IDs that are special (e.g., <|...|> tokens)."""
        if self._special_token_ids is None:
            self._special_token_ids = set()
            if self.tokenizer is not None:
                for name, tid in self.tokenizer.get_added_vocab().items():
                    if name.startswith("<|"):
                        self._special_token_ids.add(tid)
        return self._special_token_ids

    def tokenize(self, raw_text: str) -> list[dict]:
        """Tokenize raw text and return token info dicts.

        Returns:
            List of {"pos": int, "id": int, "text": str, "is_special": bool}
        """
        input_ids = self.tokenizer.encode(raw_text, add_special_tokens=False)
        special = self.special_token_ids
        tokens = []
        for i, tid in enumerate(input_ids):
            text = self.tokenizer.decode([tid])
            tokens.append({
                "pos": i,
                "id": tid,
                "text": text,
                "is_special": tid in special,
            })
        return tokens

    def forward(self, input_ids: torch.Tensor):
        """Run a forward pass and return KV cache as list of (K, V) tuples."""
        with torch.inference_mode():
            outputs = self.model(input_ids, use_cache=True)
            pkv = outputs.past_key_values
            # Convert DynamicCache to list of (K, V) tuples for consistent
            # downstream use by mask_kv_cache / clone_kv / generate_from_cache
            return [(layer.keys, layer.values) for layer in pkv.layers]

    @staticmethod
    def mask_kv_cache(past_key_values, positions_to_mask, k_scale=0.0):
        """Mask KV cache entries at specified positions.

        V is always zeroed at masked positions (no direct content leakage).
        K is scaled by k_scale (0=zeroed, 1=original, etc.).
        """
        masked = []
        for layer_kv in past_key_values:
            k, v = layer_kv[0].clone(), layer_kv[1].clone()
            if positions_to_mask:
                k[:, :, positions_to_mask, :] *= k_scale
                v[:, :, positions_to_mask, :] = 0.0
            masked.append((k, v))
        return masked

    @staticmethod
    def clone_kv(past_key_values):
        """Deep-clone a KV cache (list of (K, V) tuples)."""
        return [(lkv[0].clone(), lkv[1].clone()) for lkv in past_key_values]

    def generate_from_cache(self, input_ids, kv_list, max_new_tokens=50,
                            attn_mask=None):
        """Generate text from a (possibly modified) KV cache.

        Args:
            input_ids: Full input token IDs tensor [1, seq_len].
            kv_list: List of (K, V) tuples per layer.
            max_new_tokens: Maximum tokens to generate.
            attn_mask: Optional attention mask tensor [1, seq_len] where 0
                       means the position is excluded from attention.

        Returns:
            Generated text string.
        """
        seq_len = input_ids.shape[1]

        # Build DynamicCache from kv_list, excluding the last position
        # (it will be re-computed via the forward pass below)
        rerun_cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(kv_list):
            rerun_cache.update(k[:, :, :-1, :], v[:, :, :-1, :], layer_idx)

        # Re-run the last token through the model with the modified cache
        fwd_kwargs = dict(
            input_ids=input_ids[:, -1:],
            past_key_values=rerun_cache,
            position_ids=torch.tensor([[seq_len - 1]], device=self.device),
            use_cache=True,
        )
        if attn_mask is not None:
            fwd_kwargs["attention_mask"] = attn_mask

        with torch.inference_mode():
            out = self.model(**fwd_kwargs)

        generated_ids = []
        cache = out.past_key_values
        next_token = out.logits[0, -1, :].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
        generated_ids.append(next_token.item())

        if next_token.item() in self.eos_ids:
            return self.tokenizer.decode(generated_ids, skip_special_tokens=False)

        current_id = next_token
        current_mask = attn_mask

        for _ in range(max_new_tokens - 1):
            if current_id.item() in self.eos_ids:
                break

            if current_mask is not None:
                current_mask = torch.cat(
                    [current_mask, torch.ones(1, 1, device=self.device,
                                              dtype=current_mask.dtype)],
                    dim=1,
                )

            with torch.inference_mode():
                cache_len = cache.get_seq_length()
                gen_kwargs = dict(
                    input_ids=current_id,
                    past_key_values=cache,
                    position_ids=torch.tensor([[cache_len]], device=self.device),
                    use_cache=True,
                )
                if current_mask is not None:
                    gen_kwargs["attention_mask"] = current_mask
                out = self.model(**gen_kwargs)

            cache = out.past_key_values
            next_token = out.logits[0, -1, :].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
            generated_ids.append(next_token.item())
            current_id = next_token

        return self.tokenizer.decode(generated_ids, skip_special_tokens=False)

    def generate_batched(self, input_ids, kv_lists, max_new_tokens=20):
        """Generate from multiple masked KV caches in parallel (batched).

        Args:
            input_ids: Full input token IDs tensor [1, seq_len].
            kv_lists: List of N KV caches, each a list of (K, V) tuples per layer.
                      K, V shapes: [1, num_heads, seq_len, head_dim].
            max_new_tokens: Maximum tokens to generate per sequence.

        Returns:
            List of N generated text strings.
        """
        batch_size = len(kv_lists)
        seq_len = input_ids.shape[1]
        num_layers = len(kv_lists[0])

        # Stack KV caches along batch dimension: [1,h,s,d] -> [N,h,s,d]
        # Exclude last position (will be recomputed)
        rerun_cache = DynamicCache()
        for layer_idx in range(num_layers):
            k_stack = torch.cat([kv_lists[b][layer_idx][0][:, :, :-1, :] for b in range(batch_size)], dim=0)
            v_stack = torch.cat([kv_lists[b][layer_idx][1][:, :, :-1, :] for b in range(batch_size)], dim=0)
            rerun_cache.update(k_stack, v_stack, layer_idx)

        # Batch the last token: [N, 1]
        batched_ids = input_ids[:, -1:].expand(batch_size, -1)
        pos_ids = torch.tensor([[seq_len - 1]], device=self.device).expand(batch_size, -1)

        with torch.inference_mode():
            out = self.model(
                input_ids=batched_ids,
                past_key_values=rerun_cache,
                position_ids=pos_ids,
                use_cache=True,
            )

        cache = out.past_key_values
        # [N, 1]
        next_tokens = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Track generated IDs and active mask per sequence
        all_generated = [[t.item()] for t in next_tokens]
        active = [t.item() not in self.eos_ids for t in next_tokens]

        for step in range(max_new_tokens - 1):
            if not any(active):
                break

            cache_len = cache.get_seq_length()
            pos_ids = torch.tensor([[cache_len]], device=self.device).expand(batch_size, -1)

            with torch.inference_mode():
                out = self.model(
                    input_ids=next_tokens,
                    past_key_values=cache,
                    position_ids=pos_ids,
                    use_cache=True,
                )

            cache = out.past_key_values
            next_tokens = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            for b in range(batch_size):
                if active[b]:
                    tok = next_tokens[b].item()
                    all_generated[b].append(tok)
                    if tok in self.eos_ids:
                        active[b] = False

        return [
            self.tokenizer.decode(ids, skip_special_tokens=False)
            for ids in all_generated
        ]
