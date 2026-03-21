"""
MLX inference pipeline — THE FILE THE AGENT OPTIMISES.

This file contains the FULL inference pipeline with inlined internals:
  - KV cache implementation (data structure, update logic)
  - Generate loop (prefill, decode, async pipelining)
  - Sampling strategy
  - Memory management

The agent can modify ANY of these. The model weights are loaded externally
by prepare.py using mlx_lm.load() — the agent cannot change the model
architecture, but can change everything about how inference is performed.

The only contract is the generate_text() function signature.
"""

import functools
import time
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

# ============================================================
# CONFIGURATION
# ============================================================

MAX_TOKENS = 256
PREFILL_STEP_SIZE = 512
METAL_CACHE_LIMIT = 2 * 1024 * 1024 * 1024  # 2GB

# Dedicated GPU stream for generation pipelining
_generation_stream = mx.new_stream(mx.gpu)


# ============================================================
# KV CACHE — the agent can modify the cache data structure
# ============================================================

class KVCache:
    """
    Key-value cache for autoregressive generation.

    The agent can modify this to experiment with:
    - Different allocation strategies (step size, pre-allocation)
    - Quantised caches (store keys/values in lower precision)
    - Sliding window / rotating caches
    - Chunked caches for better memory locality
    """

    step = 256  # Allocation chunk size — grow cache in steps of this

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        """
        Append new keys/values to cache and return full cache.
        Uses in-place updates with pre-allocated buffers.
        """
        prev = self.offset

        # Grow buffer if needed
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @property
    def state(self):
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
        )

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[2]

    def make_mask(self, N, return_array=False, window_size=None):
        if N > 1:
            if return_array or (self.offset + N > (window_size or float("inf"))):
                return _create_causal_mask(N, self.offset, window_size)
            return "causal"
        return None


# ============================================================
# ATTENTION MASK UTILITIES
# ============================================================

def _create_causal_mask(N, offset=0, window_size=None):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    return mask


def _create_attention_mask(h, cache=None):
    N = h.shape[1]
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(N)
    if N == 1:
        return None
    return "causal"


# ============================================================
# ATTENTION — the agent can modify how attention is computed
# ============================================================

def scaled_dot_product_attention(queries, keys, values, scale, mask=None, cache=None):
    """
    Compute scaled dot-product attention.

    The agent can modify this to experiment with:
    - Flash attention variants
    - Sparse attention patterns
    - Linear attention approximations
    - Custom masking strategies
    """
    return mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask=mask
    )


# ============================================================
# MAKE CACHE — builds one KVCache per layer
# ============================================================

def make_cache(model):
    """Build a list of KVCache objects, one per transformer layer."""
    num_layers = len(model.layers)
    return [KVCache() for _ in range(num_layers)]


# ============================================================
# GENERATE LOOP — the agent can modify the full pipeline
# ============================================================

def _generate_step(prompt_tokens, model, cache, max_tokens, sampler):
    """
    Core generation loop with async pipelining.

    The agent can modify:
    - Prefill strategy (chunk size, parallelism)
    - Decode pipelining (how async_eval is used)
    - When to clear cache / manage memory
    - Sampling integration
    - Early stopping logic
    """

    def _prefill(tokens, cache):
        """Process prompt tokens through the model, building up KV cache."""
        with mx.stream(_generation_stream):
            total = len(tokens)
            processed = 0
            while total - processed > 1:
                n = min(PREFILL_STEP_SIZE, total - processed - 1)
                model(tokens[:n][None], cache=cache)
                mx.eval([c.state for c in cache])
                tokens = tokens[n:]
                processed += n
                mx.clear_cache()
            return tokens  # remaining (last) token

    def _decode_one(input_token, cache):
        """Single decode step: run model, sample next token."""
        with mx.stream(_generation_stream):
            logits = model(input_token[None], cache=cache)
            logits = logits[:, -1, :]
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            y = sampler(logprobs)
            return y, logprobs.squeeze(0)

    # === Prefill ===
    remaining = _prefill(prompt_tokens, cache)

    # First decode (last prompt token)
    y, logprobs = _decode_one(remaining, cache)
    mx.async_eval(y, logprobs)
    mx.eval(y)

    # === Decode loop with async pipelining ===
    n = 0
    while True:
        # Start computing next token while we yield current
        if n != max_tokens:
            next_y, next_logprobs = _decode_one(y, cache)
            mx.async_eval(next_y, next_logprobs)

        if n == max_tokens:
            break

        yield y.item(), logprobs

        # Periodic memory cleanup
        if n % 256 == 0:
            mx.clear_cache()

        y, logprobs = next_y, next_logprobs
        n += 1


# ============================================================
# ENTRY POINT — the function prepare.py calls
# ============================================================

def generate_text(model, tokenizer, prompt: str) -> dict:
    """
    Generate text and return results with metrics.

    This is the function that prepare.py calls for benchmarking.
    The agent can rewrite anything above — the only contract is:
      Input:  model, tokenizer, prompt (str)
      Output: dict with keys: text, generation_tps, prompt_tps,
              peak_memory_gb, generation_tokens, prompt_tokens
    """
    mx.set_cache_limit(METAL_CACHE_LIMIT)

    # Ensure model weights are materialized
    if not getattr(model, "_weights_ready", False):
        mx.eval(model.parameters())
        model._weights_ready = True

    # Tokenize
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if isinstance(formatted, str):
        prompt_tokens = mx.array(tokenizer.encode(formatted))
    else:
        prompt_tokens = mx.array(formatted)

    # Build EOS set
    eos = tokenizer.eos_token_id
    eos_set = {eos} if isinstance(eos, int) else set(eos) if isinstance(eos, list) else set()
    if hasattr(tokenizer, "added_tokens_encoder"):
        for tok in ["<|endoftext|>", "<|im_end|>", "<|end|>"]:
            if tok in tokenizer.added_tokens_encoder:
                eos_set.add(tokenizer.added_tokens_encoder[tok])

    # Sampler — argmax for deterministic, fastest path
    sampler = lambda x: mx.argmax(x, axis=-1)

    # Build KV cache
    cache = make_cache(model)

    # === Prefill timing ===
    prefill_start = time.perf_counter()
    gen = _generate_step(prompt_tokens, model, cache, MAX_TOKENS, sampler)
    first_token, _ = next(gen)
    prefill_time = time.perf_counter() - prefill_start

    generated_tokens = []
    if first_token not in eos_set:
        generated_tokens.append(first_token)

    # === Decode timing ===
    gen_start = time.perf_counter()
    if first_token not in eos_set:
        for token_val, _ in gen:
            if token_val in eos_set:
                break
            generated_tokens.append(token_val)
    gen_time = time.perf_counter() - gen_start

    # Results
    text = tokenizer.decode(generated_tokens)
    num_prompt = len(prompt_tokens)
    num_generated = len(generated_tokens)

    return {
        "text": text,
        "generation_tps": num_generated / gen_time if gen_time > 0 else 0,
        "prompt_tps": num_prompt / prefill_time if prefill_time > 0 else 0,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "generation_tokens": num_generated,
        "prompt_tokens": num_prompt,
    }
