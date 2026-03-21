"""
MLX inference pipeline — THE FILE THE AGENT OPTIMISES.

This is the only file the agent is allowed to modify.
The goal: maximise generation_tps while keeping perplexity below the threshold.

The agent can change anything here: configuration values, sampling strategy,
KV cache settings, prefill chunking, memory management, prompt formatting,
add speculative decoding, change generation loops, etc.
"""

import time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

# ============================================================
# CONFIGURATION — the agent optimises these values
# ============================================================

# KV cache settings
MAX_KV_SIZE = None          # None=unbounded, or int for rotating cache

# Prefill settings
PREFILL_STEP_SIZE = 2048    # tokens per prefill chunk

# Memory settings
METAL_CACHE_LIMIT = 1024 * 1024 * 1024  # 1GB Metal buffer cache

# Generation settings
MAX_TOKENS = 256            # max tokens to generate per prompt

# Raw argmax sampler — avoids make_sampler overhead and temp/top_p checks
_argmax_sampler = lambda x: mx.argmax(x, axis=-1)


def generate_text(model, tokenizer, prompt: str) -> dict:
    """
    Generate text using generate_step with streamlined decode loop.

    generate_step yields (int, logprobs) — no need for isinstance checks.
    Uses raw argmax lambda instead of make_sampler for zero overhead.
    """
    mx.set_cache_limit(METAL_CACHE_LIMIT)

    if not getattr(model, "_weights_ready", False):
        mx.eval(model.parameters())
        model._weights_ready = True

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    if isinstance(formatted, str):
        prompt_tokens = mx.array(tokenizer.encode(formatted))
    else:
        prompt_tokens = mx.array(formatted)

    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = {eos_token_id}
    elif isinstance(eos_token_id, list):
        eos_token_id = set(eos_token_id)
    else:
        eos_token_id = set()

    if hasattr(tokenizer, "added_tokens_encoder"):
        for token_str in ["<|endoftext|>", "<|im_end|>", "<|end|>"]:
            if token_str in tokenizer.added_tokens_encoder:
                eos_token_id.add(tokenizer.added_tokens_encoder[token_str])

    prompt_cache = make_prompt_cache(model, max_kv_size=MAX_KV_SIZE)
    generated_tokens = []

    # Prefill + first token
    prefill_start = time.perf_counter()

    token_generator = generate_step(
        prompt_tokens,
        model,
        max_tokens=MAX_TOKENS,
        sampler=_argmax_sampler,
        prompt_cache=prompt_cache,
        prefill_step_size=PREFILL_STEP_SIZE,
    )

    token_val, _ = next(token_generator)
    prefill_time = time.perf_counter() - prefill_start

    if token_val not in eos_token_id:
        generated_tokens.append(token_val)

    # Decode loop — generate_step yields ints, tight loop
    gen_start = time.perf_counter()

    if token_val not in eos_token_id:
        for token_val, _ in token_generator:
            if token_val in eos_token_id:
                break
            generated_tokens.append(token_val)

    gen_time = time.perf_counter() - gen_start

    text = tokenizer.decode(generated_tokens)

    num_prompt = len(prompt_tokens)
    num_generated = len(generated_tokens)
    prompt_tps = num_prompt / prefill_time if prefill_time > 0 else 0
    gen_tps = num_generated / gen_time if gen_time > 0 else 0
    peak_mem = mx.get_peak_memory() / 1e9

    return {
        "text": text,
        "generation_tps": gen_tps,
        "prompt_tps": prompt_tps,
        "peak_memory_gb": peak_mem,
        "generation_tokens": num_generated,
        "prompt_tokens": num_prompt,
    }
