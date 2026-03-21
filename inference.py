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
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

# ============================================================
# CONFIGURATION — the agent optimises these values
# ============================================================

# KV cache settings
MAX_KV_SIZE = None          # None=unbounded, or int for rotating cache
KV_BITS = None              # None=full precision, 4 or 8 for quantised KV cache
KV_GROUP_SIZE = 64          # granularity of KV quantisation

# Prefill settings
PREFILL_STEP_SIZE = 2048    # tokens per prefill chunk

# Sampling settings
TEMP = 0.0                  # 0.0=argmax (fastest), higher=more random
TOP_P = 1.0                 # nucleus sampling threshold (1.0=disabled with argmax)

# Memory settings
METAL_CACHE_LIMIT = 1024 * 1024 * 1024  # 1GB Metal buffer cache

# Generation settings
MAX_TOKENS = 256            # max tokens to generate per prompt


def generate_text(model, tokenizer, prompt: str) -> dict:
    """
    Generate text using generate_step directly, bypassing stream_generate overhead.
    """
    if METAL_CACHE_LIMIT is not None:
        mx.set_cache_limit(METAL_CACHE_LIMIT)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    if isinstance(formatted, str):
        prompt_tokens = mx.array(tokenizer.encode(formatted))
    else:
        prompt_tokens = mx.array(formatted)

    sampler = make_sampler(temp=TEMP, top_p=TOP_P)

    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = {eos_token_id}
    elif isinstance(eos_token_id, list):
        eos_token_id = set(eos_token_id)
    else:
        eos_token_id = set()

    # Also check for additional stop tokens
    if hasattr(tokenizer, "added_tokens_encoder"):
        for token_str in ["<|endoftext|>", "<|im_end|>", "<|end|>"]:
            if token_str in tokenizer.added_tokens_encoder:
                eos_token_id.add(tokenizer.added_tokens_encoder[token_str])

    generated_tokens = []

    # Time prefill
    prefill_start = time.perf_counter()

    token_generator = generate_step(
        prompt_tokens,
        model,
        max_tokens=MAX_TOKENS,
        sampler=sampler,
        max_kv_size=MAX_KV_SIZE,
        prefill_step_size=PREFILL_STEP_SIZE,
        kv_bits=KV_BITS,
        kv_group_size=KV_GROUP_SIZE,
    )

    # First token marks end of prefill
    first_token, first_logprobs = next(token_generator)
    if isinstance(first_token, mx.array):
        mx.eval(first_token)
        token_val = first_token.item()
    else:
        token_val = int(first_token)
    prefill_time = time.perf_counter() - prefill_start

    if token_val not in eos_token_id:
        generated_tokens.append(token_val)

    # Time generation
    gen_start = time.perf_counter()

    if token_val not in eos_token_id:
        for token, logprobs in token_generator:
            if isinstance(token, mx.array):
                mx.eval(token)
                token_val = token.item()
            else:
                token_val = int(token)
            if token_val in eos_token_id:
                break
            generated_tokens.append(token_val)

    gen_time = time.perf_counter() - gen_start

    # Decode
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
