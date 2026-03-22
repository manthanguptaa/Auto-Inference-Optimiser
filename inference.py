"""
MLX inference pipeline — THE FILE THE AGENT OPTIMISES.

This is the only file the agent is allowed to modify.
The goal: maximise generation_tps while keeping perplexity below the threshold.

The agent can change anything here: configuration values, sampling strategy,
KV cache settings, prefill chunking, memory management, prompt formatting,
add speculative decoding, change generation loops, etc.
"""

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

# ============================================================
# CONFIGURATION — the agent optimises these values
# ============================================================

# KV cache settings
MAX_KV_SIZE = None          # None=unbounded, or int for rotating cache
KV_BITS = None              # None=full precision, 4 or 8 for quantised KV cache
KV_GROUP_SIZE = 64          # granularity of KV quantisation

# Prefill settings
PREFILL_STEP_SIZE = 512     # tokens per prefill chunk

# Sampling settings
TEMP = 0.0                  # 0.0=argmax (fastest), higher=more random
TOP_P = 1.0                 # nucleus sampling threshold
TOP_K = 0                   # 0=disabled, else restrict to top-K tokens
MIN_P = 0.0                 # minimum probability relative to top token
REPETITION_PENALTY = 1.0    # 1.0=disabled
REPETITION_CONTEXT_SIZE = 20

# Memory settings
METAL_CACHE_LIMIT = None    # None=default, or bytes for mx.metal.set_cache_limit()

# Generation settings
MAX_TOKENS = 256            # max tokens to generate per prompt


def setup_memory():
    """Configure Metal memory settings. Called once before generation."""
    if METAL_CACHE_LIMIT is not None:
        mx.metal.set_cache_limit(METAL_CACHE_LIMIT)


def build_sampler():
    """Build the token sampler from current config."""
    return make_sampler(
        temp=TEMP,
        top_p=TOP_P,
    )


def format_prompt(tokenizer, prompt: str) -> str:
    """Format raw prompt using the model's chat template."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )


def generate_text(model, tokenizer, prompt: str) -> dict:
    """
    Generate text from a prompt and return results with metrics.

    This is the function that prepare.py calls for benchmarking.
    The agent can rewrite this entirely — the only contract is:
      Input:  model, tokenizer, prompt (str)
      Output: dict with keys: text, generation_tps, prompt_tps,
              peak_memory_gb, generation_tokens, prompt_tokens

    Args:
        model: Loaded MLX model
        tokenizer: Loaded tokenizer
        prompt: Raw user prompt string

    Returns:
        dict with generation results and performance metrics
    """
    setup_memory()

    formatted = format_prompt(tokenizer, prompt)
    sampler = build_sampler()

    output_text = ""
    final_resp = None

    for resp in stream_generate(
        model,
        tokenizer,
        formatted,
        max_tokens=MAX_TOKENS,
        sampler=sampler,
        max_kv_size=MAX_KV_SIZE,
        kv_bits=KV_BITS,
        kv_group_size=KV_GROUP_SIZE,
        prefill_step_size=PREFILL_STEP_SIZE,
    ):
        output_text += resp.text
        final_resp = resp

    if final_resp is None:
        return {
            "text": "",
            "generation_tps": 0.0,
            "prompt_tps": 0.0,
            "peak_memory_gb": 0.0,
            "generation_tokens": 0,
            "prompt_tokens": 0,
        }

    return {
        "text": output_text,
        "generation_tps": final_resp.generation_tps,
        "prompt_tps": final_resp.prompt_tps,
        "peak_memory_gb": final_resp.peak_memory,
        "generation_tokens": final_resp.generation_tokens,
        "prompt_tokens": final_resp.prompt_tokens,
    }
