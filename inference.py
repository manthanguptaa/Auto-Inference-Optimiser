"""
MLX inference pipeline — THE FILE THE AGENT OPTIMISES.

This is the only file the agent is allowed to modify.
The goal: maximise generation_tps while keeping perplexity below the threshold.
"""

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

# ============================================================
# CONFIGURATION
# ============================================================

MAX_TOKENS = 256
PREFILL_STEP_SIZE = 2048

# Singleton sampler — argmax (temp=0) is fastest
_sampler = make_sampler(temp=0.0)


def generate_text(model, tokenizer, prompt: str) -> dict:
    """
    Generate text from a prompt and return results with metrics.

    Input:  model, tokenizer, prompt (str)
    Output: dict with keys: text, generation_tps, prompt_tps,
            peak_memory_gb, generation_tokens, prompt_tokens
    """
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )

    output_text = ""
    final_resp = None

    for resp in stream_generate(
        model,
        tokenizer,
        formatted,
        max_tokens=MAX_TOKENS,
        sampler=_sampler,
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
