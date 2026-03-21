"""MLX inference pipeline — optimised for Qwen2.5-3B-Instruct-4bit."""

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

_sampler = make_sampler(temp=0.0, top_p=1.0)
_cache_set = False


def generate_text(model, tokenizer, prompt: str) -> dict:
    global _cache_set
    if not _cache_set:
        mx.set_cache_limit(2 * 1024 * 1024 * 1024)
        _cache_set = True

    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True
    )

    tokens = []
    final = None
    for resp in stream_generate(
        model, tokenizer, formatted,
        max_tokens=256, sampler=_sampler,
    ):
        tokens.append(resp.text)
        final = resp

    if final is None:
        return {"text": "", "generation_tps": 0.0, "prompt_tps": 0.0,
                "peak_memory_gb": 0.0, "generation_tokens": 0, "prompt_tokens": 0}

    return {
        "text": "".join(tokens),
        "generation_tps": final.generation_tps,
        "prompt_tps": final.prompt_tps,
        "peak_memory_gb": final.peak_memory,
        "generation_tokens": final.generation_tokens,
        "prompt_tokens": final.prompt_tokens,
    }
