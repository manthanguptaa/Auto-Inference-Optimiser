"""
Evaluation and benchmarking infrastructure for Auto-Inference-Optimiser.

THIS FILE IS READ-ONLY. The agent must NOT modify this file.
It defines the fixed evaluation protocol that all inference experiments are measured against.

The agent optimises inference.py to maximise generation_tps (tokens per second)
while keeping output quality above a perplexity threshold.
"""

import json
import time
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

# ============================================================
# Constants — fixed evaluation protocol
# ============================================================

# Model used for all benchmarks (small enough for any MacBook)
BENCHMARK_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# Generation budget per prompt
MAX_TOKENS = 256

# Number of benchmark runs to average over
NUM_RUNS = 3

# Warmup runs before measurement (first run compiles Metal kernels)
WARMUP_RUNS = 1

# Quality gate: if perplexity exceeds this, the experiment fails
# (prevents the agent from optimising speed by destroying output quality)
PERPLEXITY_THRESHOLD = 50.0

# Benchmark prompts — diverse tasks to test different inference patterns
BENCHMARK_PROMPTS = [
    # Short prompt, long generation (tests decode speed)
    "Write a detailed explanation of how transformers work in neural networks.",

    # Long prompt, short generation (tests prefill speed)
    (
        "The following is a technical discussion about compiler optimization passes. "
        "LLVM uses a series of analysis and transformation passes to optimize intermediate "
        "representation (IR) code. Key passes include: dead code elimination (DCE), which "
        "removes instructions whose results are never used; constant folding, which evaluates "
        "constant expressions at compile time; loop-invariant code motion (LICM), which moves "
        "computations that don't change across loop iterations outside the loop; common "
        "subexpression elimination (CSE), which identifies and reuses previously computed "
        "values; and inlining, which replaces function calls with the function body to reduce "
        "call overhead. The pass manager schedules these passes in an order that maximizes "
        "optimization opportunities while minimizing compile time. Modern compilers also use "
        "profile-guided optimization (PGO) to make data-driven decisions about which code "
        "paths to optimize most aggressively.\n\n"
        "Summarize the key optimization passes in one sentence."
    ),

    # Reasoning task (tests sustained generation quality)
    "Solve step by step: A train travels from city A to city B at 60 km/h and returns at 40 km/h. What is the average speed for the round trip?",

    # Creative task (tests sampling path performance)
    "Write a short poem about silicon chips and electricity.",

    # Code generation (tests structured output)
    "Write a Python function to find the longest common subsequence of two strings.",
]


def load_benchmark_model():
    """Load the fixed benchmark model. Returns (model, tokenizer)."""
    print(f"Loading benchmark model: {BENCHMARK_MODEL}")
    model, tokenizer = load(BENCHMARK_MODEL)
    print("Model loaded.")
    return model, tokenizer


def compute_perplexity(model, tokenizer, text: str) -> float:
    """
    Compute perplexity of generated text using the model.
    Lower perplexity = more confident/coherent output.
    """
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return float("inf")

    token_array = mx.array([tokens])
    logits = model(token_array)

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = token_array[:, 1:]

    # Cross-entropy loss (returns per-token nats)
    import mlx.nn as nn
    per_token_loss = nn.losses.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
    )

    avg_loss = mx.mean(per_token_loss).item()
    perplexity = float(mx.exp(mx.array(avg_loss)).item())

    return perplexity


def run_single_benchmark(generate_fn, model, tokenizer, prompt: str) -> dict:
    """
    Run a single benchmark: call the generate function and measure metrics.

    Args:
        generate_fn: The inference function from inference.py
                     Signature: generate_fn(model, tokenizer, prompt) -> dict
                     Must return dict with at least: text, generation_tps, prompt_tps, peak_memory_gb
        model: The loaded MLX model
        tokenizer: The loaded tokenizer
        prompt: The benchmark prompt

    Returns:
        dict with all metrics
    """
    start = time.perf_counter()
    result = generate_fn(model, tokenizer, prompt)
    wall_time = time.perf_counter() - start

    # Compute quality metric
    generated_text = result.get("text", "")
    if generated_text.strip():
        perplexity = compute_perplexity(model, tokenizer, generated_text)
    else:
        perplexity = float("inf")

    return {
        "prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
        "generation_tps": result.get("generation_tps", 0.0),
        "prompt_tps": result.get("prompt_tps", 0.0),
        "peak_memory_gb": result.get("peak_memory_gb", 0.0),
        "wall_time_s": wall_time,
        "generation_tokens": result.get("generation_tokens", 0),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "perplexity": perplexity,
        "text_length": len(generated_text),
    }


def evaluate(generate_fn) -> dict:
    """
    Full evaluation protocol. This is the main entry point.

    Args:
        generate_fn: The inference function from inference.py
                     Signature: generate_fn(model, tokenizer, prompt) -> dict

    Returns:
        dict with aggregated metrics and per-prompt results
    """
    model, tokenizer = load_benchmark_model()

    # Warmup
    print(f"Running {WARMUP_RUNS} warmup run(s)...")
    warmup_prompt = "Hello, how are you?"
    for _ in range(WARMUP_RUNS):
        generate_fn(model, tokenizer, warmup_prompt)

    # Benchmark
    all_results = []
    for run_idx in range(NUM_RUNS):
        print(f"\n--- Run {run_idx + 1}/{NUM_RUNS} ---")
        run_results = []
        for prompt in BENCHMARK_PROMPTS:
            result = run_single_benchmark(generate_fn, model, tokenizer, prompt)
            run_results.append(result)
            print(
                f"  gen_tps={result['generation_tps']:.1f}  "
                f"prompt_tps={result['prompt_tps']:.1f}  "
                f"mem={result['peak_memory_gb']:.2f}GB  "
                f"ppl={result['perplexity']:.1f}"
            )
        all_results.append(run_results)

    # Aggregate across runs
    num_prompts = len(BENCHMARK_PROMPTS)
    avg_gen_tps = sum(
        r["generation_tps"]
        for run in all_results
        for r in run
    ) / (NUM_RUNS * num_prompts)

    avg_prompt_tps = sum(
        r["prompt_tps"]
        for run in all_results
        for r in run
    ) / (NUM_RUNS * num_prompts)

    avg_peak_mem = sum(
        r["peak_memory_gb"]
        for run in all_results
        for r in run
    ) / (NUM_RUNS * num_prompts)

    avg_perplexity = sum(
        r["perplexity"]
        for run in all_results
        for r in run
        if r["perplexity"] != float("inf")
    )
    valid_ppl_count = sum(
        1 for run in all_results for r in run
        if r["perplexity"] != float("inf")
    )
    avg_perplexity = avg_perplexity / valid_ppl_count if valid_ppl_count > 0 else float("inf")

    # Quality gate
    quality_pass = avg_perplexity < PERPLEXITY_THRESHOLD

    summary = {
        "avg_generation_tps": round(avg_gen_tps, 2),
        "avg_prompt_tps": round(avg_prompt_tps, 2),
        "avg_peak_memory_gb": round(avg_peak_mem, 3),
        "avg_perplexity": round(avg_perplexity, 2),
        "quality_pass": quality_pass,
        "num_runs": NUM_RUNS,
        "num_prompts": num_prompts,
    }

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"avg_generation_tps: {summary['avg_generation_tps']}")
    print(f"avg_prompt_tps: {summary['avg_prompt_tps']}")
    print(f"avg_peak_memory_gb: {summary['avg_peak_memory_gb']}")
    print(f"avg_perplexity: {summary['avg_perplexity']}")
    print(f"quality_pass: {summary['quality_pass']}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    # When run directly, import inference.py and evaluate
    from inference import generate_text
    results = evaluate(generate_text)
    print("\nFull results as JSON:")
    print(json.dumps(results, indent=2))
